"""
Spiking predictive-coding network with local Adam-style updates
and optional spike-reset dynamics.
"""

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn as nn


@dataclass
class AdamConfig:
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8


class PCLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        adam_cfg: AdamConfig | None = None,
    ):
        super().__init__()

        self.W = nn.Parameter(torch.empty(output_dim, input_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))

        nn.init.xavier_uniform_(self.W)

        self.adam_cfg = adam_cfg or AdamConfig()
        self.step_count = 0

        self.register_buffer("m_W", torch.zeros_like(self.W))
        self.register_buffer("v_W", torch.zeros_like(self.W))
        self.register_buffer("m_b", torch.zeros_like(self.b))
        self.register_buffer("v_b", torch.zeros_like(self.b))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.W.T + self.b

    @torch.no_grad()
    def update_params(
        self,
        error: torch.Tensor,
        pre_activity: torch.Tensor,
        lr: float,
    ) -> None:
        batch_size = error.size(0)

        grad_W = error.T @ pre_activity / batch_size
        grad_b = error.mean(dim=0)

        self.step_count += 1

        self.m_W, self.v_W = self._adam_step(
            self.W, grad_W, self.m_W, self.v_W, self.step_count, lr
        )
        self.m_b, self.v_b = self._adam_step(
            self.b, grad_b, self.m_b, self.v_b, self.step_count, lr
        )

    def _adam_step(
        self,
        param: torch.Tensor,
        grad: torch.Tensor,
        m: torch.Tensor,
        v: torch.Tensor,
        t: int,
        lr: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cfg = self.adam_cfg

        m = cfg.beta1 * m + (1.0 - cfg.beta1) * grad
        v = cfg.beta2 * v + (1.0 - cfg.beta2) * grad.square()

        m_hat = m / (1.0 - cfg.beta1**t)
        v_hat = v / (1.0 - cfg.beta2**t)

        param.add_(lr * m_hat / (torch.sqrt(v_hat) + cfg.eps))

        return m, v


class SpikingPredictiveCodingNetwork(nn.Module):
    def __init__(
        self,
        dims: List[int],
        activation: str = "spike",
        spike_threshold: float = 1.0,
        spike_reset: bool = True,
        reset_beta: float = 1.0,
        v_reset: float = 1.0,
    ):
        super().__init__()

        if len(dims) != 4:
            raise ValueError(f"Expected dims of length 4, got {dims}")

        self.dims = dims
        self.activation = activation

        self.spike_threshold = spike_threshold
        self.spike_reset = spike_reset
        self.reset_beta = reset_beta
        self.v_reset = v_reset

        self.layer1 = PCLayer(dims[0], dims[1])
        self.layer2 = PCLayer(dims[1], dims[2])
        self.layer3 = PCLayer(dims[2], dims[3])

    def activate(self, v: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return torch.relu(v)

        if self.activation == "sigmoid":
            return torch.sigmoid(v)

        if self.activation == "spike":
            return (v > self.spike_threshold).float()

        raise ValueError(f"Unknown activation: {self.activation}")

    def activation_derivative(self, v: torch.Tensor) -> torch.Tensor:
        if self.activation == "relu":
            return (v > 0).float()

        if self.activation == "sigmoid":
            s = torch.sigmoid(v)
            return s * (1.0 - s)

        if self.activation == "spike":
            return self.spike_surrogate_derivative(v)

        raise ValueError(f"Unknown activation: {self.activation}")

    def spike_surrogate_derivative(self, v: torch.Tensor) -> torch.Tensor:
        x = v - self.spike_threshold
        return 1.0 / (torch.pi * (1.0 + (torch.pi * x).square()))

    def apply_spike_reset(
        self,
        v: torch.Tensor,
        spikes_prev: torch.Tensor,
    ) -> torch.Tensor:
        """
        Reset mechanism:

            v[t] = v[t] - beta * S_out[t-1] * v_reset
        """
        if not self.spike_reset:
            return v

        return v - self.reset_beta * spikes_prev * self.v_reset

    @staticmethod
    def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        x = torch.clamp(x, eps, 1.0 - eps)
        return torch.logit(x)

    def feedforward(
        self,
        x0: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        v0 = self.inverse_sigmoid(x0)
        s0 = self.activate(v0)

        v1 = self.layer1(s0)
        s1 = self.activate(v1)

        v2 = self.layer2(s1)
        s2 = self.activate(v2)

        v3 = self.layer3(s2)

        return v1, v2, v3

    def fit_step(
        self,
        x0: torch.Tensor,
        target: torch.Tensor,
        w_rate: float,
        v_rate: float,
        n_step: int,
    ) -> Dict[str, float]:

        v0 = self.inverse_sigmoid(x0)
        s0 = self.activate(v0)

        v1 = self.layer1(s0)
        s1 = self.activate(v1)

        v2 = self.layer2(s1)
        s2 = self.activate(v2)

        v3 = target.clone()

        s1_prev = torch.zeros_like(s1)
        s2_prev = torch.zeros_like(s2)

        for _ in range(n_step):
            if self.activation == "spike":
                v1 = self.apply_spike_reset(v1, s1_prev)
                v2 = self.apply_spike_reset(v2, s2_prev)

            s1 = self.activate(v1)
            s2 = self.activate(v2)

            v1_hat = self.layer1(s0)
            v2_hat = self.layer2(s1)
            v3_hat = self.layer3(s2)

            e1 = v1 - v1_hat
            e2 = v2 - v2_hat
            e3 = v3 - v3_hat

            grad_v1 = -e1 + self.activation_derivative(v1) * (e2 @ self.layer2.W)
            grad_v2 = -e2 + self.activation_derivative(v2) * (e3 @ self.layer3.W)

            v1 = v1 + v_rate * grad_v1
            v2 = v2 + v_rate * grad_v2

            if self.activation == "spike":
                s1_prev = s1.detach()
                s2_prev = s2.detach()

        s1 = self.activate(v1)
        s2 = self.activate(v2)

        v1_hat = self.layer1(s0)
        v2_hat = self.layer2(s1)
        v3_hat = self.layer3(s2)

        e1 = v1 - v1_hat
        e2 = v2 - v2_hat
        e3 = v3 - v3_hat

        self.layer1.update_params(e1, s0, w_rate)
        self.layer2.update_params(e2, s1, w_rate)
        self.layer3.update_params(e3, s2, w_rate)

        return {
            "e1_mse": e1.square().mean(dim=0).sum().item(),
            "e2_mse": e2.square().mean(dim=0).sum().item(),
            "e3_mse": e3.square().mean(dim=0).sum().item(),
        }

    @torch.no_grad()
    def infer(
        self,
        x0: torch.Tensor,
        target: torch.Tensor,
    ) -> Dict[str, torch.Tensor | int]:

        _, _, logits = self.feedforward(x0)

        pred_labels = logits.argmax(dim=1)
        true_labels = target.argmax(dim=1)

        correct = (pred_labels == true_labels).sum().item()

        return {
            "pred_labels": pred_labels,
            "true_labels": true_labels,
            "correct": correct,
            "num_samples": x0.size(0),
        }
                    
    @torch.no_grad()
    def record_dynamics(
        self,
        x0: torch.Tensor,
        target: torch.Tensor,
        v_rate: float,
        n_step: int,
    ) -> Dict[str, list]:

        self.eval()

        v0 = self.inverse_sigmoid(x0)
        s0 = self.activate(v0)

        v1 = self.layer1(s0)
        s1 = self.activate(v1)

        v2 = self.layer2(s1)
        s2 = self.activate(v2)

        v3 = target.clone()

        s1_prev = torch.zeros_like(s1)
        s2_prev = torch.zeros_like(s2)

        records = {
            "input": [],
            "v1": [],
            "v2": [],
            "s1": [],
            "s2": [],
            "output": [],
            "prediction": [],

            "e1": [],
            "e2": [],
            "e3": [],

            "e1_mean": [],
            "e2_mean": [],
            "e3_mean": [],
            "total_error_mean": [],
        }

        for _ in range(n_step):

            if self.activation == "spike":
                v1 = self.apply_spike_reset(v1, s1_prev)
                v2 = self.apply_spike_reset(v2, s2_prev)

            s1 = self.activate(v1)
            s2 = self.activate(v2)

            v1_hat = self.layer1(s0)
            v2_hat = self.layer2(s1)
            v3_hat = self.layer3(s2)

            e1 = v1 - v1_hat
            e2 = v2 - v2_hat
            e3 = v3 - v3_hat

            # Error vectors
            e1_vec = e1.detach().cpu().clone()
            e2_vec = e2.detach().cpu().clone()
            e3_vec = e3.detach().cpu().clone()

            # Mean signed errors
            e1_mean = e1.mean().item()
            e2_mean = e2.mean().item()
            e3_mean = e3.mean().item()

            # MSE-style scalar errors
            e1_mse = e1.square().mean(dim=0).sum().item()
            e2_mse = e2.square().mean(dim=0).sum().item()
            e3_mse = e3.square().mean(dim=0).sum().item()

            total_error = e1_mse + e2_mse + e3_mse

            grad_v1 = (
                -e1
                + self.activation_derivative(v1)
                * (e2 @ self.layer2.W)
            )

            grad_v2 = (
                -e2
                + self.activation_derivative(v2)
                * (e3 @ self.layer3.W)
            )

            v1 = v1 + v_rate * grad_v1
            v2 = v2 + v_rate * grad_v2

            s1 = self.activate(v1)
            s2 = self.activate(v2)

            v3_hat = self.layer3(s2)
            pred_label = v3_hat.argmax(dim=1).item()

            records["input"].append(s0.detach().cpu().clone())
            records["v1"].append(v1.detach().cpu().clone())
            records["v2"].append(v2.detach().cpu().clone())
            records["s1"].append(s1.detach().cpu().clone())
            records["s2"].append(s2.detach().cpu().clone())
            records["output"].append(v3_hat.detach().cpu().clone())
            records["prediction"].append(pred_label)

            records["e1"].append(e1.detach().cpu().clone())
            records["e2"].append(e2.detach().cpu().clone())
            records["e3"].append(e3.detach().cpu().clone())

            records["e1_mean"].append(e1_mse)
            records["e2_mean"].append(e2_mse)
            records["e3_mean"].append(e3_mse)
            records["total_error_mean"].append(total_error)

            if self.activation == "spike":
                s1_prev = s1.detach()
                s2_prev = s2.detach()

        return records