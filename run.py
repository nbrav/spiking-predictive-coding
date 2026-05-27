from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Dict, Tuple

import torch
from torch.utils.data import DataLoader

from datasets.fashion_mnist import (
    get_fashion_mnist_loaders,
    preprocess_fashion_mnist_batch,
)

from datasets.mnist import (
    get_mnist_loaders,
    preprocess_mnist_batch,
)

from datasets.nmnist import (
    get_nmnist_loaders,
    preprocess_nmnist_batch,
)

from datasets.cifar10 import (
    get_cifar10_loaders,
    preprocess_cifar10_batch,
)

from utils.plots import (
    plot_lines,
    plot_prediction_error_summary,
    plot_activity_and_error_gif,
    plot_n_step_training_experiment,
)

from pcn_SpikingPCN import (
    SpikingPredictiveCodingNetwork as Network,
)


# ============================================================
# Config
# ============================================================

@dataclass
class Config:

    # Experiment
    experiment: str = "reset"

    # Dataset
    dataset: str = "mnist"
    batch_size: int = 256

    # Training
    num_epochs: int = 50
    w_rate: float = 1e-3
    v_rate: float = 1e-1
    n_step: int = 100

    # Model
    dims: Tuple[int, ...] = (28 * 28, 600, 600, 10)
    activation: str = "spike"

    # Spiking parameters
    spike_threshold: float = 1.0
    spike_reset: bool = False
    reset_beta: float = 1e-2
    v_reset: float = 1.0

    # Output/system
    output_dir: Path = Path("results")
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# Dataset
# ============================================================

def get_dataset(cfg: Config) -> Tuple[DataLoader, DataLoader, Callable]:

    if cfg.dataset == "mnist":
        train_loader, test_loader = get_mnist_loaders(batch_size=cfg.batch_size)
        preprocess_fn = preprocess_mnist_batch

    elif cfg.dataset == "fashion_mnist":
        train_loader, test_loader = get_fashion_mnist_loaders(batch_size=cfg.batch_size)
        preprocess_fn = preprocess_fashion_mnist_batch

    elif cfg.dataset == "nmnist":
        train_loader, test_loader = get_nmnist_loaders(
            batch_size=cfg.batch_size,
            n_time_bins=100,
        )
        preprocess_fn = preprocess_nmnist_batch

    elif cfg.dataset == "cifar10":
        train_loader, test_loader = get_cifar10_loaders(batch_size=cfg.batch_size)
        preprocess_fn = preprocess_cifar10_batch

    else:
        raise ValueError(f"Unknown dataset: {cfg.dataset}")

    return train_loader, test_loader, preprocess_fn


# ============================================================
# Model
# ============================================================

def build_network(cfg: Config) -> Network:

    return Network(
        dims=list(cfg.dims),
        activation=cfg.activation,

        spike_threshold=cfg.spike_threshold,

        spike_reset=cfg.spike_reset,
        reset_beta=cfg.reset_beta,
        v_reset=cfg.v_reset,

    ).to(cfg.device)


# ============================================================
# Logging
# ============================================================

def init_error_logs() -> Dict[str, list]:

    return {
        "error1": [],
        "error2": [],
        "error3": [],
        "total_error": [],
    }


def init_accuracy_logs() -> Dict[str, list]:

    return {
        "Train accuracy": [],
        "Test accuracy": [],
    }


def merge_logs(all_logs: Dict[str, list], epoch_logs: Dict[str, list]) -> None:

    for key, values in epoch_logs.items():
        all_logs[key].extend(values)


def update_accuracy_logs(
    accuracy_logs: Dict[str, list],
    train_acc: float,
    test_acc: float,
) -> None:

    accuracy_logs["Train accuracy"].append(train_acc * 100.0)
    accuracy_logs["Test accuracy"].append(test_acc * 100.0)


# ============================================================
# Training
# ============================================================

def train_one_epoch(
    network: Network,
    train_loader: DataLoader,
    preprocess_fn: Callable,
    cfg: Config,
) -> Dict[str, list]:

    network.train()

    logs = init_error_logs()

    for x0, y in train_loader:

        x0, y = preprocess_fn(x0, y, cfg.device)

        out = network.fit_step(
            x0,
            y,
            w_rate=cfg.w_rate,
            v_rate=cfg.v_rate,
            n_step=cfg.n_step,
        )

        e1 = out["e1_mse"]
        e2 = out["e2_mse"]
        e3 = out["e3_mse"]

        logs["error1"].append(e1)
        logs["error2"].append(e2)
        logs["error3"].append(e3)

        logs["total_error"].append(e1 + e2 + e3)

    return logs


# ============================================================
# Evaluation
# ============================================================

@torch.no_grad()
def evaluate(
    network: Network,
    data_loader: DataLoader,
    preprocess_fn: Callable,
    cfg: Config,
) -> float:

    network.eval()

    total_correct = 0
    total_samples = 0

    for x0, y in data_loader:

        x0, y = preprocess_fn(x0, y, cfg.device)

        out = network.infer(x0, y)

        total_correct += out["correct"]
        total_samples += out["num_samples"]

    return total_correct / total_samples


# ============================================================
# Visualization
# ============================================================

def visualize_activity_dynamics(
    network: Network,
    test_loader: DataLoader,
    preprocess_fn: Callable,
    cfg: Config,
    sample_id: int = 0,
) -> None:

    x0, y = next(iter(test_loader))

    x0, y = preprocess_fn(x0, y, cfg.device)

    x_sample = x0[sample_id: sample_id + 1]
    y_sample = y[sample_id: sample_id + 1]

    true_label = y_sample.argmax(dim=1).item()

    records = network.record_dynamics(
        x0=x_sample,
        target=y_sample,
        v_rate=cfg.v_rate,
        n_step=cfg.n_step,
    )

    plot_activity_and_error_gif(
        records=records,
        filename=cfg.output_dir / "activity_dynamics",

        image_shape=(28, 28),
        layer1_shape=(20, 30),
        layer2_shape=(20, 30),

        activity_keys=("s1", "s2"),

        true_label=true_label,
        interval=120,
    )


# ============================================================
# Utilities
# ============================================================

def print_config(cfg: Config) -> None:

    print("\nExperiment config")
    print("-" * 60)

    for key, value in vars(cfg).items():
        print(f"{key}: {value}")


# ============================================================
# Main experiment
# ============================================================

def run_experiment(
    cfg: Config,
    make_plots: bool = True,
) -> Dict:

    cfg.output_dir.mkdir(parents=True, exist_ok=True)

    print_config(cfg)

    train_loader, test_loader, preprocess_fn = get_dataset(cfg)

    network = build_network(cfg)

    all_logs = init_error_logs()
    accuracy_logs = init_accuracy_logs()

    print("\nTraining phase")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    for epoch in range(cfg.num_epochs):

        epoch_logs = train_one_epoch(
            network=network,
            train_loader=train_loader,
            preprocess_fn=preprocess_fn,
            cfg=cfg,
        )

        merge_logs(all_logs, epoch_logs)

        train_acc = evaluate(network, train_loader, preprocess_fn, cfg)
        test_acc = evaluate(network, test_loader, preprocess_fn, cfg)

        update_accuracy_logs(
            accuracy_logs,
            train_acc,
            test_acc,
        )

        print(
            f"Epoch {epoch + 1:03d}/{cfg.num_epochs} | "
            f"train acc: {train_acc * 100:.2f}% | "
            f"test acc: {test_acc * 100:.2f}%"
        )

    # --------------------------------------------------------
    # Plots
    # --------------------------------------------------------

    if make_plots:

        plot_prediction_error_summary(
            errors=all_logs,
            filename=cfg.output_dir / "prediction_error",
            log_scale=True,
        )

        plot_lines(
            accuracy_logs,
            filename=cfg.output_dir / "training_accuracy",
            xlabel="Epoch",
            ylabel="Accuracy (%)",
            ylim=(None, 100),
        )

        # Optional GIF
        # visualize_activity_dynamics(
        #     network=network,
        #     test_loader=test_loader,
        #     preprocess_fn=preprocess_fn,
        #     cfg=cfg,
        #     sample_id=0,
        # )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    print("\nFinal evaluation")

    train_acc = evaluate(network, train_loader, preprocess_fn, cfg)
    test_acc = evaluate(network, test_loader, preprocess_fn, cfg)

    print(f"Train accuracy: {train_acc * 100:.2f}%")
    print(f"Test accuracy:  {test_acc * 100:.2f}%")

    print("Fin.")

    return {
        "final_train_acc": train_acc * 100.0,
        "final_test_acc": test_acc * 100.0,
        "final_total_error": all_logs["total_error"][-1],
    }
    

# ============================================================
# N-step experiment
# ============================================================

def run_n_step_training_experiment(
    base_cfg: Config,
    n_step_values: list[int],
) -> Dict[str, list]:

    results = {
        "n_step": [],
        "train_accuracy": [],
        "test_accuracy": [],
        "final_total_error": [],
    }

    for n_step in n_step_values:

        print("\n" + "=" * 60)
        print(f"Training model with n_step = {n_step}")
        print("=" * 60)

        cfg = replace(base_cfg, n_step=n_step)

        run = run_experiment(cfg, make_plots=False)

        results["n_step"].append(n_step)
        results["train_accuracy"].append(run["final_train_acc"])
        results["test_accuracy"].append(run["final_test_acc"])
        results["final_total_error"].append(run["final_total_error"])

    return results


# ============================================================
# Experiment wrappers
# ============================================================

def experiment_standard() -> None:

    cfg = Config(
        experiment="standard",

        activation="spike",
        spike_reset=False,

        output_dir=Path("results/standard"),
    )

    run_experiment(cfg)


def experiment_reset() -> None:

    cfg = Config(
        experiment="reset",

        activation="spike",
        spike_reset=True,

        output_dir=Path("results/reset"),
    )

    run_experiment(cfg)


def experiment_n_step() -> None:

    cfg = Config(
        experiment="n_step",

        activation="spike",
        spike_reset=False,
        
        output_dir=Path("results/n_step"),
    )

    results = run_n_step_training_experiment(
        base_cfg=cfg,
        n_step_values=[1, 2, 5, 10, 20, 50, 100, 200],
    )

    plot_n_step_training_experiment(
        results=results,
        filename=cfg.output_dir / "n_step_training_summary",
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    RUN_EXPERIMENT = "reset"

    if RUN_EXPERIMENT == "standard":
        experiment_standard()

    elif RUN_EXPERIMENT == "reset":
        experiment_reset()

    elif RUN_EXPERIMENT == "n_step":
        experiment_n_step()

    else:
        raise ValueError(f"Unknown experiment: {RUN_EXPERIMENT}")