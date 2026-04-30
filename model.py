import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from data import get_mnist_loaders, preprocess_batch, get_device
from plots import plot_lines

##### Model #####

class Layer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.W = nn.Parameter(torch.zeros(output_dim, input_dim))
        self.b = nn.Parameter(torch.zeros(output_dim))
        nn.init.xavier_uniform_(self.W)
           
        # Adam states as buffers
        self.register_buffer("m_W", torch.zeros_like(self.W))
        self.register_buffer("v_W", torch.zeros_like(self.W))
        self.register_buffer("m_b", torch.zeros_like(self.b))
        self.register_buffer("v_b", torch.zeros_like(self.b))

        self.t = 0
        
    # def feedback(self, x):
    #     return x @ self.W.T + self.b

    def feedforward(self, x):
        return x @ self.W.T + self.b

    def update_params(self, e, f_x, lr):
        # Learning Gradients
        grad_w = (e.T @ f_x) / B
        grad_b = e.mean(dim=0)
        # print (self.W.shape, grad_w.shape, self.b.shape, grad_b.shape)
        # Update weights and biases
        self.t += 1
        with torch.no_grad():
            self.m_W, self.v_W, self.W = self.adam_update(self.W, grad_w, self.m_W, self.v_W, self.t, lr)
            self.m_b, self.v_b, self.b = self.adam_update(self.b, grad_b, self.m_b, self.v_b, self.t, lr)
            # self.W += lr * grad_w
            # self.b += lr * grad_b

    def adam_update(self, param, grad, m, v, t, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        param += lr * m_hat / (torch.sqrt(v_hat) + eps)
        return m, v, param

class Network(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.layer3 = Layer(dims[2], dims[3])
        self.layer2 = Layer(dims[1], dims[2])
        self.layer1 = Layer(dims[0], dims[1])
        
    # def actfn(self, a):
    #     return torch.relu(a)

    # def actfn_deriv(self, a):
    #     return (a > 0).float()

    def actfn(self, a):
        return torch.sigmoid(a)

    def actfn_deriv(self, a):
        s = torch.sigmoid(a)
        return s * (1 - s)

    def inverse_sigmoid(self, x, eps=1e-6):
        x = torch.clamp(x, eps, 1 - eps)
        return torch.log(x / (1 - x))

    def train(self, input, target, device, lr, ir, T=100):
        # Batch size
        B = input.size(0)

        # Initialize x's with small random values, l = 1 to L
        x0 = self.inverse_sigmoid(input)
        x1 = self.layer1.feedforward(self.actfn(x0))
        x2 = self.layer2.feedforward(self.actfn(x1))
        x3 = target.clone()

        for t in range(T):
            # Feedforward predictions - preactivations
            x1_hat = self.layer1.feedforward(self.actfn(x0))
            x2_hat = self.layer2.feedforward(self.actfn(x1))
            x3_hat = self.layer3.feedforward(self.actfn(x2))

            # Error = x_l - x_l_hat
            e1 = x1 - x1_hat
            e2 = x2 - x2_hat
            e3 = x3 - x3_hat

            # Inference Gradients = - e_l + f'(x_l) * (w_l @ e_(l+1))
            grad_act1 = - e1 + self.actfn_deriv(x1) * (e2 @ self.layer2.W) 
            grad_act2 = - e2 + self.actfn_deriv(x2) * (e3 @ self.layer3.W) 
                        
            # Inference updates
            x1 += ir * grad_act1
            x2 += ir * grad_act2

        # Learning updates
        self.layer1.update_params(e1, self.actfn(x0), lr)
        self.layer2.update_params(e2, self.actfn(x1), lr)
        self.layer3.update_params(e3, self.actfn(x2), lr)

        return {
            "e1_mse": (e1 ** 2).mean(dim=0).sum().item(),
            "e2_mse": (e2 ** 2).mean(dim=0).sum().item(),
            "e3_mse": (e3 ** 2).mean(dim=0).sum().item(),            
        }

    def infer(self, input, target, device, ir, T=100):
        # Batch size
        B = input.size(0)

        # Feedforward sweep
        x0 = self.inverse_sigmoid(input)
        x1 = self.layer1.feedforward(self.actfn(x0))
        x2 = self.layer2.feedforward(self.actfn(x1))
        x3 = self.layer3.feedforward(self.actfn(x2))

        pred_labels = x3.argmax(dim=1)
        true_labels = target.argmax(dim=1)
        correct = (pred_labels == true_labels).sum().item()
        
        return {
            "pred_labels": pred_labels,
            "true_labels": true_labels,
            "correct": correct,
            "num_samples": B,
        }
        
#####  Main loop  #####

device = get_device()

num_epochs      = 20
batch_size      = 256
lr              = 0.001
ir              = 0.1
T               = 100

# TODO:
# Add precision waiting
# Visualize gradients per layer

train_loader, test_loader = get_mnist_loaders(batch_size=batch_size)

network = Network(dims=[784, 600, 600, 10]).to(device)

#####  Training  #####

print ("\nTraining phase")

e1_norm = []
e2_norm = []
e3_norm = []

for epoch in range(num_epochs):
    for batch_idx, (x0, y) in enumerate(train_loader):
        B = x0.size(0)
        x0, y = preprocess_batch(x0, y, device)
        out = network.train(x0, y, device, lr, ir, T)            
        e1_norm.append(out["e1_mse"])
        e2_norm.append(out["e2_mse"])
        e3_norm.append(out["e3_mse"])
    print(f"Epoch {epoch + 1}/{num_epochs} finished")
    if epoch % 5 == 0:
        plot_lines({
            "error1": e1_norm,
            "error2": e2_norm,
            "error3": e3_norm
        }, filename="exp1_errors", xlabel="Step", ylabel="Mean Prediction Error", log_scale=True)

##### Inference #####

print ("\nInference phase")

total_correct = 0
total_samples = 0
for batch_idx, (x0, y) in enumerate(train_loader):
    B = x0.size(0)
    x0, y = preprocess_batch(x0, y, device)
    out = network.infer(x0, y, device, ir, T)
    total_correct += out["correct"]
    total_samples += out["num_samples"]
epoch_acc = total_correct / total_samples
print(f"train accuracy={epoch_acc * 100:.2f}%")

total_correct = 0
total_samples = 0
for batch_idx, (x0, y) in enumerate(test_loader):
    B = x0.size(0)
    x0, y = preprocess_batch(x0, y, device)
    out = network.infer(x0, y, device, ir, T)
    total_correct += out["correct"]
    total_samples += out["num_samples"]
epoch_acc = total_correct / total_samples
print(f"test accuracy={epoch_acc * 100:.2f}%")

print("Fin.")