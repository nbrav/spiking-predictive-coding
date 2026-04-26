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

    # def feedback(self, x):
    #     return x @ self.W.T + self.b

    def feedforward(self, x):
        return x @ self.W.T + self.b

class Network(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.dims = dims
        self.layer3 = Layer(dims[2], dims[3])
        self.layer2 = Layer(dims[1], dims[2])
        self.layer1 = Layer(dims[0], dims[1])

    def train(self, input, target, device, lr, ir, T=100):
        # Batch size
        B = input.size(0)

        # Initialize x's with small random values, l = 1 to L
        x3 = target.clone()
        x2 = torch.randn(B, self.dims[2], device=device, requires_grad=False)
        x1 = torch.randn(B, self.dims[1], device=device, requires_grad=False)
        x0 = input.clone()      

        for t in range(T):

            # Feedforward predictions - preactivations
            x1_hat = self.layer1.feedforward(torch.relu(x0))
            x2_hat = self.layer2.feedforward(torch.relu(x1))
            x3_hat = self.layer3.feedforward(torch.relu(x2))

            # Error = x_l - x_l_hat
            e3 = x3 - x3_hat
            e2 = x2 - x2_hat
            e1 = x1 - x1_hat

            # Inference Gradients = - e_l + f'(x_l) * (w_l @ e_(l+1))
            grad_actfn2 = (x2 > 0).float()
            grad_act2 = -e2 + grad_actfn2 * (e3 @ self.layer3.W) 
            grad_actfn1 = (x1 > 0).float()
            grad_act1 = -e1 + grad_actfn1 * (e2 @ self.layer2.W) 
                        
            # Inference updates
            x2 += ir * grad_act2
            x1 += ir * grad_act1
            
        # Print
        print (f"t={t}, e3={(e3 ** 2).mean(dim=0).sum().item():6.4f}, e2={(e2 ** 2).mean(dim=0).sum().item():6.4f}, e1={(e1 ** 2).mean(dim=0).sum().item():6.4f}")
        e1_norm.append((e1 ** 2).mean(dim=0).sum().item())
        e2_norm.append((e2 ** 2).mean(dim=0).sum().item())
        e3_norm.append((e3 ** 2).mean(dim=0).sum().item())

        # Learning Gradients
        grad_w3 = (e3.T @ torch.relu(x2)) / B
        grad_w2 = (e2.T @ torch.relu(x1)) / B
        grad_w1 = (e1.T @ torch.relu(x0)) / B
        
        # Learning updates
        self.layer3.W.data += lr * grad_w3
        self.layer2.W.data += lr * grad_w2
        self.layer1.W.data += lr * grad_w1

#####  Training  #####

device = get_device()

train_loader, test_loader = get_mnist_loaders(batch_size=256)

network = Network(dims=[784, 1000, 800, 10]).to(device)

num_epochs  = 1
lr          = 0.001
ir          = 0.1

e1_norm = []
e2_norm = []
e3_norm = []

for epoch in range(num_epochs):
    for batch_idx, (x0, y) in enumerate(train_loader):
        B = x0.size(0)

        x0, y = preprocess_batch(x0, y, device)

        network.train(x0, y, device, lr, ir)

    print(f"Epoch {epoch + 1}/{num_epochs} finished")

plot_lines({
    "e1": e1_norm,
    "e2": e2_norm,
    "e3": e3_norm
}, filename="exp1_errors", xlabel="Step", ylabel="Mean Prediction Error", log_scale=True)

print("Fin.")