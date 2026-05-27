# Predicting in Spikes: Event-Based Predictive Coding with Surrogate Gradient Learning

Implementation of predictive coding networks (PCNs) with continuous and spiking neurons using PyTorch. The project investigates predictive coding inference dynamics with sigmoid, ReLU, and leaky integrate-and-fire (LIF) spiking neurons trained using surrogate-gradient learning.

The work explores:
- Predictive coding as a biologically inspired alternative to backpropagation
- Spiking predictive coding networks
- Effects of membrane reset dynamics
- Role of iterative inference steps during training
- Event-driven neural computation for neuromorphic systems

---

# Features

- Predictive coding inference dynamics
- Local prediction-error minimization
- Sigmoid, ReLU, and spiking LIF neurons
- Surrogate-gradient learning for spiking networks
- MNIST and Fashion-MNIST experiments
- Prediction-error visualizations
- Accuracy curves
- Inference-step experiments
- Activity dynamics visualization and GIF generation

---

# Installation

Clone the repository:

```bash
git clone <repo_url>
cd <repo_name>
```

Install dependencies:

```bash
pip install torch torchvision matplotlib numpy
```

Optional (GIF/video export):

```bash
conda install ffmpeg
```

---

# Project Structure

```text
.
├── main.py
├── pcn_SpikingPCN.py
├── datasets/
│   ├── mnist.py
│   ├── fashion_mnist.py
│   ├── cifar10.py
│   └── nmnist.py
├── utils/
│   └── plots.py
├── results/
└── report/
```

---

# Running Experiments

## Standard predictive coding experiment

```bash
python main.py
```

This trains the predictive coding network and generates:
- prediction-error plots
- training/test accuracy plots
- saved experiment results

---

# Configuration

Experiments are controlled through the `Config` dataclass in `main.py`.

Example:

```python
cfg = Config(
    dataset="fashion_mnist",
    activation="spike",
    num_epochs=50,
    n_step=20,
    spike_reset=False,
)
```

Key parameters:

| Parameter | Description |
|---|---|
| `dataset` | Dataset name |
| `activation` | `sigmoid`, `relu`, or `spike` |
| `n_step` | Predictive coding inference iterations |
| `v_rate` | Inference update rate |
| `w_rate` | Weight learning rate |
| `spike_reset` | Enable/disable membrane reset |
| `spike_threshold` | Spike threshold |
| `reset_beta` | Reset strength |

---

# Inference-Step Experiment

To evaluate the effect of predictive coding inference steps during training:

```python
cfg = Config(
    num_epochs=50,
    output_dir=Path("results/n_step"),
)

results = run_n_step_training_experiment(
    base_cfg=cfg,
    n_step_values=[1, 2, 5, 10, 20, 50, 100],
)

plot_n_step_training_experiment(
    results=results,
    filename=cfg.output_dir / "n_step_training_summary",
)
```

This produces:
- accuracy vs inference steps
- prediction error vs inference steps

---

# Activity Dynamics Visualization

To visualize predictive coding inference dynamics:

```python
visualize_activity_dynamics(
    network=network,
    test_loader=test_loader,
    preprocess_fn=preprocess_fn,
    cfg=cfg,
    sample_id=0,
)
```

This generates:
- hidden-layer activity dynamics
- output predictions
- prediction-error trajectories
- inference GIF/video visualizations

---

# Main Results

The experiments demonstrate:

- Spiking predictive coding networks can successfully learn meaningful representations
- LIF spiking neurons achieve competitive classification performance
- Strong membrane reset dynamics destabilize predictive coding inference
- Increasing inference steps improves learning up to a moderate range
- Predictive coding benefits from iterative latent-state relaxation


---

# Author

Naresh Balaji Ravichandran  
nbrav@kth.se