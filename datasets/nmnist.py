
import torch
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader


# ---------------------------
# Transforms
# ---------------------------
def get_nmnist_transform(n_time_bins: int = 30):
    """
    Convert event streams → frame tensors.

    Output shape:
        (T, 2, 34, 34)
    """
    sensor_size = tonic.datasets.NMNIST.sensor_size

    transform = transforms.Compose([
        transforms.ToFrame(
            sensor_size=sensor_size,
            n_time_bins=n_time_bins,
        )
    ])

    return transform


# ---------------------------
# Datasets
# ---------------------------
def get_nmnist_datasets(
    data_dir: str = "./data",
    n_time_bins: int = 30,
):
    transform = get_nmnist_transform(n_time_bins)

    train_dataset = tonic.datasets.NMNIST(
        save_to=data_dir,
        train=True,
        transform=transform,
    )

    test_dataset = tonic.datasets.NMNIST(
        save_to=data_dir,
        train=False,
        transform=transform,
    )

    return train_dataset, test_dataset


# ---------------------------
# Loaders
# ---------------------------
def get_nmnist_loaders(
    batch_size: int = 64,
    data_dir: str = "./data",
    n_time_bins: int = 30,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    train_dataset, test_dataset = get_nmnist_datasets(data_dir, n_time_bins)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, test_loader


# ---------------------------
# Preprocessing
# ---------------------------
def preprocess_nmnist_batch(
    x,
    y,
    device,
    flatten_spatial: bool = True,
    one_hot: bool = True,
    collapse_time: bool = False,
):
    """
    Input:
        x: (B, T, 2, 34, 34)

    Output:
        collapse_time=True:
            (B, D)

        collapse_time=False:
            (T, B, D)   ← ALWAYS time-first
    """

    # ---- convert to float ----
    x = x.float()

    B, T, C, H, W = x.shape

    # -------------------------
    # Static mode
    # -------------------------
    if collapse_time:
        x = x.sum(dim=1)  # (B, 2, 34, 34)

        if flatten_spatial:
            x = x.reshape(B, -1)  # (B, D)

        # normalize per sample
        x = x / (x.max(dim=1, keepdim=True).values + 1e-8)

    # -------------------------
    # Temporal mode (time-first)
    # -------------------------
    else:
        if flatten_spatial:
            x = x.reshape(B, T, -1)  # (B, T, D)

        # normalize per sample per timestep
        max_val = x.max(dim=2, keepdim=True).values
        x = x / (max_val + 1e-8)

        # ---- convert to (T, B, D) ----
        x = x.permute(1, 0, 2)

    x = x.to(device)

    if one_hot:
        y = torch.nn.functional.one_hot(y, num_classes=10).float()

    y = y.to(device)

    return x, y