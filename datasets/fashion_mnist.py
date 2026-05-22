import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ---------------------------
# Transforms
# ---------------------------
def get_fashion_mnist_transform(normalize: bool = True):
    """
    Returns Fashion-MNIST transform pipeline.

    Args:
        normalize: whether to apply normalization
    """
    if normalize:
        return transforms.Compose([
            transforms.ToTensor(),
            # Optional standard Fashion-MNIST normalization:
            # transforms.Normalize((0.2860,), (0.3530,))
        ])
    else:
        return transforms.ToTensor()


# ---------------------------
# Datasets
# ---------------------------
def get_fashion_mnist_datasets(data_dir: str = "./data", normalize: bool = True):
    transform = get_fashion_mnist_transform(normalize)

    train_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.FashionMNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform,
    )

    return train_dataset, test_dataset


# ---------------------------
# Loaders
# ---------------------------
def get_fashion_mnist_loaders(
    batch_size: int = 256,
    data_dir: str = "./data",
    normalize: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    train_dataset, test_dataset = get_fashion_mnist_datasets(data_dir, normalize)

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
def preprocess_fashion_mnist_batch(
    x,
    y,
    device,
    flatten: bool = True,
    one_hot: bool = True,
):
    """
    Preprocess Fashion-MNIST batch.

    Input:
        x: (B, 1, 28, 28)
        y: (B,)

    Output:
        x: (B, 784) if flatten=True
        y: (B, 10) if one_hot=True
    """
    B = x.size(0)

    if flatten:
        x = x.reshape(B, -1)

    x = x.to(device)

    if one_hot:
        y = torch.nn.functional.one_hot(y, num_classes=10).float()

    y = y.to(device)

    return x, y