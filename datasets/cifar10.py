import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


# ---------------------------
# Transforms
# ---------------------------
def get_cifar10_transform(normalize: bool = True):
    """
    Returns CIFAR-10 transform pipeline.

    Args:
        normalize: whether to apply normalization
    """
    if normalize:
        return transforms.Compose([
            transforms.ToTensor(),
            # Standard CIFAR-10 normalization (optional)
            # transforms.Normalize(
            #     mean=(0.4914, 0.4822, 0.4465),
            #     std=(0.2023, 0.1994, 0.2010)
            # )
        ])
    else:
        return transforms.ToTensor()


# ---------------------------
# Datasets
# ---------------------------
def get_cifar10_datasets(data_dir: str = "./data", normalize: bool = True):
    transform = get_cifar10_transform(normalize)

    train_dataset = datasets.CIFAR10(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.CIFAR10(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset


# ---------------------------
# Loaders
# ---------------------------
def get_cifar10_loaders(
    batch_size: int = 256,
    data_dir: str = "./data",
    normalize: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    train_dataset, test_dataset = get_cifar10_datasets(data_dir, normalize)

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
def preprocess_cifar10_batch(
    x,
    y,
    device,
    flatten: bool = True,
    one_hot: bool = True,
):
    """
    Input:
        x: (B, 3, 32, 32)

    Output:
        x: (B, 3072) if flattened
    """

    B = x.size(0)

    # Optional: normalize per sample (helps PCN)
    x = x / (x.max(dim=3, keepdim=True)[0]
               .max(dim=2, keepdim=True)[0]
               .max(dim=1, keepdim=True)[0] + 1e-8)

    if flatten:
        x = x.reshape(B, -1)  # (B, 3*32*32 = 3072)

    x = x.to(device)

    if one_hot:
        y = torch.nn.functional.one_hot(y, num_classes=10).float()

    y = y.to(device)

    return x, y