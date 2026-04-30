import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_device():
    """Return available device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_mnist_transform(normalize: bool = True):
    """
    Returns MNIST transform pipeline.

    Args:
        normalize: whether to apply standard normalization
    """
    if normalize:
        return transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        return transforms.ToTensor()

def get_mnist_datasets(data_dir: str = "./data", normalize: bool = True):
    """
    Returns MNIST train and test datasets.

    Args:
        data_dir: path to dataset
        normalize: whether to normalize inputs
    """
    transform = get_mnist_transform(normalize)

    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )

    return train_dataset, test_dataset

def get_mnist_loaders(
    batch_size: int = 256,
    data_dir: str = "./data",
    normalize: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
):
    """
    Returns MNIST train and test DataLoaders.

    Args:
        batch_size: batch size
        data_dir: dataset path
        normalize: whether to normalize inputs
        num_workers: dataloader workers
        pin_memory: speeds up GPU transfer
    """
    train_dataset, test_dataset = get_mnist_datasets(data_dir, normalize)

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

def preprocess_batch(x, y, device, flatten: bool = True, one_hot: bool = True):
    """
    Preprocess a batch:
    - move to device
    - flatten inputs
    - convert labels to one-hot

    Args:
        x: input tensor (B, 1, 28, 28)
        y: labels (B,)
        device: torch device
        flatten: flatten images to (B, 784)
        one_hot: convert labels to one-hot
    """
    B = x.size(0)

    if flatten:
        x = x.reshape(B, -1)

    x = x.to(device)

    if one_hot:
        y = torch.nn.functional.one_hot(y, num_classes=10).float()

    y = y.to(device)

    return x, y