import torch
from torchvision import datasets, transforms

# Flag for dataset selection
DATASET = 'MNIST'  # Set the dataset (can also be 'MNIST', etc.)

# Dataset-specific configuration
if DATASET == 'CIFAR10':
    # CIFAR-10 specific configuration
    input_size = 32 * 32 * 3  # CIFAR-10 images are 32x32 with 3 channels
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))  # CIFAR-10 normalization values
    ])
    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)

elif DATASET == 'MNIST':
    # MNIST specific configuration
    input_size = 28 * 28  # MNIST images are 28x28 grayscale
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))  # MNIST normalization values
    ])
    # Load MNIST dataset
    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    val_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

else:
    raise ValueError("Unsupported dataset. Set DATASET to 'CIFAR10' or 'MNIST'.")

# Create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=False)