from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from ConvNet import ConvNet

def inspect_data_splits(train_dataset, test_dataset):
    print(f'Training images: {len(train_dataset)}')
    print(f'Testing images: {len(test_dataset)}')
    print(f'Classes: {train_dataset.classes}')

    image, label = train_dataset[0]
    print(f'Image shape: {image.shape}')
    print(f'Image data type: {image.dtype}')
    print(f'Pixel range: {image.min().item():.3f} to {image.max().item():.3f}')
    print(f'Label: {label}')
    print(f'Class name: {train_dataset.classes[label]}')

def inspect_data_batches(train_loader, test_loader):
    images, labels = next(iter(train_loader))
    print(f'Image batch shape: {images.shape}')
    print(f'Label batch shape: {labels.shape}')
    print(f'Pixel range: {images.min().item():3f} to {images.max().item():3f}')
    print(f'Number of training batches: {len(train_loader)}')
    print(f'Number of testing batches: {len(test_loader)}')
    
def main():
    '''
    Load CIFAR-10 training and test splits
    '''
    # Store data set in data directory
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'

    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)

    # Convert each image into a float tensor with shape [C, H, W]
    # Convert pixel values from ints in [0, 255] to floats in [0, 1]
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

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

    # inspect_data_splits(train_dataset, test_dataset)

    # Create loader for each split
    batch_size = 128
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        drop_last=False
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False
    )

    # inspect_data_batches(train_loader, test_loader)


if __name__ == '__main__':
    main()