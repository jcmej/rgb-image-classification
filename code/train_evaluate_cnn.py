from pathlib import Path
from torchvision import datasets, transforms

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
    
def main():
    '''
    Load CIFAR-10 training and test splits
    '''
    # Store data set in data directory
    project_root = Path(__file__).resolve().parent.parent
    data_dir = project_root / 'data'

    # Convert each image into a float tensor with shape [C, H, W]
    # Convert pixel values from ints in [0, 255] to floats in [0, 1]
    transform = transforms.ToTensor()

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

    inspect_data_splits(train_dataset, test_dataset)

if __name__ == '__main__':
    main()