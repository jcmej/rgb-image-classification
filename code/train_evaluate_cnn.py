from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from ConvNet import ConvNet3, ConvNet4

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

''' Train specified model and return sample-averaged loss and accuracy '''
def train(model, device, train_loader, optimizer, criterion):
    # Configure model to train
    model.train()

    total_samples = 0
    total_loss = 0.0
    total_correct = 0

    for images, labels in train_loader:
        # Place batch on the same device as the model
        images = images.to(device)
        labels = labels.to(device)

        # Clear gradients from previous batch
        optimizer.zero_grad()

        # Predict class scores with current weights
        logits = model(images)

        # Determine the batch's mean loss
        loss = criterion(logits, labels)

        # Calculate gradients, backpropogate
        loss.backward()
        optimizer.step()

        # Weigh the batch's mean loss by its number of images
        batch_size = images.size(0)
        total_samples += batch_size
        total_loss += loss.item() * batch_size

        # Select highest-scoring class for each image
        predictions = logits.argmax(dim=1)
        total_correct += (predictions == labels).sum().item()

    epoch_loss = total_loss / total_samples
    epoch_accuracy = (total_correct / total_samples) * 100.0

    return epoch_loss, epoch_accuracy

''' Evaluate specified model's weights and return average loss and accuracy '''
def evaluate(model, device, test_loader, criterion):
    # Configure model to evaluate
    model.eval()
    total_samples = 0
    total_loss = 0.0
    total_correct = 0

    with torch.no_grad():
        for images, labels in test_loader:
            # Place batch on the same device as loader
            images = images.to(device)
            labels = labels.to(device)

            # Predict class scores with current weights
            logits = model(images)

            # Determine the batch's loss
            loss = criterion(logits, labels)

            batch_size = images.size(0)
            total_samples += batch_size
            total_loss += loss.item() * batch_size

            # Select highes-scoring class for each image
            predictions = logits.argmax(dim=1)
            total_correct += (predictions == labels).sum().item()

    epoch_loss = total_loss / total_samples
    epoch_accuracy = (total_correct / total_samples) * 100.0

    return epoch_loss, epoch_accuracy

def main():
    '''
    Load CIFAR-10 training and test splits.
    Create data loaders for each split.
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

    ''' Train and evaluate '''
    # Select device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    elif torch.backends.mps.is_available():
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'Selected device: {device.type}')

    # Initialize Conv3 and move parameters to the device
    model = ConvNet3().to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        params=model.parameters(),
        lr=0.01,
        momentum=0.9
    )

    print(f'Training ConvNet3 for 1 epoch on {device}', flush=True)

    train_loss, train_accuracy = train(
        model=model,
        device=device,
        train_loader=train_loader,
        optimizer=optimizer,
        criterion=criterion
    )

    print(f'\nTraining loss: {train_loss:.2f}')
    print(f'Training accuracy: {train_accuracy:.2f}')

    test_loss, test_accuracy = evaluate(
        model=model,
        device=device,
        test_loader=test_loader,
        criterion=criterion
    )

    print(f'Test loss: {test_loss:.2f}')
    print(f'Test accuracy: {test_accuracy:.2f}')

if __name__ == '__main__':
    main()