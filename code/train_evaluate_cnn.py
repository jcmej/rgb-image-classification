from pathlib import Path
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch
from ConvNet import ConvNet3, ConvNet4
import time
import matplotlib
matplotlib.use ('Agg')
import matplotlib.pyplot as plt

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

def train(model, device, train_loader, optimizer, criterion):
    ''' Train specified model and return sample-averaged loss and accuracy '''
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

def evaluate(model, device, test_loader, criterion):
    ''' Evaluate specified model's weights and return average loss and accuracy '''
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

def report(message, output_path):
    ''' Print a message and append it to a text results file '''
    print(message, flush=True)
    with output_path.open('a', encoding='utf-8') as output_file:
        output_file.write(message + '\n')

def train_model_spec(model, criterion, optimizer, device, train_loader, test_loader, num_epochs=30):
    ''' Reusable training and evaluation loop '''
    # Store results in a text file
    model_name = type(model).__name__
    results_dir = Path(__file__).resolve().parent.parent / 'results'
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f'{model_name}_results.txt'
    # Begin new results file
    output_path.write_text('', encoding='utf-8')

    report(
        f'Model: {model_name}\n'
        f'Device: {device}\n'
        f'Epochs: {num_epochs}\n'
        f'Batch size: {train_loader.batch_size}\n'
        f'Optimizer: {optimizer}\n'
        f'Loss function: {criterion}',
        output_path
    )
    
    history = {
        'train_loss': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
    }

    min_train_loss = float('inf')
    min_test_loss = float('inf')
    min_train_epoch = 0
    min_test_epoch = 0

    start_time = time.perf_counter()
    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}', flush=True)

        train_loss, train_accuracy = train(
            model=model,
            device=device,
            train_loader=train_loader,
            optimizer=optimizer,
            criterion=criterion
        )

        test_loss, test_accuracy = evaluate(
            model=model,
            device=device,
            test_loader=test_loader,
            criterion=criterion
        )

        history['train_loss'].append(train_loss)
        history['train_accuracy'].append(train_accuracy)
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_accuracy)

        # Track minimum losses
        if train_loss < min_train_loss:
            min_train_loss = train_loss
            min_train_epoch = epoch

        if test_loss < min_test_loss:
            min_test_loss = test_loss
            min_test_epoch = epoch

        report(
            f'\nEpoch {epoch}/{num_epochs}\n'
            f'Training loss: {train_loss:.6f}\n'
            f'Training accuracy: {train_accuracy:.2f}%\n'
            f'Testing loss: {test_loss:.6f}\n'
            f'Testing accuracy: {test_accuracy:.2f}%',
            output_path
        )

    total_seconds = time.perf_counter() - start_time
    final_train_accuracy = history['train_accuracy'][-1]
    final_test_accuracy = history['test_accuracy'][-1]

    report(
        f'\nResults after {num_epochs} epochs\n'
        f'Total training/evaluation time: {total_seconds / 60:.2f} minutes\n'
        f'Training accuracy: {final_train_accuracy:.2f}%\n'
        f'Test accuracy: {final_test_accuracy:.2f}%\n'
        f'Lowest training loss: {min_train_loss:.6f} '
        f'at epoch {min_train_epoch}\n'
        f'Lowest testing loss: {min_test_loss:.6f} '
        f'at epoch {min_test_epoch}',
        output_path
    )

    return history

def plot_history(history, model_name):
    ''' Save training/test loss and accuracy curves for a model '''
    # Save loss and accuracy plots for the specified model
    plots_dir = Path(__file__).resolve().parent.parent / 'plots'
    plots_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history['train_loss']) + 1)

    # Plot training and testing loss
    fig, ax = plt.subplots()

    ax.plot(epochs, history['train_loss'], marker='o', label='Training')
    ax.plot(epochs, history['test_loss'], marker='o', label='Test')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Cross entropy loss')
    ax.set_title(f'{model_name} Loss')
    ax.legend()

    fig.tight_layout()
    fig.savefig(plots_dir / f'{model_name}_loss.png', dpi=150)
    plt.close(fig)

    # Plot training and testing accuracy
    fig, ax = plt.subplots()

    ax.plot(epochs, history['train_accuracy'], marker='o', label='Training')
    ax.plot(epochs, history['test_accuracy'], marker='o', label='Test')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.set_title(f'{model_name} Accuracy')
    ax.legend()

    fig.tight_layout()
    fig.savefig(plots_dir / f'{model_name}_accuracy.png', dpi=150)
    plt.close(fig)

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

    criterion = torch.nn.CrossEntropyLoss()
    for convnet in (ConvNet3, ConvNet4):
        model = convnet().to(device)
        # Initialize optimizer for model
        optimizer = torch.optim.SGD(
            params = model.parameters(),
            lr=0.01,
            momentum=0.9
        )

        # Train the model
        results = train_model_spec(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train_loader=train_loader,
            test_loader=test_loader,
            num_epochs=30
        )

        # Plot results
        plot_history(results, type(model).__name__)

if __name__ == '__main__':
    main()
