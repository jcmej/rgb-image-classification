# RGB Image Classification

Two convolutional neural networks implemented in PyTorch to classify CIFAR-10
images.

The experiment compares a three-convolution network with a four-convolution
network using the same training settings and fully connected layers.

## Dataset

CIFAR-10 contains 32×32 RGB images across 10 classes: airplane, automobile, bird,
cat, deer, dog, frog, horse, ship, and truck. The project uses the official split
of 50,000 training images and 10,000 test images.

Images are converted to tensors and normalized using a mean and standard
deviation of 0.5 for each color channel, mapping pixel values to [-1, 1]. No data
augmentation is applied. Training samples are shuffled, test samples retain
their order, and both loaders keep the final partial batch.

## Architectures

Model definitions are in `code/ConvNet.py`.

| Model | Convolution output channels | Pooling placement |
| --- | --- | --- |
| ConvNet3 | 32 → 64 → 128 | After each convolution and ReLU |
| ConvNet4 | 32 → 64 → 128 → 128 | After the first, second, and fourth convolution and ReLU |

All convolutions use 3×3 kernels, stride 1, and padding that preserves spatial
dimensions. ReLU follows every convolution. Max pooling uses a 2×2 window and
stride 2, reducing spatial size from 32×32 to 16×16, 8×8, and finally 4×4.

Both networks flatten the final 128×4×4 tensor into 2,048 features and use the same
classifier:

```text
2048 → 256 → ReLU → 128 → ReLU → 10 class scores
```

The output consists of raw logits for `CrossEntropyLoss`. Predictions use the
highest-scoring class. Neither model uses dropout or batch normalization.

## Setup

The experiment was run with Python 3.13 on a MacBook Pro using the MPS device.
The package versions from that environment are listed in `requirements.txt`.

From the project directory:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

```bash
python code/train_evaluate_cnn.py
```

The script downloads CIFAR-10 into `data/` if needed, then trains ConvNet3 and
ConvNet4 sequentially from scratch. It automatically selects CUDA when
available, followed by Apple MPS, then CPU.

| Setting | Value |
| --- | --- |
| Epochs per model | 30 |
| Batch size | 128 |
| Loss | Cross entropy |
| Optimizer | SGD |
| Learning rate | 0.01, constant |
| Momentum | 0.9 |
| Weight decay | 0 |

Training and evaluation run once per epoch. Epoch losses are weighted by the
number of images in each batch so the smaller final batch is handled correctly.
Training metrics summarize predictions made while the weights are updated;
test metrics use the fixed weights at the end of each epoch.

Settings are defined directly in `main()` rather than through command line
arguments. For a quick check, change `num_epochs=30` in the `train_model_spec`
call to `num_epochs=1`. This runs one epoch for each model. Restore 30 for the
full experiment.

## Outputs

```text
results/
    ConvNet3_results.txt
    ConvNet4_results.txt
plots/
    ConvNet3_loss.png
    ConvNet3_accuracy.png
    ConvNet4_loss.png
    ConvNet4_accuracy.png
```

Each text file records the configuration, metrics after every completed epoch,
final accuracies, minimum training and test losses with their epochs, and total
training/evaluation time. Plots are saved after each model finishes, using a
noninteractive Matplotlib backend suitable for a server without a display.

Rerunning the script overwrites the files with the same names. Model weights
are not saved. Dataset files, generated results, plots, and PDFs are excluded
from Git by the current `.gitignore`.

## Results

The following values are from the recorded 30-epoch run on Apple MPS. Accuracies
are from epoch 30, rather than the best test accuracy across epochs.

| Metric | ConvNet3 | ConvNet4 |
| --- | ---: | ---: |
| Training accuracy | 99.04% | 98.76% |
| Test accuracy | 75.26% | 77.05% |
| Training loss | 0.029278 | 0.036308 |
| Test loss | 1.889967 | 1.498316 |
| Lowest test loss | 0.770829 at epoch 8 | 0.711580 at epoch 11 |
| Training and evaluation time | 2.06 minutes | 2.31 minutes |

Adding the fourth convolution improved final test accuracy by 1.79 percentage
points in this run, with approximately 15 additional seconds of training and
evaluation time.

Both networks showed overfitting: training accuracy approached 99%, while test
accuracy remained around 75–77%. Test loss generally increased after its minimum
even as training loss continued falling. Future experiments could investigate
data augmentation, weight decay, dropout, and alternative architectures to
improve generalization. These additions were not evaluated here.

These results represent one run per architecture. The script does not fix a
random seed, so initialization, training order, and results can vary between
runs. Timings depend on hardware and environment.

## Project files

```text
code/
    ConvNet.py              # Network definitions
    train_evaluate_cnn.py   # Data loading, training, evaluation, logging, and plots
requirements.txt            # Package versions
LICENSE                     # License terms
```
