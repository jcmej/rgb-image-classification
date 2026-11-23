import torch
import torch.nn as nn
import torch.nn.functional as F

''' Classify CIFAR-10 images using 3 convolutional layers '''
class ConvNet(nn.Module):
    # Define model layers
    def __init__(self):
        super(ConvNet, self).__init__()

        # First convolution, 1 rgb image -> 32 feature maps
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding='same'
        )

        # Second convolution, 32 feature maps -> 64 feature maps
        self.conv2 = nn.Conv2d(
            in_channels=32,
            out_channels=64,
            kernel_size=3,
            stride=1,
            padding='same'
        )

        # Third convolution, 64 feature maps -> 128 feature maps
        self.conv3 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=3,
            stride=1,
            padding='same'
        )

        # Max pooling layer
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )

        feature_channels = 128
        feature_height = 4
        feature_width = 4

        in_features = feature_channels * feature_height * feature_width

        # First connected hidden layer, 2048 features -> 256 neurons
        self.fc1 = nn.Linear(
            in_features=in_features,
            out_features=256
        )

        # Second connected hidden layer, 256 outputs from fc1 -> 128 neurons
        self.fc2 = nn.Linear(
            in_features=256,
            out_features=128
        )

        # Final output layer, 128 outputs from fc2 -> 10 class scores per image
        self.fc_out = nn.Linear(
            in_features=128,
            out_features=10
        )

    # Extracts rgb image features, returns 10 class scores per image
    def forward(self, x):
        # [batch, 3, 32, 32] -> [batch, 32, 16, 16]
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)

        # [batch, 32, 16, 16] -> [batch, 64, 8, 8]
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # [batch, 64, 8, 8] -> [batch, 128, 4, 4]
        x = self.conv3(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten [batch, 128, 4, 4] tensor into [batch, 2048] tensor
        x = torch.flatten(x, start_dim=1)

        # Pass x through first fully connected hidden layer, then apply ReLU
        # 2048 features -> 256 neurons
        x = self.fc1(x)
        x = F.relu(x)

        # Pass x through second fully connected hidden layer, then apply ReLU
        # 256 features -> 128 neurons
        x = self.fc2(x)
        x = F.relu(x)

        # Pass x through the output layer
        # 128 features -> 10 logits
        return self.fc_out(x)