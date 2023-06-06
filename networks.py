import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)  # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class LeNet5(nn.Module):
    def __init__(self, in_channels=1):
        super(LeNet5, self).__init__()

        self.features = nn.Sequential(  # TODO: ensure input is 32x32
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16 * 5 * 5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# Basic building block of ResNet
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.residual_layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * 4, kernel_size=1),
            nn.BatchNorm2d(out_channels * 4),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels * 4),
            )

    def forward(self, x):
        identity = x
        out = self.residual_layers(x)
        out += self.shortcut(identity)
        out = nn.ReLU(inplace=True)(out)
        return out


# ResNet-50 architecture
class ResNet50(nn.Module):
    def __init__(self, in_channels=3, input_size=224, num_classes=1000):
        super(ResNet50, self).__init__()

        self.input_size = input_size

        self.model = nn.Sequential(  # TODO: verify input format
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            self._make_layer(64, 64, 3, stride=1),
            self._make_layer(256, 128, 4, stride=2),
            self._make_layer(512, 256, 6, stride=2),
            self._make_layer(1024, 512, 3, stride=2),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, num_classes),
        )

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels * 4, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = nn.functional.interpolate(x, size=self.input_size)
        return self.model(x)


class AlexNet(nn.Module):
    def __init__(self, dataset_name="CIFAR10", in_channels=3, num_classes=1000):
        super(AlexNet, self).__init__()

        if dataset_name == "CIFAR10":
            self.features = nn.Sequential(  # TODO: input is RGB 32x32
                nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=3, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.linear_input_size = 256 * 4 * 4
        elif dataset_name == "ImageNet":
            self.features = nn.Sequential(  # TODO: input is RGB 227x227
                nn.Conv2d(in_channels, 96, kernel_size=11, stride=4, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(96, 256, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
                nn.Conv2d(256, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2),
            )
            self.linear_input_size = 256 * 6 * 6
        else:
            raise ValueError(
                "Unsupported dataset: {}. Only CIFAR10 or ImageNet are supported.".format(
                    dataset_name
                )
            )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(self.linear_input_size, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.linear_input_size)
        x = self.classifier(x)
        return x


# VGGNet architecture
class VGGNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1000, num_layers=16):
        super(
            VGGNet, self
        ).__init__()  # in_channels: 3 for RGB or 1 for greyscale, TODO: expect shape [batch_size, 1, height, width]

        self.in_channels = in_channels

        if num_layers == 16:
            cfg = [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                "M",
                512,
                512,
                512,
                "M",
            ]
        elif num_layers == 19:
            cfg = [
                64,
                64,
                "M",
                128,
                128,
                "M",
                256,
                256,
                256,
                256,
                "M",
                512,
                512,
                512,
                512,
                "M",
                512,
                512,
                512,
                512,
                "M",
            ]
        else:
            raise ValueError(
                "Unsupported number of layers for VGGNet: {}. Only 16 or 19 layers are supported.".format(
                    num_layers
                )
            )

        self.features = self._make_layers(cfg)

        self.classifier = nn.Sequential(  # TODO: RGB, verify how to handle size
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _make_layers(self, cfg):
        layers = []
        in_channels_ = self.in_channels

        for v in cfg:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.append(nn.Conv2d(in_channels_, v, kernel_size=3, padding=1))
                layers.append(nn.ReLU(inplace=True))
                in_channels_ = v

        return nn.Sequential(*layers)


class LeNetPlusPlus(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(LeNetPlusPlus, self).__init__()

        self.features = (
            nn.Sequential(  # TODO: ensure input is grayscale 32x32 or modularize
                nn.Conv2d(in_channels, 6, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(6, 16, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(16, 120, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
            )
        )

        self.classifier = nn.Sequential(
            nn.Linear(120, 84), nn.ReLU(inplace=True), nn.Linear(84, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class MiniVGG(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(MiniVGG, self).__init__()

        self.features = nn.Sequential(  # in_channels: 3 for RGB or 1 for greyscale, TODO: expect shape [batch_size, 1, height, width]
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(64 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x
