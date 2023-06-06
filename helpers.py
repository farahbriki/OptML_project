import matplotlib as plt
import numpy as np

import torch
import torch.cuda
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageNet, MNIST


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def load_dataset(dataset_name, batch_size, num_workers=2):
    train_set = None
    test_set = None
    datasets_directory = "datasets"
    path = os.path.join(datasets_directory, dataset_name)

    if dataset_name == "CIFAR10":
        cifar10_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

        train_set = CIFAR10(
            root=path, train=True, download=True, transform=cifar10_transform
        )
        test_set = CIFAR10(
            root=path, train=False, download=True, transform=cifar10_transform
        )

    elif dataset_name == "ImageNet":
        imagenet_train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        train_set = ImageNet(
            root=path, split="train", download=True, transform=imagenet_train_transform
        )
        imagenet_test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        test_set = ImageNet(
            root=path, split="val", download=True, transform=imagenet_test_transform
        )

    elif dataset_name == "MNIST":
        mnist_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
        train_set = MNIST(
            root=path, train=True, download=True, transform=mnist_transform
        )
        test_set = MNIST(
            root=path, train=False, download=True, transform=mnist_transform
        )

    else:
        raise ValueError(
            "Unsupported dataset: {}. Only CIFAR10, ImageNet or MNIST are supported.".format(
                dataset_name
            )
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_set = train_set.to(device)
    test_set = test_set.to(device)

    num_workers = 0 if device.type == "cuda" else num_workers

    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    train_loader = train_loader.to(device)
    test_loader = test_loader.to(device)
    num_classes = len(train_set.classes)
    num_channels = train_set.data.shape[3]
    image_size = train_set.data.shape[1]
    return train_loader, test_loader, num_classes, num_channels, image_size, device


# def imshow(img):
#     img = img / 2 + 0.5  # unnormalize
#     npimg = img.numpy()
#     plt.imshow(np.transpose(npimg, (1, 2, 0)))
#     plt.show()
