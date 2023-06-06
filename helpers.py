import matplotlib as plt
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageNet, MNIST


import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def load_dataset(
    dataset_name,
    batch_size,
    train_transform=transforms.ToTensor(),
    test_transform=transforms.ToTensor(),
    num_workers=2,
):
    train_set = None
    test_set = None
    datasets_directory = "datasets"
    path = os.path.join(datasets_directory, dataset_name)
    if dataset_name == "CIFAR10":
        train_set = CIFAR10(
            root=path, train=True, download=True, transform=train_transform
        )
        test_set = CIFAR10(
            root=path, train=False, download=True, transform=test_transform
        )

    elif dataset_name == "ImageNet":
        train_set = ImageNet(
            root=path, split="train", download=True, transform=train_transform
        )
        test_set = ImageNet(
            root=path, split="val", download=True, transform=test_transform
        )
    elif dataset_name == "MNIST":
        train_set = MNIST(
            root=path, train=True, download=True, transform=train_transform
        )
        test_set = MNIST(
            root=path, train=False, download=True, transform=test_transform
        )
    else:
        raise ValueError(
            "Unsupported dataset: {}. Only CIFAR10, ImageNet or MNIST are supported.".format(
                dataset_name
            )
        )

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_loader, test_loader


def imshow(img):
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# # get some random training images
# dataiter = iter(trainloader)
# images, labels = next(dataiter)

# # show images
# imshow(torchvision.utils.make_grid(images))
# # print labels
# print(' '.join(f'{classes[labels[j]]:5s}' for j in range(batch_size)))
