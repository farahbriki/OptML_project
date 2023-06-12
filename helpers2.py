import torch
import torch.cuda
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, ImageNet, MNIST

import time
import yaml
from tqdm import tqdm
from networks import *

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def load_dataset(dataset_name, batch_size, num_workers=2):
    """Loads the dataset given in the parameters and returns relevant information.

    Args:
        dataset_name (string): name of the dataset to be loaded. One of CIFAR10, ImageNet or MNIST.
        batch_size (int): size of the batches
        num_workers (int, optional): how many subprocesses to use for data loading. Defaults to 2.

    Raises:
        ValueError: if the 'dataset_name' is not one of CIFAR10, ImageNet or MNIST.

    Returns:
        tuple: returns a tuple of the train loader (DataLoader), the test loader(DataLoader), the number of category classes in the dataset (int),
        the number of color channels in the images of the dataset (int), the image size (int) and the device on which torch.Tensor will be allocated.
    """
    train_set = None
    test_set = None
    datasets_directory = "datasets"
    path = os.path.join(datasets_directory, dataset_name)

    # get the train and test sets according to the name of the dataset
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

    # get the device on which tensors will be allocated
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # update the num_workers parameters according to the device
    num_workers = 0 if device.type == "cuda" else num_workers

    # initialize the dataloaders
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

    # get the number of category classes
    num_classes = len(train_set.classes)

    # set the number of color channels
    if dataset_name == "MNIST":
        num_channels = 1
    else:
        num_channels = train_set.data.shape[3]

    # get the image size
    image_size = train_set.data.shape[1]

    return train_loader, test_loader, num_classes, num_channels, image_size, device


def build_network(dataset_name , network_type, num_classes,num_channels,image_size, device):
    """build_network builds the CNN network specified by the parameters given to the function.

    Args:
        dataset_name (string): the name of the dataset that the network will train on.
        network_type (string): the architecture of the CNN network. One of 'AlexNet', 'LeNet5', 'ResNet50', 'VGGNet', 'LeNetPlusPlus' or 'MiniVGG'.
        num_classes (int): the number of category classes in the dataset.
        num_channels (int): number of channels in the image, 1 for greyscale images and 3 for RGB images.
        image_size (int): the image size of the images in the dataset
        device (torch.device): the device on which torch.Tensor will be allocated

    Raises:
        ValueError: if the 'dataset_name' is not one of 'AlexNet', 'LeNet5', 'ResNet50', 'VGGNet', 'LeNetPlusPlus' or 'MiniVGG'.

    Returns:
        nn.Module: the neural network module.
    """
    network = None
    if network_type == 'AlexNet':
        network = AlexNet(device, dataset_name,num_channels,num_classes)
    elif  network_type == 'LeNet5':
        network = LeNet5(device, dataset_name, num_channels)
    elif  network_type == 'ResNet50':
        network = ResNet50(device, num_channels,image_size, num_classes)
    elif  network_type == 'VGGNet':
        network = VGGNet(device, num_channels,num_classes,num_layers= 16)
    elif  network_type == 'LeNetPlusPlus':
         network = LeNetPlusPlus(device, num_channels,num_classes)
    elif  network_type == 'MiniVGG':
        network = MiniVGG(device, num_channels,num_classes)
    else :
        raise ValueError(
                "Unsupported network: {}. Only AlexNet, LeNet5, ResNet50, VGGNet, LeNetPlusPlus or MiniVGG are supported.".format(
                    network_type
                )
            )
            
    return network.to(device)


def build_optimizer(optimizer_type,network):
    """build_optimizer builds the optimizer specified by the parameters.

    Args:
        optimizer_type (string): the name of the optimizer. One of 'SGD', 'SGD_Momentum', 'Adam', 'NAdam', 'AdaGrad', 'AdaDelta', 'AdaMax' or 'RMSProp'.
        network (nn.Module): the network whose parameters to optimize.

    Raises:
        ValueError: if 'optimizer_type' is not one of 'SGD', 'SGD_Momentum', 'Adam', 'NAdam', 'AdaGrad', 'AdaDelta', 'AdaMax' or 'RMSProp'.

    Returns:
        torch.optim.Optimizer: the optimizer object.
    """
    optimizer = None

    # Load the hyperparameters from the configuration file
    with open('config.yaml') as file:
        hyperparameters = yaml.safe_load(file)

    if  optimizer_type == 'SGD':
        optimizer = optim.SGD(network.parameters(), lr = hyperparameters[optimizer_type]['learning_rate'])
    elif  optimizer_type == 'SGD_Momentum':
        optimizer = optim.SGD(network.parameters(), lr = hyperparameters[optimizer_type]['learning_rate'],momentum = hyperparameters[optimizer_type]['momentum'])
    elif  optimizer_type == 'Adam':
        optimizer = optim.Adam(network.parameters(),lr = hyperparameters[optimizer_type]['learning_rate'], betas = hyperparameters[optimizer_type]['betas'])
    elif  optimizer_type == 'NAdam':
        optimizer = optim.NAdam(network.parameters(),lr = hyperparameters[optimizer_type]['learning_rate'], betas = hyperparameters[optimizer_type]['betas'])
    elif  optimizer_type == 'AdaGrad':
        optimizer = optim.Adagrad(network.parameters(),lr = hyperparameters[optimizer_type]['learning_rate'])
    elif  optimizer_type == 'AdaDelta':
        optimizer = optim.Adadelta(network.parameters(),lr = hyperparameters[optimizer_type]['learning_rate'], rho = hyperparameters[optimizer_type]['decay'])
    elif  optimizer_type == 'AdaMax':
        optimizer = optim.Adamax(network.parameters(),lr = hyperparameters[optimizer_type]['learning_rate'], betas = hyperparameters[optimizer_type]['betas'])
    elif  optimizer_type == 'RMSProp':
        optimizer = optim.RMSprop(network.parameters(),lr = hyperparameters[optimizer_type]['learning_rate'], alpha = hyperparameters[optimizer_type]['alpha'])
    else :
            raise ValueError(
                "Unsupported optimizer: {}. Only SGD, SGD_Momentum, Adam, NAdam, AdaGrad, AdaDelta,AdaMax, RMSProp are supported.".format(
                    optimizer_type
                )
            )
    return optimizer
        

def train_model(
    train_loader,
    network,
    optimizer,
    device,
    scheduler=None,
    criterion=nn.CrossEntropyLoss(),
    max_iter=1000,
    model_name="",
):
    """The training loop for the model.

    Args:
        train_loader (DataLoader): iterable that iterates through the training set of the dataset.
        network (nn.Module): the neural network to train.
        optimizer (torch.optim.Optimizer): the optimizer to be used while training.
        device (torch.device): the device on which torch.Tensor is/will be allocated.
        scheduler (torch.optim.lr_scheduler, optional): a learning rate scheduler. Defaults to None.
        criterion (function, optional): the loss function to be used. Defaults to nn.CrossEntropyLoss().
        max_iter (int, optional): the number of iterations to be done. Defaults to 1000.
        model_name (str, optional): the name to be used to save the model. Defaults to "".

    Returns:
        list: returns a list of the running losses per iteration.
    """
    losses = []
    for epoch in tqdm(range(max_iter)):
        running_loss = 0.0
        for _, data in enumerate(train_loader):
            # get the inputs
            inputs, labels = data

            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # update statistics
            running_loss += loss.item()

        if scheduler is not None:
            scheduler.step()  # Update the learning rate scheduler

        losses.append(running_loss)
        running_loss = 0.0
    # save the model
    PATH = os.path.join("models", "{}_network.pth".format(model_name))
    torch.save(network.state_dict(), PATH)
    return losses    


def test_model(test_loader, network, device):
    """Test the neural network on the test set.

    Args:
        test_loader (DataLoader): iterable that iterates through the test set of the dataset.
        network (nn.Module): the neural network used to classify.
        device (torch.device): the device on which torch.Tensor is/will be allocated.

    Returns:
        float: the accuracy of the network at the classification task.
    """
    correct_predictions = 0
    total_predictions = 0
    with torch.no_grad():
        for _, data in enumerate(test_loader):
            # get the inputs + labels
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            # make the predictions
            outputs = network(images)
            outputs = outputs.cpu()
            _, predictions = torch.max(outputs, 1)
            # collect the correct predictions for each class
            for label, prediction in zip(labels.cpu(), predictions):
                if label == prediction:
                    correct_predictions += 1
                total_predictions += 1
    # calculate the accuracy
    accuracy = correct_predictions / total_predictions
    return accuracy


def train_test_model(
    dataset_name,
    network_type,
    optimizer_type,
    lr_scheduler=None,
    max_iter=1000,
    batch_size=10,
    num_workers=4,
):
    """Main function to run the experiments. It that takes the names of the dataset, the network and the optimizer, then loads the training and
    test sets, trains the networks, tests it and then outputs the training trace, the accuracy and the training time.

    Args:
        dataset_name (string): the name of the dataset to be used. One of 'MNIST', 'CIFAR10' or 'ImageNet'
        network_type (string): the architecture of the CNN network. One of 'AlexNet', 'LeNet5', 'ResNet50', 'VGGNet', 'LeNetPlusPlus' or 'MiniVGG'.
        optimizer_type (string): the name of the optimizer. One of 'SGD', 'SGD_Momentum', 'Adam', 'NAdam', 'AdaGrad', 'AdaDelta', 'AdaMax' or 'RMSProp'.
        lr_scheduler (torch.optim.lr_scheduler, optional): a learning rate scheduler. Defaults to None.
        max_iter (int, optional): the number of iterations to be done. Defaults to 1000.
        batch_size (int, optional): the size of the batches. Defaults to 10.
        num_workers (int, optional): how many subprocesses to use for data loading. Defaults to 4.

    Returns:
        tuple: returns a tuple of the running losses per iteration during training (list), the accuracy on the test set (float), 
        and the time training the network took (float)
    """
    
    # load  dataset
    (
        train_loader,
        test_loader,
        num_classes,
        num_channels,
        image_size,
        device,
    ) = load_dataset(
        dataset_name=dataset_name, batch_size=batch_size, num_workers=num_workers
    )
    # build network
    network = build_network(
        dataset_name, network_type, num_classes, num_channels, image_size, device
    )
    # initialize optimizer
    optimizer = build_optimizer(optimizer_type, network)

    since = time.time()

    # train the network
    losses = train_model(
        train_loader,
        network,
        optimizer,
        device,
        lr_scheduler,
        max_iter=max_iter,
        model_name="{}_{}_{}".format(dataset_name, network_type, optimizer_type),
    )

    time_elapsed = time.time() - since

    # test the network
    accuracy = test_model(test_loader, network, device)

    return losses, accuracy, time_elapsed