# OptML_project

## Description 

This project was done for the [Optimization for machine learning](https://github.com/epfml/OptML_course) course at [EPFL](https://www.epfl.ch/fr/).
In this project, we evaluate different optimization methods in Deep Learning.

## Requirements 

This project requires the following to run :
* Python ( version => 3.10 )
* PyTorch
* tqdm


If your machine is connected to a GPU that you would like to use, you need to activate the GPU and install the corresponding version of [PyTorch](https://pytorch.org/get-started/locally/).

## Instructions 

To run the project, you need to clone the repo and then run [run.py](/run.py)

## Data sources 

The datasets used in this project are [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) and [MNIST](https://en.wikipedia.org/wiki/MNIST_database).Both were imported using builtin Pytorch modules.

### CIFAR-10 
The CIFAR-10 dataset consists of 60000 32x32 colour images equally partionned into 10 classes. The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship and truc.

Of the 60000, 50000 are used for training and 10000 are used for testing.

It is one of the most used datasets in machine learning studies.
Its purpose is to teach computers to recognise objects more specifically the 10 classes.
<figure align="center">
    <img src="./images/CIFAR-10.png" alt="CIFAR-10 Examples, Source: https://www.cs.toronto.edu/~kriz/cifar.html">
    <figcaption >CIFAR-10 Examples, Source: https://www.cs.toronto.edu/~kriz/cifar.html</figcaption>

</figure>

### MNIST
The MNIST dataset consists of 70000 greyscale images of handwritten digits, so like the CIFAR-10 , we have 10 classes. 
In this dataset, 60000 images are used for training and  10000 images are used for testing.
This dataset is used to recognise handwritten digits. 

<figure align="center">
    <img src="./images/MnistExamplesModified.png" alt="MNIST Examples, Source: https://en.wikipedia.org/wiki/MNIST_database">
    <figcaption >MNIST Examples, Source: https://en.wikipedia.org/wiki/MNIST_database </figcaption>
</figure>

## Code structure

The code is split into the following:
*  [helpers.py](helpers.py) : contains the functions that load the dataset, build the networks and optimizers,train the model and test it.Results are stored in the [result](/result) folder and the models are stored in the [models](/models) folder.
* [networks.py](networks.py): contains the classe definitions of the networks used in the project.
* [run.py](run.py): contains the code used to train 
* [results.ipynb](results.ipynb) : a notebook that loads the results and showcases the differents plots and tables used in the report.










