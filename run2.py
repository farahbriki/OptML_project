import os
import pickle
from helpers2 import *
Models = {
    "CIFAR10": ["VGGNet", "ResNet50", "AlexNet"],
}
Optimizers = [
    "SGD",
    "SGD_Momentum",
    "Adam",
    "NAdam",
    "AdaGrad",
    "AdaDelta",
    "AdaMax",
    "RMSProp",
]

os.makedirs("result", exist_ok=True)
for dataset in list(Models.keys()):
    for model in Models[dataset]:
        for optimizer in Optimizers:
            losses, accuracy, time_elapsed = train_test_model(
                dataset, model, optimizer, batch_size=128, max_iter=50
            )
            result_file = "{}_{}_{}.pkl".format(dataset, model, optimizer)
            file_path = os.path.join("result", result_file)
            with open(file_path, "wb") as f:
                pickle.dump((losses, accuracy, time_elapsed), f)