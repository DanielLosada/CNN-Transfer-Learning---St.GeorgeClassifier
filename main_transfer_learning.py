import torch
import os
import zipfile
import shutil
import random

from tqdm import tqdm
from torchvision.models import vgg16
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from utils import organize_images
from feature_classifier import FeatureClassifier

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def fine_tuning(config):
    pretrained_model = vgg16(pretrained=True)
    pretrained_model.to(device)

    feature_extractor = pretrained_model.features
    for layer in feature_extractor[:24]:  # Freeze layers 0 to 23
        for param in layer.parameters():
            param.requires_grad = False

    for layer in feature_extractor[24:]:  # Train layers 24 to 30
        for param in layer.parameters():
            param.requires_grad = True

    model = FeatureClassifier(feature_extractor, dropout=0.5)
    model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])



if __name__ == "__main__":

    config = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 20,
        "data_zip_name": "george_test_task.zip",
        "folder_data_path": "./data/george_test_task",
        "train_dir": "./train",
        "eval_dir": "./eval",
        "test_dir": "./test"
    }

    organize_images(config)
   
