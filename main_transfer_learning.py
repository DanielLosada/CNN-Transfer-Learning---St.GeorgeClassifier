import torch
import os
import zipfile
import shutil
import random

from tqdm import tqdm
from torchvision.models import vgg16, VGG16_Weights
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torchvision import transforms
from utils import organize_images, train_single_epoch, validate_single_epoch, plot_loss_accuracy, save_model
from feature_classifier import FeatureClassifier

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def fine_tuning(config, train_loader, val_loader):
    pretrained_model = vgg16(weights=VGG16_Weights.DEFAULT)
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

    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []

    for epoch in range(config["epochs"]):
        print("epoch: ", epoch)
        train_loss, train_acc = train_single_epoch(model, optimizer, criterion, train_loader, (epoch, config["epochs"]))
        val_loss, val_acc = validate_single_epoch(model, criterion, val_loader, (epoch, config["epochs"]))
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    plot_loss_accuracy(train_accuracies, train_losses, val_accuracies, val_losses)
    save_model(model, "my_model_transfer.pth")


if __name__ == "__main__":

    config = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "weight_decay": 0.01,
        "epochs": 10,
        "data_zip_name": "george_test_task.zip",
        "folder_data_path": "./data",
        "train_dir": "./train",
        "eval_dir": "./eval",
        "test_dir": "./test",
        
    }

    train_loader, val_loader, test_loader = organize_images(config)
    fine_tuning(config, train_loader, val_loader)
