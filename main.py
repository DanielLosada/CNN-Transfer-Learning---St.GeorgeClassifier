import torch
import os
import zipfile
import shutil

from model import MyModel
from utils import AverageMeter, save_model, organize_images, train_single_epoch, validate_single_epoch, plot_loss_accuracy
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")




def train_model(config, train_loader, val_loader):

    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []

    my_model = MyModel().to(device)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = torch.nn.BCELoss()

    for epoch in range(config["epochs"]):
        print("epoch: ", epoch)
        train_loss, train_acc = train_single_epoch(my_model, optimizer, criterion, train_loader, (epoch, config["epochs"]))
        val_loss, val_acc = validate_single_epoch(my_model, criterion, val_loader, (epoch, config["epochs"]))
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    plot_loss_accuracy(train_accuracies, train_losses, val_accuracies, val_losses)
    save_model(my_model, "my_model.pth")

if __name__ == "__main__":

    config = {
        "batch_size": 100,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "epochs": 16,
        "mean_std": False,
        "data_zip_name": "george_test_task.zip",
        "folder_data_path": "./data/george_test_task",
        "train_dir": "./train",
        "eval_dir": "./eval",
        "test_dir": "./test",
        "mean": (0.5499, 0.4868, 0.4171),
        "std": (0.2335, 0.2315, 0.2216)
    }

    train_loader, val_loader, test_loader = organize_images(config)
    train_model(config, train_loader, val_loader)
