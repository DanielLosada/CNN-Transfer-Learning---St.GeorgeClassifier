import torch
import os
import zipfile
import shutil
import matplotlib.pyplot as plt

from model import MyModel
from utils import AverageMeter, save_model
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def plot_loss_accuracy(train_accuracies, train_losses, val_accuracies, val_losses):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    ax[0].plot(train_losses, label="Train")
    ax[0].plot(val_losses, label="Validation")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Loss")
    ax[0].legend()
    ax[1].plot(train_accuracies, label="Train")
    ax[1].plot(val_accuracies, label="Validation")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Accuracy")
    ax[1].legend()
    plt.show()    

def organize_images(config):
    if not os.path.exists(config["folder_data_path"]) or not os.path.isdir(config["folder_data_path"]) or not os.listdir(config["folder_data_path"]):
        with zipfile.ZipFile(config["data_zip_name"]) as zf:
            for member in tqdm(zf.infolist(), desc='Extracting '):
                try:
                    zf.extract(member, config["folder_data_path"])
                except zipfile.error as e:
                    pass
    else:
        print("Images already extracted")
    
    # Definition of percentages of the three data splits (train, validation, test)
    train_pct = 0.7
    val_pct = 0.2

    #Creation of the dataset from the images source
    trans = transforms.Compose([
                                transforms.Resize(150), # Resize the short side of the image to 150 keeping aspect ratio
                                transforms.CenterCrop(150), # Crop a square in the center of the image
                                transforms.ToTensor(), # Convert the image to a tensor with pixels in the range [0, 1]
                                ])
    
    dataset = ImageFolder(config["folder_data_path"] + "/george_test_task", trans)

    # Split the dataset into train, validation, and test sets
    num_train = int(len(dataset) * train_pct)
    num_val = int(len(dataset) * val_pct)
    num_test = len(dataset) - num_train - num_val
    train_dataset, val_dataset, test_dataset = random_split(dataset, [num_train, num_val, num_test])

    # Creation of data loaders for the train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

    return train_loader, val_loader, test_loader

#model: model to use
#optimizer: optimizer to use
#criterion: loss function to use
#dataloader: data loader to use
#epoch_info: tuple with the current epoch and the total number of epochs
def train_single_epoch(model, optimizer, criterion, dataloader, epoch_info):
    model.train()
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    train_loop = tqdm(dataloader, unit=" batches")
    for data, target in train_loop:
        train_loop.set_description('[TRAIN] Epoch {}/{}'.format(epoch_info[0] + 1, epoch_info[1]))
        data, target = data.float().to(device), target.float().to(device)
        target = target.unsqueeze(-1)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss.update(loss.item(), n=len(target))
        pred = output.round()  # get the prediction
        acc = pred.eq(target.view_as(pred)).sum().item()/len(target) #we get the accuracy of the prediction
        train_accuracy.update(acc, n=len(target))
        train_loop.set_postfix(loss=train_loss.avg, accuracy=train_accuracy.avg)
    return train_loss.avg, train_accuracy.avg

def validate_single_epoch(model, criterion, dataloader, epoch_info):
    model.eval()
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    val_loop = tqdm(dataloader, unit=" batches")
    with torch.no_grad():
        for data, target in val_loop:
            val_loop.set_description('[VALIDATION] Epoch {}/{}'.format(epoch_info[0] + 1, epoch_info[1]))
            data, target = data.float().to(device), target.float().to(device)
            target = target.unsqueeze(-1)
            output = model(data)
            loss = criterion(output, target)
            val_loss.update(loss.item(), n=len(target))
            pred = output.round()  # get the prediction
            acc = pred.eq(target.view_as(pred)).sum().item()/len(target) #we get the accuracy of the prediction
            val_accuracy.update(acc, n=len(target))
            val_loop.set_postfix(loss=val_loss.avg, accuracy=val_accuracy.avg)
    return val_loss.avg, val_accuracy.avg

def train_model(train_loader, val_loader, test_loader, config):

    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []

    my_model = MyModel().to(device)
    optimizer = torch.optim.Adam(my_model.parameters(), lr=config["learning_rate"])
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
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 100,
        "data_zip_name": "george_test_task.zip",
        "folder_data_path": "./data"
    }

    train_loader, val_loader, test_loader = organize_images(config)
    train_model(train_loader, val_loader, test_loader, config)
