import torch
import os
import zipfile
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np

from imgaug import augmenters as iaa
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(model, path):
    torch.save(model.state_dict(), path)

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

def compute_mean_std(dataset):
    # Compute the mean and std of the dataset
    mean = 0.
    std = 0.
    print("Going to compute mean std")
    for images, _ in dataset:
        # We assume the images are in range [0, 1]
        mean += images.mean(axis=(1, 2))
        std += images.std(axis=(1, 2))

    mean /= len(dataset)
    std /= len(dataset)
    print("mean: ", mean)
    print("std: ", std)
    return mean, std

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

#model: model to use
#criterion: loss function to use
#dataloader: data loader to use
#epoch_info: tuple with the current epoch and the total number of epochs
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
    
    class_names = ["george", "no_george"]

    # Definition of percentages of the three data splits (train, validation, test)
    train_ratio = 0.7
    val_ratio = 0.20

    # Loop over the subdirectories and create train, validation, and test sets
    for class_name in class_names:
        # Set the path to the class directory
        class_dir = os.path.join(config["folder_data_path"],"george_test_task", class_name)

        # Get a list of all the image filenames in the class directory
        image_filenames = os.listdir(class_dir)

        # Filter out the filenames that don't end with .jpg
        image_filenames = [filename for filename in image_filenames if filename.endswith(".jpg")]

        # Set the random seed to a fixed value
        random.seed(1234)

        # Shuffle the image filenames randomly
        random.shuffle(image_filenames)

        # Calculate the number of images for each set based on the ratios
        num_images = len(image_filenames)
        num_train = int(num_images * train_ratio)
        num_eval = int(num_images * val_ratio)
        num_test = num_images - num_train - num_eval

        # Split the image filenames into train, validation, and test sets
        train_filenames = image_filenames[:num_train]
        eval_filenames = image_filenames[num_train:num_train+num_eval]
        test_filenames = image_filenames[num_train+num_eval:]

        # Create the train, validation, and test directories for this class
        for dirname in ["train", "eval", "test"]:
            os.makedirs(os.path.join('./', dirname, class_name), exist_ok=True)

        # Copy the images to the train, validation, and test directories
        for filename, dirname in zip(train_filenames , ["train"]*num_train):
            src_path = os.path.join(class_dir, filename)
            dst_path = os.path.join('./', dirname, class_name, filename)
            shutil.copyfile(src_path, dst_path)
        
        for filename, dirname in zip(eval_filenames , ["eval"]*num_train):
            src_path = os.path.join(class_dir, filename)
            dst_path = os.path.join('./', dirname, class_name, filename)
            shutil.copyfile(src_path, dst_path)

        for filename, dirname in zip(test_filenames , ["test"]*num_train):
            src_path = os.path.join(class_dir, filename)
            dst_path = os.path.join('./', dirname, class_name, filename)
            shutil.copyfile(src_path, dst_path)
    
    if "mean_std" in config and config["mean_std"]:
        trans_to_tensor = transforms.Compose([
                            transforms.ToTensor(), # Convert the image to a tensor with pixels in the range [0, 1]
                            ])
        dataset = ImageFolder(config["train_dir"], trans_to_tensor)
        mean, std = compute_mean_std(dataset)
        print("Mean: ", mean)
        print("Std: ", std)

    trans = transforms.Compose([
                            transforms.Resize((150, 150)), # Resize the short side of the image to 150 keeping aspect ratio
                            transforms.CenterCrop(150), # Crop a square in the center of the image
                            transforms.ToTensor(), # Convert the image to a tensor with pixels in the range [0, 1]
                            #transforms.Normalize(config["mean"], config["std"]) # Normalize the image with the mean and std computed above
                            ])
    
    trans_train = transforms.Compose([
        transforms.Resize((150,150)), # Resize the short side of the image to 150 keeping aspect ratio
        transforms.CenterCrop(150),
        np.asarray,
        iaa.Sequential([
            iaa.flip.Fliplr(p=0.5),
            iaa.flip.Flipud(p=0.5),
            iaa.Sometimes(0.2, [
            iaa.Grayscale(),
            ]),
            iaa.AverageBlur(k = (0,5)),
            iaa.MultiplyBrightness(mul=(0.65, 1.35)),
        ], random_order = True).augment_image,
        #save_img,
        np.copy,
        transforms.ToTensor()
    ])
    
    if "mean" in config and "std" in config:
        trans.transforms.append(transforms.Normalize(config["mean"], config["std"]))
    
    train_dataset = ImageFolder(config["train_dir"], trans_train)
    eval_dataset = ImageFolder(config["eval_dir"], trans)
    test_dataset = ImageFolder(config["test_dir"], trans)


    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

    return train_loader, eval_loader, test_loader

def save_img(x):
    from torchvision.utils import save_image
    from PIL import Image
    import uuid

    im = Image.fromarray(x)
    im.save(f"imgs/img{str(uuid.uuid4())}.png")
    return x 