import torch
import os
import json
import zipfile
import shutil
import random
import matplotlib.pyplot as plt
import numpy as np

from tkinter import filedialog
from imgaug import augmenters as iaa
from tqdm import tqdm
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")

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
    """Save the model state"""
    torch.save(model.state_dict(), path)
    
def load_config(config_path):
    """Load the configuration file"""
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def recall(outputs, targets):
    """Calculate the recall of the model"""
    tp = torch.sum((outputs == 1) & (targets == 1)).item()
    fn = torch.sum((outputs == 0) & (targets == 1)).item()
    recall = tp / (tp + fn)
    return recall

def plot_loss_accuracy(train_accuracies, train_losses, val_accuracies, val_losses):
    """Generates the plot of the accuracy and loss of the train and validation process."""
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
    return plt
    #plt.show() 

def test_single_img(model, trans):
    """Test a single image. Asks the user to upload a jpg image, 
        apply the transformation 'trans', and run the model to get a prediction
        Args:
            model: model to use
            trans: transform to apply"""
    
    # Open a file dialog box to allow the user to select an image
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=(("jpeg files","*.jpg"),("all files","*.*")))

    # Print the file path
    print("Selected file:", file_path)

    # Load the image and preprocess it
    image = Image.open(file_path)
    image_tensor = trans(image)
    image_batch = image_tensor.unsqueeze(0)
    image_batch = image_batch.to(device)

    # Define the classes names
    classes = ["George", "No George"]

    
    with torch.no_grad():
        # Pass the image to the model
        output = model(image_batch)

        # Get the prediction
        if(int(output.round())):
            print("The prediction is that the image belongs to 'No George' class with a probability of ", output.item())
        else:
            print("The prediction is that the image belongs to 'George' class with a probability of ", 1 - output.item())

def calculate_mean_std(config):
    """Calculate the mean and std of the dataset
    Args:
        config: configuration dictionary"""

    trans_to_tensor = transforms.Compose([
                            transforms.ToTensor(), # Convert the image to a tensor with pixels in the range [0, 1]
                            ])
    dataset = ImageFolder(config["train_dir"], trans_to_tensor)

    #Check if we already have that information to save time
    if "mean" in config and "std" in config:
        print("We already have the mean and std: ")
        print("Mean: ", config["mean"])
        print("Std: ", config["std"])
    else:
        mean = 0.
        std = 0.
        print("Going to compute mean std")
        for images, _ in dataset:
            mean += images.mean(axis=(1, 2))
            std += images.std(axis=(1, 2))

        mean /= len(dataset)
        std /= len(dataset)
        print("Mean: ", mean)
        print("Std: ", std)

        
        with open("config_smallnet.json", "w") as f:
            print("Saving mean and std in the config_smallnet.json")
            config["mean"] = mean.numpy().tolist()
            config["std"] = std.numpy().tolist()
            json.dump(config, f, indent=4)


def train_single_epoch(model, optimizer, criterion, dataloader, epoch_info):
    """Train a single epoch.
    Args:
        model: model to use
        optimizer: optimizer to use
        criterion: loss function to use
        dataloader: data loader to use
        epoch_info: tuple with the current epoch and the total number of epochs
        """
    
    #Set the model to train mode
    model.train()

    #Initialize the meters
    train_loss = AverageMeter()
    train_accuracy = AverageMeter()
    train_loop = tqdm(dataloader, unit=" batches")

    for data, target in train_loop:
        #Set description to the progress bar on the terminal
        train_loop.set_description('[TRAIN] Epoch {}/{}'.format(epoch_info[0] + 1, epoch_info[1]))
        
        data, target = data.float().to(device), target.float().to(device)
        target = target.unsqueeze(-1)

        #Set gradients to None
        optimizer.zero_grad()

        #Forward pass
        output = model(data)
        loss = criterion(output, target)

        #Backward pass
        loss.backward()
        optimizer.step()

        #Update the meters
        train_loss.update(loss.item(), n=len(target))
        pred = output.round()  # get the prediction
        acc = pred.eq(target.view_as(pred)).sum().item()/len(target) #we get the accuracy of the prediction
        train_accuracy.update(acc, n=len(target))
        train_loop.set_postfix(loss=train_loss.avg, accuracy=train_accuracy.avg)

    return train_loss.avg, train_accuracy.avg

def validate_single_epoch(model, criterion, dataloader, epoch_info, counter, best_val_loss, name_to_save_model = "best_model.pth"):
    """Validate single epoch.
    Args:
        model: model to use
        criterion: loss function to use
        dataloader: data loader to use
        epoch_info: tuple with the current epoch and the total number of epochs"""
    
    #Set the model to validation mode
    model.eval()

    #Initialize the meters
    val_loss = AverageMeter()
    val_accuracy = AverageMeter()
    val_loop = tqdm(dataloader, unit=" batches")
    with torch.no_grad():
        for data, target in val_loop:
            #Set description to the progress bar on the terminal
            val_loop.set_description('[VALIDATION] Epoch {}/{}'.format(epoch_info[0] + 1, epoch_info[1]))
            data, target = data.float().to(device), target.float().to(device)
            target = target.unsqueeze(-1)

            #Forward pass
            output = model(data)
            loss = criterion(output, target)

            #Update the meters
            val_loss.update(loss.item(), n=len(target))
            pred = output.round()  # get the prediction
            acc = pred.eq(target.view_as(pred)).sum().item()/len(target) #we get the accuracy of the prediction
            val_accuracy.update(acc, n=len(target))
            val_loop.set_postfix(loss=val_loss.avg, accuracy=val_accuracy.avg)

    print("Current validation loss avg: ", val_loss.avg, type(val_loss.avg))
    print("Best validation loss avg obtained: ", best_val_loss, type(best_val_loss))
    
    #Save the model if the loss improves the best loss obtained till the moment
    if(val_loss.avg < best_val_loss):
        best_val_loss = val_loss.avg
        torch.save(model.state_dict(), name_to_save_model)
        counter = 0
    #If not improving, increase the counter that will be compared with the 'patience' for early stop
    else:
        counter += 1
    print("Counter: ", counter)
    return val_loss.avg, val_accuracy.avg, counter, best_val_loss


def test_model(model, criterion, dataloader):
    """Test the model.
    Args:
        model: model to use
        criterion: loss function to use
        dataloader: data loader to use"""
    
    #Set the model to validation mode
    model.eval()

    #Initialize the meters
    test_loss = AverageMeter()
    test_accuracy = AverageMeter()
    test_loop = tqdm(dataloader, unit=" batches")
    with torch.no_grad():
        for data, target in test_loop:
            #Set description to the progress bar on the terminal
            test_loop.set_description('[TEST]')

            data, target = data.float().to(device), target.float().to(device)
            target = target.unsqueeze(-1)

            #Forward pass
            output = model(data)
            loss = criterion(output, target)

            #Update the meters
            test_loss.update(loss.item(), n=len(target))
            pred = output.round()  # get the prediction
            acc = pred.eq(target.view_as(pred)).sum().item()/len(target) #we get the accuracy of the prediction
            test_accuracy.update(acc, n=len(target))
            test_loop.set_postfix(loss=test_loss.avg, accuracy=test_accuracy.avg)
    
    return test_loss.avg, test_accuracy.avg

def prepare_dataset(config):
    """Prepare the dataset.
    Args:
        config: configuration dictionary"""
    
    # Check if the images are already extracted. If not extract the zip file
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
    train_ratio = 0.70
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


    

def load_datasets_smallnet(config):
    """Load the datasets.
    Args:
        config: configuration dictionary"""
    
    #Transform to apply to the images
    trans = transforms.Compose([
                            transforms.Resize((150)), # Resize the short side of the image to 150 keeping aspect ratio
                            transforms.CenterCrop(150), # Crop a square in the center of the image
                            transforms.ToTensor(), # Convert the image to a tensor with pixels in the range [0, 1]
                            ])
    
    #If the mean and std are specified in the config file, add the normalization transform
    if "mean" in config and "std" in config:
        trans.transforms.append(transforms.Normalize(config["mean"], config["std"]))
    
    #Load the datasets
    train_dataset = ImageFolder(config["train_dir"], trans)
    eval_dataset = ImageFolder(config["eval_dir"], trans)

    #Create the dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=True)

    return train_loader, eval_loader

def load_test_dataset_smallnet(config):
    """Load the test dataset for the smallnet.
    Args:
        config: configuration dictionary"""
    
    trans = transforms.Compose([
                            transforms.Resize((150)), # Resize the short side of the image to 150 keeping aspect ratio
                            transforms.CenterCrop(150), # Crop a square in the center of the image
                            transforms.ToTensor(), # Convert the image to a tensor with pixels in the range [0, 1]
                            ])
    
    #If the mean and std are specified in the config file, add the normalization transform
    if "mean" in config and "std" in config:
        trans.transforms.append(transforms.Normalize(config["mean"], config["std"]))

    #Load dataset
    test_dataset = ImageFolder(config["test_dir"], trans)

    #Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
    return test_loader

def load_datasets_vgg16(config):
    """Load the datasets.
    Args:
        config: configuration dictionary"""
    
    #Transform to apply to the eval images
    trans = transforms.Compose([
                            transforms.Resize((224, 224)), # Resize the short side of the image to 224
                            transforms.ToTensor(), # Convert the image to a tensor with pixels in the range [0, 1]
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])

    #Transform to apply to the train images. Will apply random transformation for data augmentation
    transform_augmentation = transforms.Compose([
        transforms.Resize((224,224)), # Resize the short side of the image to 224 
        #transforms.CenterCrop(224),
        np.asarray,
        iaa.Sequential([
            iaa.flip.Fliplr(p=0.5),
            iaa.flip.Flipud(p=0.5),
            iaa.Sometimes(0.5, [
                iaa.Affine(rotate=(-20, 20), scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                iaa.Crop(percent=(0, 0.3)),
            ]),
            iaa.Sometimes(0.1, [
                iaa.Grayscale(),
                #iaa.pillike.Affine(rotate=(-20, 20), scale={"x": (0.8, 1.2), "y": (0.8, 1.2)}),
                iaa.ElasticTransformation(alpha=50.0, sigma=5.0),
                iaa.AdditiveGaussianNoise(scale=0.1*255),
                iaa.Dropout(p=(0, 0.2)),
                iaa.MultiplyBrightness((0.5, 1.5)),
                iaa.GaussianBlur(sigma=(0.0, 3.0)),
                iaa.Sharpen(alpha=1.0)
            ]),
            #iaa.AverageBlur(k = (0,5)),
            #iaa.MultiplyBrightness(mul=(0.65, 1.35)),
        ], random_order = True).augment_image,
        #save_img,
        np.copy,
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    #Load datasets
    train_dataset = ImageFolder(config["train_dir"], transform_augmentation)
    eval_dataset = ImageFolder(config["eval_dir"], trans)

    #Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=True)

    return train_loader, eval_loader

def load_test_dataset_vgg16(config):
    """Load the test dataset for the VGG16 model.
    Args:
        config: configuration dictionary"""
    
    #Transform to apply to the images
    trans = transforms.Compose([
                            transforms.Resize((224, 224)), # Resize the short side of the image to 224
                            transforms.ToTensor(), # Convert the image to a tensor with pixels in the range [0, 1]
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
    
    #Create the dataset
    test_dataset = ImageFolder(config["test_dir"], trans)

    #Create the dataloader
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)
    return test_loader


def save_img(x):
    "Save an image"
    from torchvision.utils import save_image
    from PIL import Image
    import uuid

    im = Image.fromarray(x)
    im.save(f"imgs/img{str(uuid.uuid4())}.png")
    return x 