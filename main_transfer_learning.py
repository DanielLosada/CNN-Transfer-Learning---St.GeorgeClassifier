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

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

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
        class_dir = os.path.join(config["folder_data_path"], class_name)

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

        

if __name__ == "__main__":

    config = {
        "batch_size": 64,
        "learning_rate": 0.001,
        "epochs": 20,
        "data_zip_name": "george_test_task.zip",
        "folder_data_path": "./data/george_test_task",
        "train_dir": "./train",
        "eval_dir": "./eval",
        "test_dir": "./test"
    }

    organize_images(config)

    #pretrained_model = vgg16(pretrained=True)
    #pretrained_model.eval()
    #pretrained_model.to(device)

    #feature_extractor = pretrained_model.features
