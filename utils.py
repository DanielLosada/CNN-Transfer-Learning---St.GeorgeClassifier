import torch
import os
import zipfile
import shutil
import random
import tqdm

from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader

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
        
    trans = transforms.Compose([
                            transforms.Resize(150), # Resize the short side of the image to 150 keeping aspect ratio
                            transforms.CenterCrop(150), # Crop a square in the center of the image
                            transforms.ToTensor(), # Convert the image to a tensor with pixels in the range [0, 1]
                            ])
    
    train_dataset = ImageFolder(config["train_dir"], trans)
    eval_dataset = ImageFolder(config["eval_dir"], trans)
    test_dataset = ImageFolder(config["test_dir"], trans)

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

    return train_loader, eval_loader, test_loader
