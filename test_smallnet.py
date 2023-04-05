import torch
import argparse

from torchvision import transforms
from model import MyModel
from utils import test_single_img, test_model, load_test_dataset_smallnet, load_config
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":

    #Load the configuration file
    config = load_config("config_smallnet.json")

    #Create the argument parser and define the arguments
    parser = argparse.ArgumentParser(prog='St.GeorgeCkassufier Smallnet Test',
                    description='Test the smallnet model with the test dataset or with single jpg images')
    
    parser.add_argument("--singleimg",action="store_true", help="Single img test. A jpg img will be requested")

    #Parse the command-line arguments
    args = parser.parse_args()

    #Create the model
    model = MyModel().to(device)

    #Load the best weights from the .pth file
    model.load_state_dict(torch.load('./best_model_smallnet.pth'))

    if(args.singleimg):
        print("Going to test an img. A .jpg file will be requested")

        #Transform to apply to the image
        trans = transforms.Compose([
            transforms.Resize((150, 150)),
            transforms.ToTensor(),
        ])

        #If the mean and std are specified in the config file, add the normalization transform
        if "mean" in config and "std" in config:
            trans.transforms.append(transforms.Normalize(config["mean"], config["std"]))

        test_single_img(model, trans)
    else:
        #Test the model with the test datset
        print("Going to test the model with the test partition")
        
        #Define the loss function
        criterion = torch.nn.BCELoss()

        #Get the dataloader
        test_loader = load_test_dataset_smallnet(config)

        #Test the model
        test_loss, test_acc = test_model(model, criterion, test_loader)

        #Show the metrics on the terminal
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        

    

    

