import torch
import argparse
import json

from model import Smallnet
from utils import train_single_epoch, validate_single_epoch, plot_loss_accuracy, prepare_dataset, calculate_mean_std, load_datasets_smallnet

#If cuda is available, we use it for better performance
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")



def train_model(config, train_loader, val_loader):

    #Define a stopping criterion, if loss doesn't improve in 'patience' epochs, we stop the training
    best_val_loss = float("inf")
    patience = 3
    counter = 0

    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []

    #Define the model
    model = Smallnet().to(device)

    #Define the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    #Define the loss function
    criterion = torch.nn.BCELoss()

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_single_epoch(model, optimizer, criterion, train_loader, (epoch, config["epochs"]))
        val_loss, val_acc, counter, best_val_loss_after_eval = validate_single_epoch(model, criterion, val_loader, (epoch, config["epochs"]), counter, best_val_loss,  config["name_to_save_model"])
        
        #Save the accuracy and loss of the training and validation
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)

        #Show the metrics on the terminal
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        #Uptate the best loss we have till the moment
        best_val_loss = best_val_loss_after_eval

        #Apply the early stop criterion
        if(counter >= patience):
            print(f"Validation loss did not improve for {patience} epochs. Stopping.")
            break
    
    #Generate, save, and show the accuracy and loss graphs of the train validate process
    plot = plot_loss_accuracy(train_accuracies, train_losses, val_accuracies, val_losses)
    plot.savefig("plot_acc_loss_train_eval_smallnet.jpg")
    plot.show()

if __name__ == "__main__":

    #Load the configuration file
    with open('config_smallnet.json', 'r') as f:
        config = json.load(f)

    #Create the argument parser and define the arguments
    parser = argparse.ArgumentParser(prog='St.GeorgeCkassufier Smallnet',
                    description='Train a small network to classify images in two classes. St George is in the image, and St George is not in the image')
    parser.add_argument("--preparedataset",action="store_true", help="Prepare the dataset. Unzip the dataset. Split the dataset in train, eval, test folders")
    parser.add_argument("--calcmeanstd",action="store_true", help="Calculate the mean and std of the dataset.")
    parser.add_argument("--all",action="store_true", help="Prepare the dataset and calculate the mean and std of the dataset. And start the train/eval process.")
    
    #Parse the command-line arguments
    args = parser.parse_args()

    if(args.preparedataset):
        print("Preparing the dataset")
        prepare_dataset(config)
        exit()

    elif(args.calcmeanstd):
        print("Calculating the mean and the std of the dataset")
        calculate_mean_std(config)
        exit()
    elif(args.all):
        print("Preparing the dataset")
        prepare_dataset(config)
        print("Calculating the mean and the std of the dataset")
        calculate_mean_std(config)
        train_loader, val_loader = load_datasets_smallnet(config)
        train_model(config, train_loader, val_loader)
    else:
        train_loader, val_loader = load_datasets_smallnet(config)
        train_model(config, train_loader, val_loader)
