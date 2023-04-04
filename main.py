import torch
import argparse
import json

from model import MyModel
from utils import save_model, organize_images, train_single_epoch, validate_single_epoch, plot_loss_accuracy, prepare_dataset, calculate_mean_std, load_datasets_smallnet

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")



def train_model(config, train_loader, val_loader):

    # Define a stopping criterion
    best_val_loss = float("inf")
    patience = 3
    counter = 0

    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []

    model = MyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
    criterion = torch.nn.BCELoss()

    for epoch in range(config["epochs"]):
        train_loss, train_acc = train_single_epoch(model, optimizer, criterion, train_loader, (epoch, config["epochs"]))
        val_loss, val_acc, counter, best_val_loss_after_eval = validate_single_epoch(model, criterion, val_loader, (epoch, config["epochs"]), counter, best_val_loss)
        train_accuracies.append(train_acc)
        train_losses.append(train_loss)
        val_accuracies.append(val_acc)
        val_losses.append(val_loss)
        print(f"Epoch {epoch+1}/{config['epochs']}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        best_val_loss = best_val_loss_after_eval
        if(counter >= patience):
            print(f"Validation loss did not improve for {patience} epochs. Stopping.")
            break
    
    plot = plot_loss_accuracy(train_accuracies, train_losses, val_accuracies, val_losses)
    plot.savefig("plot_acc_loss_train_eval_smallnet.jpg")
    plot.show()
    save_model(model, "model_smallnet.pth")

if __name__ == "__main__":

    # Load the configuration file
    with open('config_smallnet.json', 'r') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(prog='St.GeorgeCkassufier Smallnet',
                    description='Train a small network to classify images in two classes. St George is in the image, and St George is not in the image')
    parser.add_argument("--preparedataset",action="store_true", help="Prepare the dataset. Unzip the dataset. Split the dataset in train, eval, test folders")
    parser.add_argument("--calcmeanstd",action="store_true", help="Calculate the mean and std of the dataset.")
    parser.add_argument("--all",action="store_true", help="Prepare the dataset and calculate the mean and std of the dataset. And start the train/eval process.")
    # Parse the command-line arguments
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
    

    #train_loader, val_loader, test_loader = organize_images(config)
    #train_model(config, train_loader, val_loader)
