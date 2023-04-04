import torch
import argparse

from utils import train_single_epoch, validate_single_epoch, plot_loss_accuracy, save_model, load_datasets_vgg16, load_config, prepare_dataset
from feature_classifier import FeatureClassifier

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")

def fine_tuning(config, train_loader, eval_loader):
    

    # Define a stopping criterion
    best_val_loss = float("inf")
    patience = 3
    counter = 0

    model = FeatureClassifier(dropout=0.25)
    model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []
    
    for epoch in range(config["epochs"]):
        print("epoch: ", epoch)
        train_loss, train_acc = train_single_epoch(model, optimizer, criterion, train_loader, (epoch, config["epochs"]))
        val_loss, val_acc, counter, best_val_loss_after_eval = validate_single_epoch(model, criterion, eval_loader, (epoch, config["epochs"]), counter, best_val_loss, config["name_to_save_model"])
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
    plot.savefig("plot_acc_loss_train_eval_transfer_learning.jpg")
    plot.show()
    save_model(model, "model_transfer.pth")


if __name__ == "__main__":

    # Load the configuration file
    config = load_config("config_vgg16.json")

    init_mem = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"Initial GPU memory usage: {init_mem:.2f} MiB")
    torch.cuda.empty_cache()

    empty_mem = torch.cuda.memory_allocated() / (1024 * 1024)
    print(f"GPU memory usage after emptying the cache: {empty_mem:.2f} MiB")

    parser = argparse.ArgumentParser(prog='St.GeorgeClassifier VGG16 transfer learning',
                    description='Re-training of the last layers of VGG16. Using VGG16 as a feature extractor and then a classifier to classify images in two classes. St George is in the image, and St George is not in the image')
    
    parser.add_argument("--preparedataset",action="store_true", help="Prepare the dataset. Unzip the dataset. Split the dataset in train, eval, test folders")
    parser.add_argument("--all",action="store_true", help="Prepare the dataset as the previous option and start the train/eval process.")
    
    # Parse the command-line arguments
    args = parser.parse_args()

    if(args.preparedataset):
        print("Preparing the dataset")
        prepare_dataset(config)
        exit()
    elif(args.all):
        print("Preparing the dataset")
        prepare_dataset(config)
        train_loader, eval_loader = load_datasets_vgg16(config)
        fine_tuning(config, train_loader, eval_loader)
    else:
        train_loader, eval_loader = load_datasets_vgg16(config)
        fine_tuning(config, train_loader, eval_loader)
