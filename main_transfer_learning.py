import torch

from torchvision.models import vgg16, VGG16_Weights
from utils import train_single_epoch, validate_single_epoch, plot_loss_accuracy, save_model, load_datasets_vgg16
from feature_classifier import FeatureClassifier

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#device = torch.device("cpu")

def fine_tuning(config, train_loader, eval_loader):
    pretrained_model = vgg16(weights=VGG16_Weights.DEFAULT)
    pretrained_model.to(device)

    feature_extractor = pretrained_model.features
    for layer in feature_extractor[:24]:  # Freeze layers 0 to 23
        for param in layer.parameters():
            param.requires_grad = False

    for layer in feature_extractor[24:]:  # Train layers 24 to 30
        for param in layer.parameters():
            param.requires_grad = True

    # Define a stopping criterion
    best_val_loss = float("inf")
    patience = 3
    counter = 0

    model = FeatureClassifier(feature_extractor, dropout=0.25)
    model.to(device)

    criterion = torch.nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])

    train_accuracies, train_losses, val_accuracies, val_losses = [], [], [], []
    
    for epoch in range(config["epochs"]):
        print("epoch: ", epoch)
        train_loss, train_acc = train_single_epoch(model, optimizer, criterion, train_loader, (epoch, config["epochs"]))
        val_loss, val_acc, counter, best_val_loss_after_eval = validate_single_epoch(model, criterion, eval_loader, (epoch, config["epochs"]), counter, best_val_loss)
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
    plot.show()
    save_model(model, "my_model_transfer.pth")


if __name__ == "__main__":

    config = {
        "batch_size": 20,
        "learning_rate": 0.001,
        "weight_decay": 0.001,
        "epochs": 20,
        "data_zip_name": "george_test_task.zip",
        "folder_data_path": "./data",
        "train_dir": "./train",
        "eval_dir": "./eval",
        "test_dir": "./test"
    }

    train_loader, eval_loader = load_datasets_vgg16(config)
    fine_tuning(config, train_loader, eval_loader)
