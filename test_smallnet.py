import torch
import argparse
import json

from torchvision import transforms
from model import MyModel
from utils import test_single_img, test_model, load_test_dataset_smallnet
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    # Load the configuration file
    with open('config_smallnet.json', 'r') as f:
        config = json.load(f)

    parser = argparse.ArgumentParser(prog='St.GeorgeCkassufier Baseline Test',
                    description='Test the baseline model with the test dataset or with single jpg images')
    
    parser.add_argument("--singleimg",action="store_true", help="Single img test. A jpg img will be requested")

    # Parse the command-line arguments
    args = parser.parse_args()

    my_model = MyModel().to(device)
    # Load the best weights from the .pth file
    my_model.load_state_dict(torch.load('./best_model_smallnet.pth'))
    # # Set the model to evaluation mode
    # my_model.eval()

    test_loader = load_test_dataset_smallnet(config)


    # Load the image and preprocess it
    trans = transforms.Compose([
        transforms.Resize((150, 150)),
        transforms.ToTensor(),
    ])

    if "mean" in config and "std" in config:
        trans.transforms.append(transforms.Normalize(config["mean"], config["std"]))

    if(args.singleimg):
        print("Going to test an img. A .jpg file will be requested")
        test_single_img(my_model, trans)
    else:
        print("Going to test the model with the test partition")
        test_accuracies, test_losses = [], []
        criterion = torch.nn.BCELoss()

        test_loss, test_acc = test_model(my_model, criterion, test_loader)
        print("test_loss: ", test_loss)
        print("test_acc: ", test_acc)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        

    

    

