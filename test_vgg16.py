import torch
import argparse
import json

from torchvision import transforms
from feature_classifier import FeatureClassifier
from utils import test_single_img, test_model, load_test_dataset_vgg16, load_config
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":

    # Load the configuration file
    config = load_config("config_vgg16.json")

    parser = argparse.ArgumentParser(prog='St.GeorgeClassifier VGG16 transfer learning',
                    description='Test the VGG16 transfer learning model with the test dataset or with single jpg images')
    
    parser.add_argument("--singleimg",action="store_true", help="Single img test. A jpg img will be requested")

    # Parse the command-line arguments
    args = parser.parse_args()

    model = FeatureClassifier(dropout=0.25).to(device)
    # Load the best weights from the .pth file
    model.load_state_dict(torch.load('./best_model_vgg16.pth'))
    
    # # Set the model to evaluation mode
    # model.eval()

    

    if(args.singleimg):
        print("Going to test an img. A .jpg file will be requested")

        trans = transforms.Compose([
                            transforms.Resize((224, 224)), # Resize the short side of the image to 224
                            transforms.ToTensor(), # Convert the image to a tensor with pixels in the range [0, 1]
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
        
        test_single_img(model, trans)
    else:
        print("Going to test the model with the test partition")
        test_accuracies, test_losses = [], []
        criterion = torch.nn.BCELoss()
        test_loader = load_test_dataset_vgg16(config)

        test_loss, test_acc = test_model(model, criterion, test_loader)
        print("test_loss: ", test_loss)
        print("test_acc: ", test_acc)
        test_accuracies.append(test_acc)
        test_losses.append(test_loss)
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")
        

    

    

