import torch
import argparse

from torchvision import transforms
from feature_classifier import FeatureClassifier
from utils import test_single_img, test_model, load_test_dataset_vgg16, load_config
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":

    #Load the configuration file
    config = load_config("config_feature_classifier.json")

    #Create the argument parser and define the arguments
    parser = argparse.ArgumentParser(prog='St.GeorgeClassifier VGG16 transfer learning',
                    description='Test the VGG16 transfer learning model with the test dataset or with single jpg images')
    
    parser.add_argument("--singleimg",action="store_true", help="Single img test. A jpg img will be requested")

    #Parse the command-line arguments
    args = parser.parse_args()

    #Create the model
    model = FeatureClassifier(dropout=0.25).to(device)

    # Load the best weights from the .pth file
    model.load_state_dict(torch.load(config["name_to_save_model"]))
    
    if(args.singleimg):
        print("Going to test an img. A .jpg file will be requested")

        #Transform to apply to the image
        trans = transforms.Compose([
                            transforms.Resize((224, 224)), # Resize the short side of the image to 224
                            transforms.ToTensor(), # Convert the image to a tensor with pixels in the range [0, 1]
                            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                            ])
        
        test_single_img(model, trans)

    else:
        #Test the model with the test datset
        print("Going to test the model with the test partition")

        #Define the loss function
        criterion = torch.nn.BCELoss()

        #Get the dataloader
        test_loader = load_test_dataset_vgg16(config)

        #Test the model
        test_loss, test_acc, recall_class0, recall_class1, confusion_matrix = test_model(model, criterion, test_loader)
        
        #Show the metrics on the terminal
        print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}, Recall No George: {recall_class0:.4f}, Recall George: {recall_class1:.4f}\n")
        print(f'Confusion Matrix: \nTP: {confusion_matrix["TP"]}\tFP: {confusion_matrix["FP"]}\nFN: {confusion_matrix["FN"]}\tTN: {confusion_matrix["TN"]}')

        

    

    

