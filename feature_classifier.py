import torch.nn as nn
from linearblock import LinearBlock
from torchvision.models import vgg16, VGG16_Weights

class FeatureClassifier(nn.Module):

    def __init__(self, dropout):
        super().__init__()
        pretrained_model = vgg16(weights=VGG16_Weights.DEFAULT)

        feature_extractor = pretrained_model.features
        for layer in feature_extractor[:24]:  # Freeze layers 0 to 23
            for param in layer.parameters():
                param.requires_grad = False

        for layer in feature_extractor[24:]:  # Train layers 24 to 30
            for param in layer.parameters():
                param.requires_grad = True
        
        self.feature_extractor = feature_extractor

        self.mlp = nn.Sequential(
            LinearBlock(25088, 256),
            nn.Dropout(dropout),
            LinearBlock(256, 256),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # flatten the features
        x = self.mlp(x)
        return x
