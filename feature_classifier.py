import torch.nn as nn
from linearblock import LinearBlock

class FeatureClassifier(nn.Module):

    def __init__(self, feature_extractor, dropout):
        super().__init__()
        
        self.feature_extractor = feature_extractor

        self.mlp = nn.Sequential(
            LinearBlock(25088, 256),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view(x.size(0), -1)  # flatten the features
        x = self.mlp(x)
        return x
