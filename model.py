import torch.nn as nn
from convblock import ConvBlock
from linearblock import LinearBlock

class Smallnet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(num_inp_channels=3, num_out_fmaps=20, kernel_size=5)
        self.conv2 = ConvBlock(num_inp_channels=20, num_out_fmaps=40, kernel_size=5)
        self.conv3 = ConvBlock(num_inp_channels=40, num_out_fmaps=40, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        self.mlp = nn.Sequential(
            LinearBlock(12960, 84),
            nn.Dropout(0.5),
            nn.Linear(84, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.conv3(x)
        x = self.dropout(x)

        x = x.view(x.size(0), -1)  # flatten the output for the MLP
        x = self.mlp(x)
        return x
