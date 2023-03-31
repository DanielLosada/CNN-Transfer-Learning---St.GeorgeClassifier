import torch.nn as nn
from convblock import ConvBlock
from linearblock import LinearBlock

class MyModel(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = ConvBlock(num_inp_channels=3, num_out_fmaps=6, kernel_size=5)
        self.conv2 = ConvBlock(num_inp_channels=6, num_out_fmaps=6, kernel_size=5)
        self.conv3 = ConvBlock(num_inp_channels=6, num_out_fmaps=6, kernel_size=5)

        self.mlp = nn.Sequential(
            LinearBlock(1944, 84),
            nn.Linear(84, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.view(x.size(0), -1)  # flatten the output for the MLP
        x = self.mlp(x)
        return x
