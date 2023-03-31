import torch

class ConvBlock(torch.nn.Module):

    def __init__(
            self, 
            num_inp_channels: int, 
            num_out_fmaps: int,
            kernel_size: int, 
            pool_size: int=2,
            padding: int = None) -> None:
        super().__init__()

        if padding is None:
            padding = (kernel_size - 1) // 2

        self.conv = torch.nn.Conv2d(
            in_channels=num_inp_channels, 
            out_channels=num_out_fmaps, 
            kernel_size=kernel_size,
            padding=padding)
        self.relu = torch.nn.ReLU()
        self.maxpool = torch.nn.MaxPool2d(kernel_size=pool_size, stride=pool_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.maxpool(self.relu(self.conv(x)))
