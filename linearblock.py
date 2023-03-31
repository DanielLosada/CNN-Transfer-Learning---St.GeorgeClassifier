import torch

class LinearBlock(torch.nn.Module):

    def __init__(
            self, 
            input_size: int, 
            output_size: int) -> None:

        super().__init__()

        self.linear = torch.nn.Linear(input_size, output_size)  
        self.relu = torch.nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        x = self.relu(x)
        return x
