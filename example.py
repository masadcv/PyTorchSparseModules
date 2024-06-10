import torch.nn as nn
import torch

from torchsparsemodules import MaskedSparseConv2d, MaskedSparseLinear, replace_layers

if __name__ == "__main__":

    # Create a model with Conv2d, and Linear layers
    # it takes as input image of size 1, 1, 128, 128 and outputs two classes
    # three conv2d layers followed by three linear layers
    class MyModel(nn.Module):
        def __init__(self):
            super(MyModel, self).__init__()
            self.conv_layers = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3),
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
                nn.MaxPool2d(kernel_size=2),
            )
            self.fully_connected_layers = nn.Sequential(
                nn.Linear(64 * 14 * 14, 128),
                nn.Linear(128, 64),
                nn.Linear(64, 2),
            )
        
        def forward(self, x):
            x = self.conv_layers(x)
            x = torch.flatten(x, 1)
            x = self.fully_connected_layers(x)
            return x

    image = torch.rand(size=(1, 1, 128, 128))
    model = MyModel()
    print(model)
    output = model(image)

    # Replace all Conv2d layers with MaskedSparseConv2d
    # and all Linear layers with MaskedSparseLinear
    replace_layers(model, {nn.Conv2d: MaskedSparseConv2d, nn.Linear: MaskedSparseLinear})
    print(model)
    output2 = model(image)

    # all close assert on output and output2
    assert torch.allclose(output, output2)