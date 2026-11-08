import torch.nn as nn
import torch


class NiNBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(NiNBlock, self).__init__()
        # TODO: Create an empty list of layers.
        layers = []
        # TODO: Add to the list of layers 2D convolution, 2D batch normalization, and ReLU.
        # The convolution must have the provided number of input channels,
        # number of output channels, kernel size, stride, and padding.
        # The convolution must have no bias.
        # Perform ReLU in place.
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size = kernel_size, stride =stride, padding = padding, bias=False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace= True))
        # TODO: Add to the list of layers 2D convolution, batch normalization, and ReLU.
        # The convolution must have a number of input channels equal to the provided number of output channels,
        # a number of output channels equal to the provided number of output channels, a kernel size of 1, and no bias.
        # Perform ReLU in place.
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size =1, bias= False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        # TODO: Add to the list of layers 2D convolution, batch normalization, and ReLU.
        # The convolution must have a number of input channels equal to the provided number of output channels,
        # a number of output channels equal to the provided number of output channels, a kernel size of 1, and no bias.
        # Perform ReLU in place.
        layers.append(nn.Conv2d(out_channels, out_channels, kernel_size= 1, bias = False))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace= True))
        # TODO: Add layers to an object of type Sequential.
        # Assign that object to an instance attribute called sequential.
        self.sequential = nn.Sequential(*layers)


    def forward(self, x):
        # TODO: Return the output of passing the provided input to sequential.
        return self.sequential(x)

class NiN(nn.Module):
    def __init__(self, num_classes= 18):
        super(NiN, self).__init__()
        # TODO: Create an empty list of layers.
        layers = []
        # TODO: Add to the list of layers a NiN block with 3 input channels,
        # 96 output channels, a kernel size of 11, and a stride of 4.
        layers.append(NiNBlock(3, 96, kernel_size= 11, stride = 4, padding = 0))
        # TODO: Add to the list of layers 2D max pooling with a kernel size of 3 and a stride of 2.
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))
        # TODO: Add to the list of layers a NiN block with 96 input channels,
        # 256 output channels, a kernel size of 5, and padding of 2.
        layers.append(NiNBlock(96, 256, kernel_size=5, stride=1, padding=2))

        # TODO: Add to the list of layers 2D max pooling with a kernel size of 3 and a stride of 2.
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        # TODO: Add to the list of layers a NiN block with 256 input channels,
        # 384 output channels, a kernel size of 3, and padding of 1.
        layers.append(NiNBlock(256, 384, kernel_size=3, stride=1, padding=1))
        
        # TODO: Add to the list of layers 2D max pooling with a kernel size of 3 and a stride of 2.
        layers.append(nn.MaxPool2d(kernel_size=3, stride=2))

        # TODO: Add to the list of layers a NiN block with 384 input channels,
        # a number of output channels equal to the provided number of classes, a kernel size of 3, and padding of 1.
        layers.append(NiNBlock(384, num_classes, kernel_size=3, stride=1, padding=1))

        # TODO: Add to the list of layers 2D average pooling with height of 1 and width of 1.
        layers.append(nn.AdaptiveAvgPool2d((1, 1)))

        # TODO: Add layers to an object of type Sequential.
        # Assign that object to an instance attribute called sequential.
        self.sequential = nn.Sequential(*layers)
    def forward(self, x):
        # TODO: Assign to a local variable called intermediate
        # the output of passing the provided input to sequential.
        intermediate = self.sequential(x)
        # TODO: Return the output of flattening intermediate from start dimension 1 on.
        # The output tensor of flattening has shape (number of images in batch, number of classes).
        return torch.flatten(intermediate, start_dim=1)