import torch.nn as nn
import torch


class InceptionBlock(nn.Module):

    def __init__(self, in_channels, ch1x1, ch3x3_reduce, ch3x3, ch5x5_reduce, ch5x5, pool_proj):
        super(InceptionBlock, self).__init__()

        # TODO: Assign to an instance attribute called `branch_1` an object of type Sequential.
        # The object must be constructed with 1x1 2D convolution, batch normalization, and ReLU.
        # The convolution must have a number of input channels equal to the provided number of input channels,
        # a number of output channels equal to the provided number of output channels for 1x1 convolution, and no bias.
        # Perform ReLU in place.
        self.branch_1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1, bias=False),
            nn.BatchNorm2d(ch1x1),
            nn.ReLU(inplace=True)
        )
        

        # TODO: Assign to an instance attribute called `branch_2` an object of type Sequential.
        # The object must be constructed with 1x1 2D convolution, batch normalization, and ReLU,
        # 3x3 2D convolution, batch normalization, and ReLU.
        # The first convolution must have a number of input channels equal to the provided number of input channels,
        # a number of output channels equal to `ch3x3_reduce`, and no bias.
        # The second convolution must have a number of input channels equal to `ch3x3_reduce`,
        # a number of output channels equal to the provided number of channels for 3x3 convolution,
        # padding of 1, and no bias.
        # Perform ReLU in place.  
        self.branch_2 = nn.Sequential(
                nn.Conv2d(in_channels, ch3x3_reduce, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch3x3_reduce),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch3x3_reduce, ch3x3, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(ch3x3),
                nn.ReLU(inplace=True)
            )
    

        # TODO: Assign to an instance attribute called `branch_3` an object of type Sequential.
        # The object must be constructed with 1x1 2D convolution, batch normalization, and ReLU,
        # 5x5 2D convolution, batch normalization, and ReLU.
        # The first convolution must have a number of input channels equal to the provided number of input channels,
        # a number of output channels equal to `ch5x5_reduce`, and no bias.
        # The second convolution must have a number of input channels equal to `ch5x5_reduce`,
        # a number of output channels equal to the provided number of channels for 5x5 convolution,
        # padding of 2, and no bias.
        # Perform ReLU in place.
        self.branch_3 = nn.Sequential(
                nn.Conv2d(in_channels, ch5x5_reduce, kernel_size=1, bias=False),
                nn.BatchNorm2d(ch5x5_reduce),
                nn.ReLU(inplace=True),
                nn.Conv2d(ch5x5_reduce, ch5x5, kernel_size=5, padding=2, bias=False),
                nn.BatchNorm2d(ch5x5),
                nn.ReLU(inplace=True)
            )
        

        # TODO: Assign to an instance attribute called `branch_4` an object of type Sequential.
        # The object must be constructed with 2D max pooling, 1x1 2D convolution, batch normalization, and ReLU.
        # Max pooling must have a kernel size of 3, a stride of 1, and padding of 1.
        # The convolution must have a number of input channels equal to the provided number of input channels,
        # a number of output channels equal to the provided number of output channels after pooling, and no bias.
        # Perform ReLU in place.        
        self.branch_4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, pool_proj, kernel_size=1, bias=False),
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )


    def forward(self, x):

        # TODO: Assign to a local variable called `output_1` the output of passing the provided input to branch 1.
        output_1 = self.branch_1(x)

        # TODO: Assign to a local variable called `output_2` the output of passing the provided input to branch 2.
        output_2 = self.branch_2(x)

        # TODO: Assign to a local variable called `output_3` the output of passing the provided input to branch 3.
        output_3 = self.branch_3(x)

        # TODO: Assign to a local variable called `output_4` the output of passing the provided input to branch 4.
        output_4 = self.branch_4(x)

        # TODO: Construct a list called `list_of_outputs` with the above outputs.
        list_of_outputs = [output_1, output_2, output_3, output_4]

        # Each output is a tensor with shape (number of images in batch, number of channels, height, width).
        # Return the result of concatenating outputs along their channel / first dimension.
        return torch.cat(list_of_outputs, dim=1)


class GoogLeNet(nn.Module):

    def __init__(self, num_classes=18):
        super(GoogLeNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias = False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.inception3a = InceptionBlock(64, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(480, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x