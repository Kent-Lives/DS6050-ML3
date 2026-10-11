import torch.nn as nn
import torch


class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        # TODO: Assign to an instance attribute called `convolution_1` a 3x3 2D convolution with
        # a number of input channels equal to the provided number of input channels,
        # a number of output channels equal to the provided number of output channels,
        # a stride equal to the provided stride, padding of 1, and no bias.
        self.convolution_1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        # TODO: Assign to an instance attribute called `batch_normalization_1` 2D batch normalization.
        self.batch_normalization_1 = nn.BatchNorm2d(out_channels)

        # TODO: Assign to an instance attribute called `relu` ReLU. ReLU must be performed in place.
        self.relu = nn.ReLU(inplace=True)

        # TODO: Assign to an instance attribute called `convolution_2` a 3x3 2D convolution with
        # a number of input channels equal to the provided number of output channels,
        # a number of output channels equal to the provided number of output channels,
        # a stride of 1, padding of 1, and no bias.
        self.convolution_2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        # TODO: Assign to an instance attribute called `batch_normalization_2` 2D batch normalization.
        self.batch_normalization_2 = nn.BatchNorm2d(out_channels)

        # TODO: Assign to an instance attribute called shortcut an empty object of type Sequential.
        # If the provided stride is not equal to 1 or
        # the provided number of input channels does not equal the provided number of output channels,
        # reassign shortcut an object of type Sequential constructed with a 1x1 2D convolution and 2D batch normalization.
        # The convolution must have a number of input channels equal to the provided number of input channels,
        # a number of output channels equal to the provided number of output channels, stride equal to the provided stride,
        # and no bias.
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )


    def forward(self, x):
        # TODO: Assign to a local variable called intermediate
        # the output of passing the input through the first convolution.
        intermediate = self.convolution_1(x)

        # TODO: Assign to intermediate the output of passing intermediate through the first batch normalization.
        intermediate = self.batch_normalization_1(intermediate)

        # TODO: Assign to intermediate the output of passing intermediate through ReLU.
        intermediate = self.relu(intermediate)

        # TODO: Assign to intermediate the output of passing intermediate through the second convolution.
        intermediate = self.convolution_2(intermediate)

        # TODO: Assign to intermediate the output of passing intermediate through the second batch normalization.
        intermediate = self.batch_normalization_2(intermediate)

        # TODO: Assign to intermediate the output of adding intermediate and the output of passing the provided input
        # through the shortcut.
        intermediate = intermediate + self.shortcut(x)

        # TODO: Return the output of passing intermediate through ReLU.
        intermediate = self.relu(intermediate)
        return intermediate


class ResNet(nn.Module):
    def __init__(self, num_classes=18):
        super(ResNet, self).__init__()
        
        # TODO: Assign to an instance attribute called `convolution` an object of type Sequential constructed with
        # 7x7 2D convolution, 2D batch normalization, and ReLU.
        # The convolution must have 3 input channels, 64 output channels, stride of 2, padding of 3, and no bias.
        # ReLU must be performed in place.
        self.convolution = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # TODO: Assign to an instance attribute called `max_pooling` 2D max pooling with
        # a kernel size of 3, stride of 2, and padding of 1.
        self.max_pooling = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # TODO: Assign to a instance attribute called `residual_layer_1` the output of method `_make_layer`
        # with 64 input channels, 64 output channels, 2 blocks, and stride of 1.
        self.residual_layer_1 = self._make_layer(64, 64, num_blocks=2, stride=1)

        # TODO: Assign to a instance attribute called `residual_layer_2` the output of method `_make_layer`
        # with 64 input channels, 128 output channels, 2 blocks, and stride of 2.
        self.residual_layer_2 = self._make_layer(64, 128, num_blocks=2, stride=2)

        # TODO: Assign to a instance attribute called `residual_layer_3` the output of method `_make_layer`
        # with 128 input channels, 256 output channels, 2 blocks, and stride of 2.
        self.residual_layer_3 = self._make_layer(128, 256, num_blocks=2, stride=2)

        # TODO: Assign to a instance attribute called `residual_layer_4` the output of method `_make_layer`
        # with 256 input channels, 512 output channels, 2 blocks, and stride of 2.
        self.residual_layer_4 = self._make_layer(256, 512, num_blocks=2, stride=2)

        # TODO: Assign to an instance attribute called `average_pooling` 2D average pooling with height of 1 and width of 1.
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # TODO: Assign to an instance attribute called `linear_transformation` a linear transformation
        # with 512 input features and a number of output features equal to the provided number of classes.
        self.linear_transformation = nn.Linear(512, num_classes)


    def _make_layer(self, in_channels, out_channels, num_blocks, stride):

        # TODO: Create an empty list called `list_of_blocks`.
        list_of_blocks = []

        # TODO: Add to the list of blocks a basic block with
        # a number of input channels equal to the provided number of input channels,
        # a number of output channels equal to the provided number of output channels,
        # and a stride equal to the provided stride.
        list_of_blocks.append(BasicBlock(in_channels, out_channels, stride))

        # TODO: For each remaining block, add to the list of blocks a basic block with
        # a number of input channels equal to the provided number of output channels and
        # a number of output channels equal to the provided number of output channels.
        for _ in range(num_blocks - 1):
            list_of_blocks.append(BasicBlock(out_channels, out_channels, stride=1))

        # TODO: Return an object of type Sequential constructed with the blocks in the list of blocks.
        return nn.Sequential(*list_of_blocks)


    def forward(self, x):        
        # TODO: Assign to a local variable called intermediate the output of passing the provided input through
        # the convolution.
        intermediate = self.convolution(x)

        # TODO: Assign to intermediate the output of passing intermediate through max pooling.
        intermediate = self.max_pooling(intermediate)

        # TODO: Assign to intermediate the output of passing intermediate through the first residual layer.
        intermediate = self.residual_layer_1(intermediate)

        # TODO: Assign to intermediate the output of passing intermediate through the second residual layer.
        intermediate = self.residual_layer_2(intermediate)

        # TODO: Assign to intermediate the output of passing intermediate through the third residual layer.
        intermediate = self.residual_layer_3(intermediate)

        # TODO: Assign to intermediate the output of passing intermediate through the fourth residual layer.
        intermediate = self.residual_layer_4(intermediate)

        # TODO: Assign to intermediate the output of passing intermediate through average pooling.
        intermediate = self.average_pooling(intermediate)

        # TODO: Flatten intermediate from start dimension 1 on.
        # The output tensor of flattening has shape (number of images in batch, 512).
        intermediate = intermediate.view(intermediate.size(0), -1)

        # TODO: Return the output of passing intermediate through the linear transformation.
        # The output tensor of the linear transformation has shape (number of images in batch, 18).
        return self.linear_transformation(intermediate)