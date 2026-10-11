import torch.nn as nn
import torch


class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()

        # TODO: Assign to an instance attribute called `depthwise_convolution` an object of type Sequential
        # constructed with 3x3 2D convolution, batch normalization, and ReLU6.
        # The convolution should have a number of input channels equal to the provided number of input channels,
        # a number of output channels equal to the provided number of input channels,
        # stride equal to the provided stride, a number of groups equal to the provided number of input channels,
        # padding of 1, and no bias.
        # Perform ReLU6 in place.
        self.depthwise_convolution = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU6(inplace=True)
        )
        
        # TODO: Assign to an instance attribute called `pointwise_convolution` an object of type Sequential
        # constructed with 1x1 2D convolution, batch normalization, and ReLU6.
        # The convolution should have a number of input channels equal to the provided number of input channels,
        # a number of output channels equal to the provided number of output channels, stride of 1, and no bias.
        # Perform ReLU6 in place.
        self.pointwise_convolution = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU6(inplace=True)
        )


    def forward(self, x):

        # TODO: Assign to a local variable called intermediate the output of passing the input through depthwise convolution.
        intermediate = self.depthwise_convolution(x)

        # TODO: Return the output of passing intermediate through pointwise convolution.
        return self.pointwise_convolution(intermediate)


class InvertedResidual(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, expand_ratio=6):
        super(InvertedResidual, self).__init__()

        # TODO: Assign to an instance attribute called `residual_will_be_used` an indicator
        # that stride is 1 and the number of input channels equals the number of output channels.
        self.residual_will_be_used = stride == 1 and in_channels == out_channels

        # TODO: Assign to a local variable called `hidden_dim`
        # the product of the provided number of input channels and the provided expansion ratio.
        hidden_dim = in_channels * expand_ratio

        # TODO: Create an empty list called `list_of_layers`.
        list_of_layers = []

        # TODO: If the provided expansion ratio is not equal to 1,
        #     add to the list of layers 1x1 2D convolution, 2D batch normalization, and ReLU6.
        #     The convolution must have a number of input channels equal to the provided number of input channels,
        #     a number of output channels equal to the hidden dimension, and no bias.
        #     Add to the list of layers batch normalization.
        #     Perform ReLU6 in place.
        if expand_ratio != 1:
            list_of_layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, stride=1, bias=False))
            list_of_layers.append(nn.BatchNorm2d(hidden_dim))
            list_of_layers.append(nn.ReLU6(inplace=True))

        # TODO: Add to the list of layers 3x3 2D convolution, 2D batch normalization, and ReLU6.
        # The convolution should have a number of input channels equal to the hidden dimension,
        # a number of output channels equal to the hidden dimension, stride equal to the provided stride,
        # a number of groups equal to the hidden dimension, padding of 1, and no bias.
        # Perform ReLU6 in place.
        list_of_layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False))
        list_of_layers.append(nn.BatchNorm2d(hidden_dim))
        list_of_layers.append(nn.ReLU6(inplace=True))

        # TODO: Add to the list of layers 1x1 2D convolution and 2D batch normalization.
        # The convolution must have a number of input channels equal to the hidden dimension,
        # a number of output channels equal to the provided number of output channels, and no bias.
        list_of_layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, bias=False))
        list_of_layers.append(nn.BatchNorm2d(out_channels))

        # TODO: Add layers to an object of type Sequential.
        # Assign that object to an instance attribute called sequential.
        self.sequential = nn.Sequential(*list_of_layers)
        


    def forward(self, x):

        # TODO: If residual will be used, return the output of adding the provided input and
        # the output of passing the provided input through sequential.
        # Otherwise, return the output of passing the provided input through sequential.
        if self.residual_will_be_used:
            return x + self.sequential(x)
        else:
            return self.sequential(x)


class MobileNet(nn.Module):
    def __init__(self, num_classes=18, width_mult=1.0, dropout_prob=0.2):
        super(MobileNet, self).__init__()

        # TODO: Define a local variable called `number_of_output_channels_in_initial_convolution`
        # equal to the product of 32 and the provided multiplier.
        # Cast the product to an integer.
        number_of_output_channels_in_initial_convolution = int(32 * width_mult)

        # TODO: Assign to an instance attribute called `initial_convolution` an object of type Sequential
        # constructed with a 3x3 2D convolution, 2D batch normalization, and ReLU6.
        # The convolution must have a number of input channels of 3,
        # a number of output channels equal to the defined number of output channels in the initial convolution,
        # stride of 2, padding of 1, and no bias.
        # ReLU6 should be performed in place.
        self.initial_convolution = nn.Sequential(
            nn.Conv2d(3, number_of_output_channels_in_initial_convolution, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(number_of_output_channels_in_initial_convolution),
            nn.ReLU6(inplace=True)
        )

        # TODO: Create an empty list of layers called `list_of_layers`.
        list_of_layers = []

        configuration = [
            (32, 16, 1, 1),
            (16, 24, 2, 6),
            (24, 24, 1, 6),
            (24, 32, 2, 6),
            (32, 32, 1, 6),
            (32, 32, 1, 6),
            (32, 64, 2, 6),
            (64, 64, 1, 6),
            (64, 64, 1, 6),
            (64, 64, 1, 6),
            (64, 96, 1, 6),
            (96, 96, 1, 6),
            (96, 96, 1, 6),
            (96, 160, 2, 6),
            (160, 160, 1, 6),
            (160, 160, 1, 6),
            (160, 320, 1, 6)
        ]

        # TODO: For number of input channels, number of output channels, stride, and expansion ratio in configuration,
        #     define a local variable called `scaled_number_of_input_channels` that is
        #     the product of the number of input channels and the provided multiplier.
        #     Cast the product to an integer.
        #     Define a local variable called `scaled_number_of_output_channels` that is
        #     the product of the number of output channels and the provided multiplier.
        #     Cast the product to an integer.
        #     Add to the list of layers an object of type `InvertedResidual` with
        #     a number of input channels equal to the scaled number of input channels,
        #     a number of output channels equal to the scaled number of output channels,
        #     the appropriate stride, and the appropriate expansion ratio.

        for in_channels, out_channels, stride, expand_ratio in configuration:
            scaled_number_of_input_channels = int(in_channels * width_mult)
            scaled_number_of_output_channels = int(out_channels * width_mult)
            list_of_layers.append(InvertedResidual(scaled_number_of_input_channels, scaled_number_of_output_channels, stride=stride, expand_ratio=expand_ratio))

        # TODO: Add layers in the list of layers to an object of type Sequential.
        # Assign that object to an instance attribute called sequential.
        self.sequential = nn.Sequential(*list_of_layers)

        # TODO: Define a local variable called `number_of_output_channels_in_final_convolution`
        # equal to the product of 1280 and the provided multiplier if the multiplier is greater than 1.0 and 1280 otherwise.
        # Cast the product to an integer.
        number_of_output_channels_in_final_convolution = int(1280 * width_mult) if width_mult > 1.0 else 1280

        # TODO: Assign to an instance attribute called `final_convolution` an object of type Sequential
        # constructed with a 1x1 2D convolution, 2D batch normalization, and ReLU6.
        # The convolution must have a number of input channels equal to the product of 320 and the provided multiplier,
        # a number of output channels equal to the defined number of output channels in the final convolution, and no bias.
        # Cast the product to an integer.
        # ReLU6 should be performed in place.
        self.final_convolution = nn.Sequential(
            nn.Conv2d(int(320 * width_mult), number_of_output_channels_in_final_convolution, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(number_of_output_channels_in_final_convolution),
            nn.ReLU6(inplace=True)
        )

        # TODO: Assign to an instance attribute called `average_pooling` 2D average pooling with height of 1 and width of 1.
        self.average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        # TODO: Assign to an instance attribute called `dropout` dropout with the provided probability.
        self.dropout = nn.Dropout(p=dropout_prob)

        # TODO: Assign to an instance attribute called `linear_transformation` a linear transformation
        # with a number of input features equal to the number of output channels in the final convolution and
        # a number of output features equal to the provided number of classes.
        self.linear_transformation = nn.Linear(number_of_output_channels_in_final_convolution, num_classes)

        # TODO: Call method `initialize_weights`.
        self.initialize_weights()


    def initialize_weights(self):

        # TODO: For each module of this neural network,
        #     if the module is a 2D convolution,
        #         fill the module weight with values using a Kaiming normal distribution.
        #         if the module bias exists,
        #             fill the module bias with 0.
        #     otherwise, if the module is 2D batch normalization,
        #         fill the module weight with 1.
        #         Fill the module bias with 0.
        #     otherwise, if the module is a linear transformation,
        #         fill the module weight with values drawn from the standard normal distribution.
        #         Fill the module bias with 0.
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.constant_(module.bias, 0)


    def forward(self, x):

        # TODO: Assign to a local variable called intermediate the output of
        # passing the provided input through the initial convolution.
        intermediate = self.initial_convolution(x)

        # TODO: Assign to intermediate the output of passing intermediate through the sequential.
        intermediate = self.sequential(intermediate)

        # TODO: Assign to intermediate the output of passing intermediate through the final convolution.
        intermediate = self.final_convolution(intermediate)

        # TODO: Assign to intermediate the output of passing intermediate through average pooling.
        intermediate = self.average_pooling(intermediate)

        # TODO: Flatten intermediate from start dimension 1 on.
        intermediate = torch.flatten(intermediate, 1)

        # TODO: Assign to intermediate the output of passing intermediate through dropout.
        intermediate = self.dropout(intermediate)

        # TODO: Return the output of passing intermediate through the linear transformation.
        return self.linear_transformation(intermediate)