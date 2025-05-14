import torch.nn as nn
import torch
from torchvision.models.resnet import Bottleneck


# Here we define an MLP model parametrized with a list of layer sizes.
class MLP(nn.Module):
    def __init__(self, layer_sizes, bn):
        super().__init__()
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:  # No BatchNorm and ReLU on the last layer
                if bn:
                    layers.append(nn.BatchNorm1d(layer_sizes[i + 1]))
                layers.append(nn.ReLU())
        self.model = nn.Sequential(nn.Flatten(), *layers)

    def forward(self, x):
        return self.model(x)


class ResidualMLPBlock(nn.Module):
    def __init__(self, width, bn):
        super().__init__()
        self.fc1 = nn.Linear(width, width)
        self.fc2 = nn.Linear(width, width)
        if bn:
            self.model = nn.Sequential(
                self.fc1,
                nn.BatchNorm1d(width),
                nn.ReLU(),
                self.fc2,
                nn.BatchNorm1d(width),
                nn.ReLU(),
            )
        else:
            self.model = nn.Sequential(self.fc1, nn.ReLU(), self.fc2, nn.ReLU())

    def forward(self, x):
        return self.model(x) + x


class ResidualMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_blocks, num_classes, bn):
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        blocks = []
        for i in range(num_blocks):
            blocks.append(ResidualMLPBlock(hidden_dim, bn))
        self.output_layer = nn.Linear(hidden_dim, num_classes)
        self.model = nn.Sequential(
            nn.Flatten(),
            self.input_layer,
            nn.ReLU(),
            *blocks,
            self.output_layer,
        )

    def forward(self, x):
        return self.model(x)


class CustomCNN(nn.Module):
    def __init__(
        self,
        block_type="basic",
        layers=[2, 2, 2, 2],
        num_classes=1000,
        use_skip=True,
        zero_init_residual=False,
    ):
        super(CustomCNN, self).__init__()

        self.use_skip = use_skip
        self.inplanes = 64

        # Initial convolution layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Select block type
        if block_type == "basic":
            block = BasicBlock
        elif block_type == "bottleneck":
            block = Bottleneck
        else:
            raise ValueError("block_type must be 'basic' or 'bottleneck'")

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        # Create downsample layer if needed (for skip connection when dimensions change)
        if stride != 1 or self.inplanes != planes * block.expansion:
            if self.use_skip:
                downsample = nn.Sequential(
                    nn.Conv2d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(planes * block.expansion),
                )

        layers = []
        # First block in the layer might need downsample
        layers.append(
            block(self.inplanes, planes, stride, downsample, use_skip=self.use_skip)
        )
        self.inplanes = planes * block.expansion

        # Subsequent blocks
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, use_skip=self.use_skip))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, use_skip=True):
        super(BasicBlock, self).__init__()
        self.use_skip = use_skip
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.use_skip and self.downsample is not None:
            identity = self.downsample(x)

        if self.use_skip:
            out += identity
        out = self.relu(out)

        return out
