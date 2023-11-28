import torch
import torch.nn as nn
import torchvision.transforms.functional as TF


# Creating Double Convolution module
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()

        # Main part of DoubleConv
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# Creating UNET model
class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512]
    ):
        super(UNET, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Down scale part of UNET
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # Up part of UNET
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # Bottom part of UNET
        self.bottom = DoubleConv(features[-1], features[-1]*2)

        # Output conv
        self.out_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    # UNET architecture
    def forward(self, x):
        # list where we store our skip connection
        skip_connections = []

        # downscaling the image
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # putting img through bottom of UNET
        x = self.bottom(x)
        skip_connections = skip_connections[::-1]

        # upscaling the image
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]

            if x.shape != skip_connection.shape:
                x = TF.resize(x, size=skip_connection.shape[2:])

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)

        return self.out_conv(x)
