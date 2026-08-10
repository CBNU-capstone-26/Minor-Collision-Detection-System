import torch
import torch.nn as nn


def conv_block(in_channels, out_channels, **kwargs):
    """Conv3d -> BatchNorm3d -> ReLU block."""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, bias=False, **kwargs),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True),
    )


class InceptionModule3D(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super(InceptionModule3D, self).__init__()
        self.branch1 = conv_block(in_channels, n1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            conv_block(in_channels, n3x3_reduce, kernel_size=1),
            conv_block(n3x3_reduce, n3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            conv_block(in_channels, n5x5_reduce, kernel_size=1),
            conv_block(n5x5_reduce, n5x5, kernel_size=3, padding=1),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            conv_block(in_channels, pool_proj, kernel_size=1),
        )

    def forward(self, x):
        out1 = self.branch1(x)
        out2 = self.branch2(x)
        out3 = self.branch3(x)
        out4 = self.branch4(x)
        return torch.cat([out1, out2, out3, out4], dim=1)


class HitAndRun3DCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(HitAndRun3DCNN, self).__init__()

        self.conv1 = conv_block(
            3, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3)
        )
        self.maxpool1 = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        self.conv2 = nn.Sequential(
            conv_block(64, 64, kernel_size=1),
            conv_block(64, 192, kernel_size=3, padding=1),
        )
        self.maxpool2 = nn.MaxPool3d(
            kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1)
        )

        self.inception3a = InceptionModule3D(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionModule3D(256, 128, 128, 192, 32, 96, 64)
        self.maxpool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule3D(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionModule3D(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionModule3D(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionModule3D(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionModule3D(528, 256, 160, 320, 32, 128, 128)
        self.maxpool4 = nn.MaxPool3d(
            kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1
        )

        self.inception5a = InceptionModule3D(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule3D(832, 384, 192, 384, 48, 128, 128)

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.head_conv = nn.Conv3d(1024, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.head_conv(x)
        return x.squeeze(-1).squeeze(-1).squeeze(-1)
