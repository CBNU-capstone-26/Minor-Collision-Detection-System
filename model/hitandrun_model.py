import torch
import torch.nn as nn
from torchvision.models.video import S3D_Weights, s3d


class HitAndRun3DCNN(nn.Module):
    """S3D backbone based binary classifier for minor-collision detection."""

    def __init__(self, num_classes=2, pretrained=False):
        super().__init__()
        weights = S3D_Weights.KINETICS400_V1 if pretrained else None
        base = s3d(weights=weights)
        self.features = base.features
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.2)
        self.head_conv = nn.Conv3d(1024, num_classes, kernel_size=1)

    @property
    def inception5b(self):
        """Expose the final S3D block for the existing CAM hook code."""
        return self.features[-1]

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.head_conv(x)
        return x.squeeze(-1).squeeze(-1).squeeze(-1)
