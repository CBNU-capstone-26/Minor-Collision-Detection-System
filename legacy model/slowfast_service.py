"""SlowFast R50 adapter for the service's tensor input and linear CAM API."""
from types import SimpleNamespace

import torch
from torch import nn
from pytorchvideo.models.slowfast import create_slowfast


class SlowFastService(nn.Module):
    def __init__(self, state_dict, num_classes=2):
        super().__init__()
        self.backbone = create_slowfast(model_num_class=num_classes)
        self.backbone.load_state_dict(state_dict, strict=True)
        # Expose temporally averaged features to the existing CAM hook.
        self.inception5b = nn.Identity()

    @property
    def head_conv(self):
        return SimpleNamespace(
            weight=self.backbone.blocks[6].proj.weight[:, :, None, None, None]
        )

    def forward(self, x):
        # Standard R50 pathways: 32 fast frames and 8 slow frames (alpha=4).
        fast_indices = torch.linspace(0, x.shape[2] - 1, 32, device=x.device).long()
        fast = x.index_select(2, fast_indices)
        slow_indices = torch.linspace(0, 31, 8, device=x.device).long()
        pathways = [fast.index_select(2, slow_indices), fast]
        for block in self.backbone.blocks[:5]:
            pathways = block(pathways)
        self.inception5b(torch.cat(
            [feature.mean(dim=2, keepdim=True) for feature in pathways], dim=1
        ))
        return self.backbone.blocks[6](self.backbone.blocks[5](pathways))
