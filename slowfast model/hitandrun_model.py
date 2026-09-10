import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorchvideo.models.hub import slowfast_r50


# Fast:Slow 프레임 비율. 원 논문/pytorchvideo 기본값과 동일(=4).
SLOWFAST_ALPHA = 4
# blocks[4] 출력 채널: slow 2048 + fast 256 (구조 확인으로 검증한 값)
SLOWFAST_FUSED_CHANNELS = 2048 + 256


class _SlowFastFuse(nn.Module):
    """[slow, fast] 두 경로를 하나의 특징맵으로 합친다.

    시간 길이가 서로 다르므로(slow T/α, fast T) 시간축만 1로 풀링해 맞춘 뒤
    채널 방향으로 concat한다. 공간 해상도(H×W)는 그대로 두어야 CAM을 그릴 수
    있으므로 건드리지 않는다. 출력: (B, 2304, 1, H, W)

    이 모듈이 CAM forward hook 대상(`inception5b`)이다.
    """

    def forward(self, xs):
        slow, fast = xs
        slow = F.adaptive_avg_pool3d(slow, (1, slow.shape[-2], slow.shape[-1]))
        fast = F.adaptive_avg_pool3d(fast, (1, fast.shape[-2], fast.shape[-1]))
        return torch.cat([slow, fast], dim=1)


class HitAndRun3DCNN(nn.Module):
    """SlowFast-R50 백본 기반 물피도주 감지 모델.

    SlowFast(Feichtenhofer et al., ICCV 2019)는 두 경로를 쓴다.
      - Slow: 프레임을 듬성듬성(1/α) 보되 채널이 두꺼움 → 공간·의미 정보
      - Fast: 모든 프레임을 보되 채널이 얇음 → 빠른 모션 정보
    충돌 순간의 '짧은 흔들림'은 Fast 경로가 담당하는 영역이라, 이 태스크와
    궁합이 좋을 가능성이 있다(다만 검증 전까지는 가설이다).
    (파라미터: SlowFast-R50 34.6M vs S3D 7.9M — 훨씬 무겁다)

    ⚠️ pytorchvideo 원본의 blocks[5](PoolConcatPathway)는 AvgPool 커널이
       fast T=32 기준으로 고정되어 있어 T=30 입력에서 깨진다. 그래서 blocks[0..4]
       까지만 쓰고, 경로 융합과 헤드는 직접 붙인다.

    서비스 통합 인터페이스는 S3D 버전과 동일하게 유지한다:
      - 입력 (B, 3, T, 224, 224) → 출력 (B, num_classes) logits
        (내부에서 slow/fast 두 경로로 나눠 넣으므로 호출부는 바뀌지 않는다)
      - `head_conv`   : 1×1×1 Conv3d 분류 헤드 (CAM 가중치로 사용)
      - `inception5b` : CAM forward hook 대상 (두 경로 융합 모듈 별칭)

    Args:
        num_classes: 분류 클래스 수 (기본 2: S/A)
        pretrained : True면 Kinetics-400 사전학습 가중치로 백본 초기화.
    """

    def __init__(self, num_classes=2, pretrained=False):
        super(HitAndRun3DCNN, self).__init__()
        base = slowfast_r50(pretrained=pretrained)

        # blocks[5](PoolConcatPathway)·blocks[6](헤드)는 고정 커널이라 제외
        self.features = nn.ModuleList(list(base.blocks)[:5])
        self.fuse = _SlowFastFuse()

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.2)
        # 분류 헤드는 logit 출력이므로 BN/ReLU 없이 Conv3d 단독 (CAM 가중치 겸용)
        self.head_conv = nn.Conv3d(
            SLOWFAST_FUSED_CHANNELS, num_classes, kernel_size=1)

    @property
    def inception5b(self):
        """CAM hook 호환 별칭 — 두 경로를 합친 특징맵(2304채널)을 내보내는 모듈."""
        return self.fuse

    def forward(self, x):
        # 단일 입력 (B,3,T,H,W)을 SlowFast가 요구하는 두 경로로 나눈다.
        #   slow = α프레임마다 1장, fast = 전체 프레임
        slow = x[:, :, ::SLOWFAST_ALPHA]
        h = [slow, x]
        for blk in self.features:
            h = blk(h)
        h = self.fuse(h)
        h = self.avg_pool(h)
        h = self.dropout(h)
        h = self.head_conv(h)
        return h.squeeze(-1).squeeze(-1).squeeze(-1)
