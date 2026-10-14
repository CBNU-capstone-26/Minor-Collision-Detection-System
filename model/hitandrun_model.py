import os

import torch.nn as nn
from pytorchvideo.models.hub import x3d_m
from torch.utils.checkpoint import checkpoint_sequential


# X3D-M의 마지막 res 스테이지 출력 채널 수(구조 확인으로 검증한 값).
X3D_M_FEATURE_CHANNELS = 192
# 헤드 확장부(post_conv)를 거친 뒤의 채널 수 — 분류/CAM은 이 차원에서 이뤄진다.
X3D_HEAD_CHANNELS = 2048


# 분류 헤드 앞 dropout 비율 — 세 백본 공통값.
# 백본 비교 실험에서 정규화 강도가 모델마다 다르면 결과 해석이 불가능해지므로
# 반드시 같은 값을 쓴다. (기존: s3d 0.2 / x3d 0.5 / slowfast 0.2 로 제각각이었음)
DROPOUT_P = 0.5

# gradient checkpointing — 활성값을 들고 있지 않고 backward 때 forward를 다시
# 계산한다. gradient가 수학적으로 동일해 **정확도에 영향이 없고**, 메모리만 줄고
# 속도가 약 30% 느려진다.
#
# X3D는 파라미터가 S3D의 1/2.7(2.98M vs 7.91M)인데 활성값은 1.7배다(25.4GB vs
# 15.0GB, batch 15 기준). 시간축을 30프레임 그대로 유지하고 병목 안에서 채널을
# 2.25배 확장하기 때문이다. RTX 3080 10GB에서 batch 15가 OOM 나는 원인이라,
# batch를 줄이는 대신(→ 배치에 A가 없어져 학습이 깨진다) 이걸 쓴다.
# 학습 때만 작동하며(self.training), 추론·CAM 경로는 건드리지 않는다.
GRAD_CHECKPOINT = os.getenv("GRAD_CHECKPOINT", "1").strip().lower() not in ("0", "false", "no")
# 백본 블록이 5개뿐이라 segments=5 면 세그먼트당 1블록 = 블록 하나를 통째로
# 재계산해야 한다. 그 피크가 features[1] 에서 8.67GB(batch 15)로 10GB에 빠듯하다.
# 블록 수보다 크게 잡으면 checkpoint_sequential 이 블록 내부까지 쪼개 피크가 낮아진다.
GRAD_CHECKPOINT_SEGMENTS = int(os.getenv("GRAD_CHECKPOINT_SEGMENTS", "5"))

class HitAndRun3DCNN(nn.Module):
    """X3D-M 백본 기반 물피도주 감지 모델.

    X3D(Feichtenhofer, CVPR 2020)는 2D 이미지 분류망을 폭·깊이·해상도·프레임
    축으로 점진 확장해 얻은 효율적 3D-CNN이다. 같은 정확도를 S3D보다 훨씬 적은
    연산으로 내는 것이 특징이라, 멀티-아워 영상을 슬라이딩 윈도로 훑는 이
    서비스에서는 '윈도우당 비용'을 줄이는 쪽에 이점이 있다.
    (파라미터: X3D-M 3.8M vs S3D 7.9M)

    ⚠️ pytorchvideo의 기본 헤드(ResNetBasicHead)는 AvgPool3d 커널이 (16,7,7)로
       고정되어 있어 T=30 입력에서 깨진다. 하지만 헤드를 통째로 버리면 그 안의
       확장 conv(192→432→2048)까지 같이 날아가, 원본 구조로 학습된 가중치를
       이식할 수 없다. 그래서 **고정 커널 pool만** 교체한다:

           AvgPool3d((16,7,7))  →  AdaptiveAvgPool3d((1, None, None))

       시간축만 뭉개고 공간 7×7을 남기므로 (a) T가 몇이든 동작하고
       (b) 2048채널 공간 특징맵이 살아 있어 CAM을 뽑을 수 있다.
       CAM은 어차피 시간축을 평균내므로(predict_cam.py의 mean(cam, dim=0))
       시간축 보존은 필요 없고, post_conv를 T=1에서만 돌려 연산도 아낀다.

    서비스 통합 인터페이스는 S3D 버전과 동일하게 유지한다:
      - 입력 (B, 3, T, 224, 224) → 출력 (B, num_classes) logits
      - `head_conv`   : 1×1×1 Conv3d 분류 헤드 (CAM 가중치로 사용)
      - `inception5b` : CAM forward hook 대상 (2048채널 공간 특징맵)

    Args:
        num_classes: 분류 클래스 수 (기본 2: S/A)
        pretrained : True면 Kinetics-400 사전학습 가중치로 백본 초기화.
                     추론에서는 학습된 state_dict를 로드하므로 False로 생성해
                     불필요한 다운로드를 피한다.
    """

    def __init__(self, num_classes=2, pretrained=False, dropout_p=DROPOUT_P):
        super(HitAndRun3DCNN, self).__init__()
        base = x3d_m(pretrained=pretrained)

        # blocks[-1]은 헤드. stem+res 스테이지와 분리한다.
        self.features = nn.Sequential(*list(base.blocks)[:-1])

        # 헤드의 확장부(pre_conv 192→432 → BN → ReLU → pool → post_conv 432→2048
        # → ReLU)는 그대로 살리고, 고정 커널 pool만 교체한다.
        head_pool = base.blocks[-1].pool
        head_pool.pool = nn.AdaptiveAvgPool3d((1, None, None))
        self.head_pool = head_pool

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        # 분류 헤드는 logit 출력이므로 BN/ReLU 없이 Conv3d 단독 (CAM 가중치 겸용).
        # 원본 헤드의 Linear(2048→C)와 같은 연산이라 가중치를 reshape해 이식할 수 있다.
        self.head_conv = nn.Conv3d(
            X3D_HEAD_CHANNELS, num_classes, kernel_size=1)

    @property
    def inception5b(self):
        """CAM hook 호환 별칭 — 헤드 확장부 출력(2048채널, 공간 7×7 유지).

        property라 모듈이 중복 등록되지 않으며, predict_cam.py 의
        `model.inception5b.register_forward_hook(...)`이 수정 없이 동작한다.
        head_conv.weight(2048채널)와 채널 수가 맞아야 CAM 가중합이 성립한다.
        """
        return self.head_pool

    def forward(self, x):
        if self.training and GRAD_CHECKPOINT:
            # 백본만 체크포인팅. head_pool 은 CAM hook 대상이라 밖에 둔다.
            x = checkpoint_sequential(
                self.features, GRAD_CHECKPOINT_SEGMENTS, x, use_reentrant=False)
        else:
            x = self.features(x)    # (B, 192, T, 7, 7)
        x = self.head_pool(x)       # (B, 2048, 1, 7, 7)
        x = self.avg_pool(x)        # (B, 2048, 1, 1, 1)
        x = self.dropout(x)
        x = self.head_conv(x)
        return x.squeeze(-1).squeeze(-1).squeeze(-1)
