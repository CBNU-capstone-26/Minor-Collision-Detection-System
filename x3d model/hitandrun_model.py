import torch.nn as nn
from pytorchvideo.models.hub import x3d_m


# X3D-M의 마지막 res 스테이지 출력 채널 수(구조 확인으로 검증한 값).
X3D_M_FEATURE_CHANNELS = 192


class HitAndRun3DCNN(nn.Module):
    """X3D-M 백본 기반 물피도주 감지 모델.

    X3D(Feichtenhofer, CVPR 2020)는 2D 이미지 분류망을 폭·깊이·해상도·프레임
    축으로 점진 확장해 얻은 효율적 3D-CNN이다. 같은 정확도를 S3D보다 훨씬 적은
    연산으로 내는 것이 특징이라, 멀티-아워 영상을 슬라이딩 윈도로 훑는 이
    서비스에서는 '윈도우당 비용'을 줄이는 쪽에 이점이 있다.
    (파라미터: X3D-M 3.8M vs S3D 7.9M)

    ⚠️ pytorchvideo의 기본 헤드(ResNetBasicHead)는 AvgPool3d 커널이 (16,7,7)로
       고정되어 있어 T=30 입력에서 깨진다. 그래서 헤드를 버리고 res 스테이지만
       특징추출부로 쓰고, S3D 래퍼와 동일한 형태의 헤드를 새로 붙인다.

    서비스 통합 인터페이스는 S3D 버전과 동일하게 유지한다:
      - 입력 (B, 3, T, 224, 224) → 출력 (B, num_classes) logits
      - `head_conv`   : 1×1×1 Conv3d 분류 헤드 (CAM 가중치로 사용)
      - `inception5b` : CAM forward hook 대상 (마지막 res 스테이지 별칭)

    Args:
        num_classes: 분류 클래스 수 (기본 2: S/A)
        pretrained : True면 Kinetics-400 사전학습 가중치로 백본 초기화.
                     추론에서는 학습된 state_dict를 로드하므로 False로 생성해
                     불필요한 다운로드를 피한다.
    """

    def __init__(self, num_classes=2, pretrained=False):
        super(HitAndRun3DCNN, self).__init__()
        base = x3d_m(pretrained=pretrained)

        # blocks[-1]은 고정 커널 헤드라 제외하고 stem+res 스테이지만 사용
        self.features = nn.Sequential(*list(base.blocks)[:-1])

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.2)
        # 분류 헤드는 logit 출력이므로 BN/ReLU 없이 Conv3d 단독 (CAM 가중치 겸용)
        self.head_conv = nn.Conv3d(
            X3D_M_FEATURE_CHANNELS, num_classes, kernel_size=1)

    @property
    def inception5b(self):
        """CAM hook 호환 별칭 — 마지막 res 스테이지(192채널 출력).

        property라 모듈이 중복 등록되지 않으며, predict_cam.py 의
        `model.inception5b.register_forward_hook(...)`이 수정 없이 동작한다.
        """
        return self.features[-1]

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.head_conv(x)
        return x.squeeze(-1).squeeze(-1).squeeze(-1)
