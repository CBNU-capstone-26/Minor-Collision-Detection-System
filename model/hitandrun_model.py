import torch
import torch.nn as nn
from torchvision.models.video import s3d, S3D_Weights


class HitAndRun3DCNN(nn.Module):
    """S3D 백본 기반 물피도주 감지 모델.

    기존 커스텀 I3D GoogLeNet(12.3M)을 torchvision S3D(≈8M)로 교체했다.
    S3D는 I3D의 3D conv를 '공간(1×k×k) + 시간(k×1×1)' 분리 컨볼루션으로
    인수분해한 구조로, I3D 대비 가볍고 빠르면서 정확도는 동등 이상이다
    (Xie et al., ECCV 2018). Kinetics-400 사전학습 가중치로 초기화하면 
    모션 필터를 이미 갖춘 상태에서 미세조정을 시작하므로 소규모 데이터에서
    일반화가 크게 개선된다.

    서비스 통합 인터페이스는 기존과 동일하게 유지한다:
      - 입력 (B, 3, T, 224, 224) → 출력 (B, num_classes) logits
      - `head_conv`   : 1×1×1 Conv3d 분류 헤드 (CAM 가중치로 사용)
      - `inception5b` : CAM forward hook 대상 (S3D의 마지막 인셉션 블록 별칭)

    Args:
        num_classes: 분류 클래스 수 (기본 2: S/A)
        pretrained : True면 Kinetics-400 사전학습 가중치로 백본 초기화.
                     추론(서비스 워커)에서는 학습된 state_dict를 로드하므로
                     False(기본값)로 생성해 불필요한 다운로드를 피한다.
    """

    def __init__(self, num_classes=2, pretrained=False):
        super(HitAndRun3DCNN, self).__init__()
        weights = S3D_Weights.KINETICS400_V1 if pretrained else None
        base = s3d(weights=weights)

        # S3D 특징 추출부 (마지막 블록 출력: 1024채널)
        self.features = base.features

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.2) #dropout 값 0.4 -> 0.2로 변경 (이정주)
        # 분류 헤드는 logit을 출력하므로 BN/ReLU 없이 Conv3d 단독 (CAM 가중치로도 사용)
        self.head_conv = nn.Conv3d(1024, num_classes, kernel_size=1)

    @property
    def inception5b(self):
        """CAM hook 호환 별칭 — S3D의 마지막 인셉션 블록(1024채널 출력).

        property라 모듈이 중복 등록되지 않으며(state_dict 키는 features.* 유지),
        predict_cam.py 의 `model.inception5b.register_forward_hook(...)`이
        코드 수정 없이 그대로 동작한다.
        """
        return self.features[-1]

    def forward(self, x):
        x = self.features(x)
        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.head_conv(x)
        return x.squeeze(-1).squeeze(-1).squeeze(-1)
