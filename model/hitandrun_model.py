import os

import torch
import torch.nn as nn
from pytorchvideo.models.hub import x3d_m
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


def _tag_names(path):
    """조건 표시를 찾을 이름들을 돌려줍니다. 파일 이름과, 학습 회차 폴더(<시각>_<모델>_<조건>) 안의 파일이면 그 폴더 이름입니다.
    work_best.pth나 epochs/ 안 파일처럼 이름에 표시가 없어도 회차 폴더 이름으로 맞춥니다.
    """
    import re
    p = os.path.abspath(str(path))
    names = [os.path.basename(p)]
    for d in (os.path.dirname(p), os.path.dirname(os.path.dirname(p))):
        if re.match(r'^\d{6}_\d{4}', os.path.basename(d)):
            names.append(os.path.basename(d))
            break
    return names


def weights_input_size(path):
    """가중치 파일(또는 회차 폴더) 이름에 있는 입력 크기 표시(_s256 등)를 돌려줍니다. 224가 아닌 크기로 학습하면 train.py가 붙이며, 없으면 None입니다."""
    import re
    for n in _tag_names(path):
        m = re.search(r'_s(\d{3})(?=[_.]|$)', n)
        if m:
            return int(m.group(1))
    return None


def check_input_size(path):
    """가중치를 학습한 입력 크기와 지금 config.INPUT_SIZE가 다르면 멈춥니다. 크기가 다르면 판정이 틀어지기 때문입니다.
    이름에 표시가 없으면 224로 학습한 것으로 봅니다.
    """
    import config
    n = weights_input_size(path) or 224
    if n != config.RESIZE[0]:
        raise RuntimeError(f"가중치 {os.path.basename(str(path))} 은(는) 입력 {n}×{n} 으로 학습했는데 지금 입력 크기는 "
                           f"{config.RESIZE[0]} 입니다 — 환경변수 INPUT_SIZE={n} 으로 실행하세요(config.INPUT_SIZE)")


HEAD_POOL_MODES = ('avg', 'top3', 'avgtop3')


def weights_head_pool(path):
    """가중치 파일(또는 회차 폴더) 이름에 있는 헤드 시간 풀링 표시(_top3, _avgtop3)를 돌려줍니다. avg가 아닌 풀링으로 학습하면 train.py가 붙이며, 없으면 'avg'입니다."""
    import re
    for n in _tag_names(path):
        m = re.search(r'_(avgtop3|top3)(?=[_.]|$)', n)
        if m:
            return m.group(1)
    return 'avg'


def check_head_pool(path):
    """가중치를 학습한 헤드 시간 풀링에 맞춥니다. 모델을 만들기 전에 불러야 합니다.
    환경변수 HEAD_TEMPORAL_POOL을 주지 않았으면 이름의 표시(없으면 avg)를 따라 config.HEAD_TEMPORAL_POOL을 바꾸고,
    줬는데 다르면 멈춥니다. 풀링에는 가중치가 없어 다른 풀링으로도 가중치는 그대로 읽히고 판정만 틀어지므로 이름으로 맞춥니다.
    """
    import config
    mode = weights_head_pool(path)
    if mode == config.HEAD_TEMPORAL_POOL:
        return
    if not getattr(config, 'HEAD_TEMPORAL_POOL_SET', True):
        print(f"[헤드 풀링] 가중치 이름에 따라 {mode} 로 맞춤 — {os.path.basename(str(path))}")
        config.HEAD_TEMPORAL_POOL = mode
        return
    raise RuntimeError(f"가중치 {os.path.basename(str(path))} 은(는) 헤드 시간 풀링 {mode} 로 학습했는데 환경변수 "
                       f"HEAD_TEMPORAL_POOL={config.HEAD_TEMPORAL_POOL} 입니다 — {mode} 로 바꾸거나 환경변수를 비우면 "
                       f"가중치 이름을 따라갑니다(config.HEAD_TEMPORAL_POOL)")


def check_weights(path):
    """가중치를 읽기 전에 학습 조건을 확인합니다. 입력 크기(_s256)는 지금 설정과 같아야 하고(다르면 멈춤),
    헤드 시간 풀링(_top3 등)은 환경변수로 정하지 않았으면 이름을 따라갑니다. 모델을 만들기 전에 불러야 합니다.
    """
    check_input_size(path)
    check_head_pool(path)


class TemporalPool(nn.Module):
    """헤드에서 시간 축을 하나로 모읍니다. (B, C, T, H, W)를 (B, C, 1, H, W)로 만듭니다(config.HEAD_TEMPORAL_POOL).

    top3: 채널과 칸마다 시간 축에서 가장 큰 k(=3)개를 평균합니다. 2~6프레임짜리 짧은 흔들림이 1/30로 묽어지지 않습니다.
    avgtop3: 평균과 상위 k개 평균을 다시 평균합니다. 긴 맥락과 짧은 순간을 함께 보며, 값의 크기가 둘 사이라
    사전학습 post_conv의 입력 범위에서 크게 벗어나지 않습니다.
    avg는 이 모듈 대신 원래 AdaptiveAvgPool3d를 그대로 씁니다.
    mean과 topk만 써서 ONNX와 OpenVINO로 그대로 변환됩니다(infer_runtime 자가 검사로 결과를 확인합니다).
    """

    def __init__(self, mode, k=3):
        super().__init__()
        if mode not in ('top3', 'avgtop3'):
            raise ValueError(f'TemporalPool 은 top3·avgtop3 만: {mode!r}')
        self.mode, self.k = mode, int(k)

    def forward(self, x):
        top = torch.topk(x, min(self.k, x.shape[2]), dim=2).values.mean(dim=2, keepdim=True)
        if self.mode == 'top3':
            return top
        return 0.5 * (x.mean(dim=2, keepdim=True) + top)


# X3D-M 마지막 res 스테이지의 출력 채널 수입니다.
X3D_M_FEATURE_CHANNELS = 192
# 헤드 확장부(post_conv)를 거친 뒤의 채널 수입니다. 분류와 CAM은 이 차원에서 계산합니다.
X3D_HEAD_CHANNELS = 2048


# 분류 헤드 앞 dropout 비율입니다.
DROPOUT_P = 0.5

# gradient checkpointing입니다. 활성값을 저장하지 않고 역전파 때 앞 계산을 다시 해서 메모리를 줄입니다.
# 기울기는 같아 정확도에 영향이 없고, 속도는 약 30% 느려집니다.
# X3D는 시간축 30프레임을 그대로 유지하고 병목 안에서 채널을 넓혀 활성값이 큽니다. 10GB GPU에서 배치 15를 쓰려면 필요하며,
# 배치를 줄이면 배치에 사고 클립이 빠져 학습이 망가집니다. 학습 때만 동작하고 추론과 CAM 경로는 건드리지 않습니다.
GRAD_CHECKPOINT = os.getenv("GRAD_CHECKPOINT", "1").strip().lower() not in ("0", "false", "no")
# 백본 블록이 5개라 세그먼트를 5로 두면 블록 하나를 통째로 다시 계산합니다.
# 블록 수보다 크게 잡으면 checkpoint_sequential이 블록 안까지 나눠 메모리 최대치가 낮아집니다.
GRAD_CHECKPOINT_SEGMENTS = int(os.getenv("GRAD_CHECKPOINT_SEGMENTS", "5"))
# 체크포인팅할 앞쪽 블록 수입니다(features의 줄기, res2, res3, res4, res5 중 앞에서 몇 개).
# 비우면 checkpoint_sequential(세그먼트 5)로 줄기부터 res4까지 다시 계산하며, 4를 준 것과 같습니다.
# 메모리가 남으면 줄여서 재계산을 덜 합니다. python -m model.speed_check로 시간과 메모리를 재고 고릅니다.
# 주의: 체크포인팅한 블록은 다시 계산할 때 BN 이동평균도 같은 배치 통계로 한 번 더 갱신합니다(기울기는 같습니다).
# 그래서 범위가 바뀌는 블록은 BN momentum을 맞춰 기본 설정과 같은 이동평균이 되게 합니다(_match_bn_momentum).
_ckpt_blocks_env = os.getenv("GRAD_CHECKPOINT_BLOCKS", "").strip()
GRAD_CHECKPOINT_BLOCKS = int(_ckpt_blocks_env) if _ckpt_blocks_env else None

class HitAndRun3DCNN(nn.Module):
    """X3D-M 백본을 쓰는 물피도주 감지 모델입니다.

    X3D(Feichtenhofer, CVPR 2020)는 2D 분류망을 폭, 깊이, 해상도, 프레임 축으로 점진적으로 넓혀 얻은 효율적인 3D-CNN입니다.
    적은 연산으로 높은 정확도를 내므로, 긴 영상을 슬라이딩 윈도우로 훑는 이 서비스에서 묶음당 비용을 줄이기에 좋습니다.

    pytorchvideo 기본 헤드의 AvgPool3d 커널은 (16,7,7)로 고정돼 있어 30프레임 입력에서 동작하지 않습니다.
    헤드를 통째로 버리면 그 안의 확장 conv(192에서 432, 2048로)까지 사라져 사전학습 가중치를 옮길 수 없으므로,
    고정 커널 pool만 AdaptiveAvgPool3d((1, None, None))로 바꿉니다. 시간축만 모으고 공간 7x7을 남겨
    프레임 수와 상관없이 동작하고, 2048채널 공간 특징맵이 살아 있어 CAM을 뽑을 수 있습니다.

    서비스(inference.py)가 쓰는 인터페이스
    - 입력 (B, 3, T, 224, 224), 출력 (B, num_classes) logits
    - head_conv: 1x1x1 Conv3d 분류 헤드(CAM 가중치로도 씁니다)
    - cam_layer: CAM forward hook 대상(2048채널 공간 특징맵)
    - forward_with_cam: logits와 칸별 클래스 지도(CAM)를 함께 돌려줍니다(사고 클립 히트맵용)

    Args:
        num_classes: 분류 클래스 수(기본 2, S와 A)
        pretrained: True면 Kinetics-400 사전학습 가중치로 백본을 초기화합니다.
                    추론에서는 학습한 state_dict를 불러오므로 False로 만들어 불필요한 내려받기를 피합니다.
    """

    def __init__(self, num_classes=2, pretrained=False, dropout_p=DROPOUT_P, temporal_pool=None):
        super(HitAndRun3DCNN, self).__init__()
        base = x3d_m(pretrained=pretrained)
        # 헤드 시간 풀링(config.HEAD_TEMPORAL_POOL)입니다. avg는 원래 연산이고 top3, avgtop3는 TemporalPool이며, 가중치 키는 같습니다.
        if temporal_pool is None:
            import config
            temporal_pool = getattr(config, 'HEAD_TEMPORAL_POOL', 'avg')
        if temporal_pool not in HEAD_POOL_MODES:
            raise ValueError(f"HEAD_TEMPORAL_POOL 은 {'·'.join(HEAD_POOL_MODES)} 중 하나여야 합니다: {temporal_pool!r}")
        self.temporal_pool = temporal_pool

        # blocks[-1]은 헤드이므로 줄기와 res 스테이지만 떼어 냅니다.
        self.features = nn.Sequential(*list(base.blocks)[:-1])

        # 헤드의 확장부(pre_conv, BN, ReLU, pool, post_conv, ReLU)는 그대로 두고 고정 커널 pool만 바꿉니다.
        head_pool = base.blocks[-1].pool
        if temporal_pool == 'avg':
            head_pool.pool = nn.AdaptiveAvgPool3d((1, None, None))
        else:
            import config
            head_pool.pool = TemporalPool(temporal_pool, getattr(config, 'HEAD_TEMPORAL_TOPK', 3))
        self.head_pool = head_pool

        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=dropout_p)
        # 분류 헤드는 logit을 내므로 BN과 ReLU 없이 Conv3d 하나만 둡니다(CAM 가중치 겸용).
        # 원본 헤드의 Linear(2048, C)와 같은 연산이라 가중치 모양만 바꿔 옮길 수 있습니다.
        self.head_conv = nn.Conv3d(
            X3D_HEAD_CHANNELS, num_classes, kernel_size=1)
        self._match_bn_momentum()

    def default_checkpoint_blocks(self):
        """GRAD_CHECKPOINT_BLOCKS를 비웠을 때(checkpoint_sequential) 다시 계산하는 앞쪽 블록 수입니다. 기본은 4입니다."""
        n = len(self.features)
        seg = max(1, min(GRAD_CHECKPOINT_SEGMENTS, n))
        return (n // seg) * (seg - 1)

    def _match_bn_momentum(self):
        """체크포인팅 범위가 기본과 달라지는 블록의 BN momentum을 맞춥니다(위 GRAD_CHECKPOINT_BLOCKS 주석 참고).

        같은 배치 통계로 momentum m을 두 번 갱신하면 1-(1-m)^2로 한 번 갱신한 것과 같습니다.
        기본에서 다시 계산하던 블록이 빠지면 momentum을 올리고, 기본에서 안 하던 블록이 들어가면 1-sqrt(1-m)로 내립니다.
        momentum은 가중치 파일에 없어 추론과는 무관합니다.
        """
        if not GRAD_CHECKPOINT or GRAD_CHECKPOINT_BLOCKS is None:
            return
        k, d = GRAD_CHECKPOINT_BLOCKS, self.default_checkpoint_blocks()
        for i in range(min(k, d), min(max(k, d), len(self.features))):
            for m in self.features[i].modules():
                if isinstance(m, nn.modules.batchnorm._BatchNorm) and m.momentum is not None:
                    m.momentum = (1 - (1 - m.momentum) ** 2 if i >= k
                                  else 1 - (1 - m.momentum) ** 0.5)

    @property
    def cam_layer(self):
        """CAM이 특징맵을 받아 가는 층으로, 헤드 확장부 출력(2048채널, 공간 7x7)입니다.

        property라 모듈이 중복 등록되지 않습니다(가중치 키에 영향 없음). inference.py가
        model.cam_layer.register_forward_hook(...)으로 특징맵을 받습니다.
        head_conv.weight(2048채널)와 채널 수가 맞아야 CAM 가중합이 성립합니다.
        """
        return self.head_pool

    def forward(self, x):
        if self.training and GRAD_CHECKPOINT and GRAD_CHECKPOINT_BLOCKS is not None:
            # 앞쪽 GRAD_CHECKPOINT_BLOCKS개만 블록마다 체크포인팅합니다(checkpoint_sequential과 같은 계산).
            for i, block in enumerate(self.features):
                x = (checkpoint(block, x, use_reentrant=False)
                     if i < GRAD_CHECKPOINT_BLOCKS else block(x))
        elif self.training and GRAD_CHECKPOINT:
            # 백본만 체크포인팅합니다. head_pool은 CAM hook 대상이라 밖에 둡니다.
            x = checkpoint_sequential(
                self.features, GRAD_CHECKPOINT_SEGMENTS, x, use_reentrant=False)
        else:
            x = self.features(x)    # (B, 192, T, 7, 7)
        x = self.head_pool(x)       # (B, 2048, 1, 7, 7)
        x = self.avg_pool(x)        # (B, 2048, 1, 1, 1)
        x = self.dropout(x)
        x = self.head_conv(x)
        return x.squeeze(-1).squeeze(-1).squeeze(-1)

    def forward_with_cam(self, x):
        """forward와 같은 logits (B, C)와 칸별 클래스 지도(CAM) (B, C, 7, 7)를 돌려줍니다. 추론 전용입니다.

        CAM은 분류 헤드(1x1x1 Conv)를 공간 평균 전에 7x7 칸마다 적용한 것입니다. 평균과 1x1 Conv는 순서를 바꿔도 결과가 같아
        칸 평균이 곧 logits이고, 칸마다 'A - S'를 보면 그 칸이 사고 쪽으로 얼마나 기울었는지 알 수 있습니다.
        판정에 이미 계산한 특징맵을 쓰므로 추가 비용은 칸 49개의 1x1 곱셈뿐이며, ONNX와 OpenVINO도 두 출력을 함께 내보냅니다(infer_runtime.py).
        logits는 forward와 같은 경로(평균 뒤 Conv)로 계산해 판정 결과가 바뀌지 않게 합니다.
        """
        x = self.head_pool(self.features(x))            # (B, 2048, 1, 7, 7)
        cam = self.head_conv(x)[:, :, 0]                # (B, C, 7, 7)
        x = self.head_conv(self.dropout(self.avg_pool(x)))
        return x.squeeze(-1).squeeze(-1).squeeze(-1), cam
