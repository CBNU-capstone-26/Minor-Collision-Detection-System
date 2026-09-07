import os
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import config
from dataset import HitAndRunDataset
from device_utils import get_device, is_cuda_like, is_channels_last_3d_supported
from hitandrun_model import HitAndRun3DCNN


class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        # 최종 파일명 규칙에 쓸 값 — best 가중치가 저장된 에포크(1-index)
        self.best_epoch = None

    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'조기 종료 카운트: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        if val_loss < self.val_loss_min:
            print(
                f'검증 손실 감소 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 모델 저장 중...')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
            self.best_epoch = epoch


def _make_loader(dataset, batch_size, shuffle, device):
    # CUDA: 멀티워커 + prefetch 활성화 (① Ryzen 7 5700X 16스레드 기준)
    # CPU : Windows 멀티프로세싱 충돌 방지 및 디바이스 컨텍스트 공유 불가 문제로 단일 프로세스 사용
    if is_cuda_like(device):
        num_workers = min(8, os.cpu_count() or 2)
    else:
        num_workers = 0
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': is_cuda_like(device),
    }
    if num_workers > 0:
        # ② prefetch_factor 증가 — GPU 연산 중 다음 배치를 더 많이 미리 준비
        kwargs.update({'persistent_workers': True, 'prefetch_factor': 4})
    return DataLoader(dataset, **kwargs)


def train_model(
    data_dir=config.DATA_DIR,
    num_classes=config.MODEL_NUM_CLASSES,
    batch_size=config.TRAIN_BATCH_SIZE,
    num_epochs=config.TRAIN_NUM_EPOCHS,
    clip_length=config.CLIP_LENGTH,
    r_value=config.R_VALUE,
    resize=config.RESIZE,
    save_path=config.TRAIN_BEST_MODEL_SAVE_PATH,
    train_split_ratio=config.TRAIN_SPLIT_RATIO,
    early_stopping_patience=config.TRAIN_EARLY_STOPPING_PATIENCE,
    learning_rate=config.TRAIN_LEARNING_RATE,
    use_amp=config.USE_AMP,
    use_channels_last=config.USE_CHANNELS_LAST,
):
    device = get_device(config.TRAIN_DEVICE_TYPE)
    cuda_like = is_cuda_like(device)
    print(f"사용 중인 디바이스: {device}")

    if cuda_like:
        torch.backends.cudnn.benchmark = True

    # 학습용(증강 O) / 검증용(증강 X) 데이터셋을 각각 생성해 동일 인덱스로 분할.
    # → 검증 세트는 매 epoch 결정적이라 val loss가 안정되고 best model 선택이 신뢰됨.
    train_dataset = HitAndRunDataset(
        data_dir=data_dir, clip_length=clip_length, r_value=r_value,
        resize=resize, augment=True,
    )
    val_dataset = HitAndRunDataset(
        data_dir=data_dir, clip_length=clip_length, r_value=r_value,
        resize=resize, augment=False,
    )
    # 계층적 분할(stratified): 그룹을 세분화해 각 그룹을 train_split_ratio(8:2)로 나눈다.
    #   rc  : 방향(N/S/L/R) × 클래스(A/S) → 최대 8그룹 (각 방향·클래스가 train/val에 고루 분포)
    #   real: 클래스(A/S)만 → 2그룹 (실제 영상엔 방향 개념이 없음)
    #   → train은 모든 그룹의 학습분(80%)을 합쳐 학습, val은 rc용/real용을 분리해 각각 측정.
    def _domain(mp4_path):
        parts = os.path.normpath(mp4_path).split(os.sep)
        return 'real' if 'realdata' in parts else 'rc'

    def _direction(file_name):
        # 파일명 예: 220510_LA_0001 → 두 번째 토큰의 첫 글자가 방향(N/L/R/S)
        seg = file_name.split('_')
        return seg[1][0] if len(seg) > 1 and seg[1] else '?'

    def _scenario(file_name):
        # 두 번째 토큰의 마지막 글자가 시나리오/클래스 코드
        #   rc: A(충돌)/V(주차)/S(직진)/W(배회),  real: A(충돌)/S(비충돌)
        seg = file_name.split('_')
        return seg[1][-1] if len(seg) > 1 and seg[1] else '?'

    # 층화 그룹: rc = 방향(N/L/R/S) × 시나리오(A/V/S/W) → 최대 16그룹,
    #            real = 시나리오(A/S) → 2그룹. 각 그룹을 train_split_ratio(8:2)로 나눈다.
    #   ※ 실제 학습 라벨은 그대로 txt 기반 이진값(A vs 비A) — 시나리오는 '분할 균형'에만 사용.
    groups = {}  # (domain, direction, scenario) -> [sample_idx, ...]
    for i, s in enumerate(train_dataset.samples):
        dom = _domain(s['mp4_path'])
        direction = _direction(s['file_name']) if dom == 'rc' else '-'
        groups.setdefault((dom, direction, _scenario(s['file_name'])), []).append(i)

    gen = torch.Generator().manual_seed(42)  # 재현 가능한 분할 (seed 고정)
    train_idx, val_rc_idx, val_real_idx = [], [], []
    for (dom, direction, scen), idxs in sorted(groups.items()):
        shuffled = [idxs[i]
                    for i in torch.randperm(len(idxs), generator=gen).tolist()]
        cut = int(train_split_ratio * len(idxs))
        train_idx += shuffled[:cut]
        (val_real_idx if dom == 'real' else val_rc_idx).extend(shuffled[cut:])

    print(f"[분할] train {len(train_idx)} / val_rc {len(val_rc_idx)} / val_real {len(val_real_idx)}")
    print("  그룹별 개수: " + ", ".join(
        f"{d}{'' if dr == '-' else '-' + dr}-{sc}:{len(v)}"
        for (d, dr, sc), v in sorted(groups.items())))

    train_loader = _make_loader(
        Subset(train_dataset, train_idx), batch_size=batch_size,
        shuffle=True, device=device)
    val_rc_loader = _make_loader(
        Subset(val_dataset, val_rc_idx), batch_size=batch_size,
        shuffle=False, device=device) if val_rc_idx else None
    val_real_loader = _make_loader(
        Subset(val_dataset, val_real_idx), batch_size=batch_size,
        shuffle=False, device=device) if val_real_idx else None

    # 학습 시에는 Kinetics-400 사전학습 가중치로 초기화 (config.PRETRAINED)
    model = HitAndRun3DCNN(
        num_classes=num_classes, pretrained=config.PRETRAINED).to(device)

    # AMP, GradScaler: CUDA/ROCm 공통 지원
    # channels_last_3d: NVIDIA CUDA 전용 (ROCm 미지원)
    amp_enabled = use_amp and cuda_like
    channels_last_enabled = use_channels_last and is_channels_last_3d_supported(
        device)

    if channels_last_enabled:
        model = model.to(memory_format=torch.channels_last_3d)

    criterion = nn.CrossEntropyLoss()
    # 미세조정 표준 관행: 사전학습된 백본(features)은 낮은 LR(×0.1)로 보수적으로,
    # 새로 초기화된 분류 헤드(head_conv)는 기본 LR로 학습한다.
    optimizer = optim.Adam([
        {'params': model.features.parameters(), 'lr': learning_rate * 0.1},
        {'params': model.head_conv.parameters(), 'lr': learning_rate},
    ])

    # ── 안전한 학습 안정화 장치 (모델 구조·손실 불변, 성능 저하 없음) ──
    # 1) ReduceLROnPlateau: val loss가 정체되면 LR을 절반으로 낮춰 수렴을 돕는다.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    # 2) gradient clipping: 그래디언트 노름 상한(폭주/진동 억제). 필요 시 조정.
    grad_clip_norm = 1.0

    if amp_enabled:
        scaler = torch.amp.GradScaler("cuda")

    early_stopping = EarlyStopping(
        patience=early_stopping_patience, path=save_path)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            non_blocking = cuda_like
            inputs = inputs.to(device, non_blocking=non_blocking)
            labels = labels.to(device, non_blocking=non_blocking)
            if channels_last_enabled:
                inputs = inputs.contiguous(
                    memory_format=torch.channels_last_3d)

            optimizer.zero_grad(set_to_none=True)

            if amp_enabled:
                with torch.amp.autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)  # 클리핑 전 스케일 해제 (AMP)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_norm)
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        # rc용/real용 val을 각각 분리해서 성능 측정
        def _evaluate(loader):
            """val 로더 하나에 대해 (avg_loss, acc) 반환. 로더 없으면 (None, None)."""
            if loader is None:
                return None, None
            model.eval()
            vloss, correct = 0.0, 0
            with torch.inference_mode():
                for inputs, labels in loader:
                    inputs = inputs.to(device, non_blocking=cuda_like)
                    labels = labels.to(device, non_blocking=cuda_like)
                    if channels_last_enabled:
                        inputs = inputs.contiguous(
                            memory_format=torch.channels_last_3d)
                    outputs = model(inputs)
                    vloss += criterion(outputs, labels).item() * inputs.size(0)
                    correct += torch.sum(outputs.argmax(dim=1) == labels).item()
            n = len(loader.dataset)
            return vloss / n, correct / n

        rc_loss, rc_acc = _evaluate(val_rc_loader)
        real_loss, real_acc = _evaluate(val_real_loader)

        # 조기종료·스케줄러·best 저장 기준: 실제 영상 성능이 목표이므로 real val 우선.
        # real val이 없으면(rc만 학습) rc val로 대체.
        monitor_loss = real_loss if real_loss is not None else rc_loss
        scheduler.step(monitor_loss)
        current_lr = optimizer.param_groups[-1]['lr']  # 헤드 LR 표시

        def _fmt(l, a):
            return f'{l:.4f}/{a:.4f}' if l is not None else 'N/A'
        print(f'Epoch [{epoch+1}/{num_epochs}] Train {avg_train_loss:.4f} | '
              f'val_rc(L/Acc) {_fmt(rc_loss, rc_acc)} | '
              f'val_real(L/Acc) {_fmt(real_loss, real_acc)} | LR {current_lr:.2e}')

        # best 저장 기준은 real val (파일명 손실율도 real val 기준으로 기록됨)
        early_stopping(monitor_loss, model, epoch + 1)
        if early_stopping.early_stop:
            print("조기 종료 조건 충족. 학습을 중단합니다.")
            break

    # ── 최종 가중치 파일명 규칙 적용 ───────────────────────────────
    # hitandrun_[날짜YYMMDD]_[에포크수]ep_[earlyY|earlyN]_[손실율]
    #   · 날짜   : 학습 종료일 (예: 2026-07-24 → 260724)
    #   · 에포크 : best 가중치가 저장된 에포크 (예: 50 → 50ep)
    #   · early  : 조기종료로 멈췄으면 earlyY, 아니면 earlyN
    #   · 손실율 : 저장 당시 val loss를 소수점 포함해 표기 (예: 1.2345 → 1.2345)
    date_str = datetime.now().strftime('%y%m%d')
    epoch_str = f'{early_stopping.best_epoch}ep'
    early_str = 'earlyY' if early_stopping.early_stop else 'earlyN'
    loss_str = f'{early_stopping.val_loss_min:.4f}'
    final_name = f'hitandrun_{date_str}_{epoch_str}_{early_str}_{loss_str}.pth'
    final_path = Path(save_path).with_name(final_name)
    # 학습 중엔 save_path(작업용)로 저장해 두고, 종료 시 규칙 파일명으로 이동
    os.replace(save_path, final_path)
    print(f'\n최종 가중치 저장: {final_path}')

    state_dict = torch.load(final_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
