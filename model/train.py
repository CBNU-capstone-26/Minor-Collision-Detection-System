import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

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

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'조기 종료 카운트: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if val_loss < self.val_loss_min:
            print(
                f'검증 손실 감소 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 모델 저장 중...')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss


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
    device = get_device()
    cuda_like = is_cuda_like(device)
    print(f"사용 중인 디바이스: {device}")

    if cuda_like:
        torch.backends.cudnn.benchmark = True

    full_dataset = HitAndRunDataset(
        data_dir=data_dir, clip_length=clip_length, r_value=r_value, resize=resize
    )
    train_size = int(train_split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(
        full_dataset, [train_size, val_size])

    train_loader = _make_loader(
        train_subset, batch_size=batch_size, shuffle=True, device=device)
    val_loader = _make_loader(
        val_subset, batch_size=batch_size, shuffle=False, device=device)

    model = HitAndRun3DCNN(num_classes=num_classes).to(device)

    # AMP, GradScaler: CUDA/ROCm 공통 지원
    # channels_last_3d: NVIDIA CUDA 전용 (ROCm 미지원)
    amp_enabled = use_amp and cuda_like
    channels_last_enabled = use_channels_last and is_channels_last_3d_supported(
        device)

    if channels_last_enabled:
        model = model.to(memory_format=torch.channels_last_3d)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

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
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                non_blocking = cuda_like
                inputs = inputs.to(device, non_blocking=non_blocking)
                labels = labels.to(device, non_blocking=non_blocking)
                if channels_last_enabled:
                    inputs = inputs.contiguous(
                        memory_format=torch.channels_last_3d)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                preds = outputs.argmax(dim=1)
                correct += torch.sum(preds == labels)

        avg_val_loss = val_loss / len(val_loader.dataset)
        val_acc = correct.double() / len(val_loader.dataset)

        print(f'Epoch [{epoch+1}/{num_epochs}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}')

        early_stopping(avg_val_loss, model)
        if early_stopping.early_stop:
            print("조기 종료 조건 충족. 학습을 중단합니다.")
            break

    # DirectML은 map_location 직접 지원이 불안정하므로 CPU 경유 로드
    state_dict = torch.load(save_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
