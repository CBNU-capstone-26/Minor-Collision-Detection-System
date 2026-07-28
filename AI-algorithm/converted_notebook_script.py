# ============================================================from google.colab import drive
drive.mount('/content/drive')

# ============================================================from pathlib import Path

# ==========================================
# [파라미터 사용자 설정]
# 실행 환경에 맞게 이 셀의 값만 수정하세요.
# ==========================================

# ---------- 공통 설정 ----------
DATA_DIR = Path("/content/drive/MyDrive/capstone-26/dataset")  # 학습/평가에 사용할 mp4, txt 데이터 폴더 경로
MODEL_NUM_CLASSES = 2  # 분류할 클래스 수: 0=정상, 1=충돌
CLIP_LENGTH = 30  # 3D-CNN에 입력할 클립 길이(프레임 수)
RESIZE = (224, 224)  # 모델 입력용 프레임 크기(width, height)
R_VALUE = 1.0  # 차량 bbox 주변 크롭 여유 비율
TARGET_ID = 0  # 예측/평가할 기본 차량 ID
USE_AMP = True  # CUDA 학습 시 자동 혼합 정밀도(AMP) 사용 여부
USE_CHANNELS_LAST = True  # CUDA 사용 시 3D 텐서 channels_last 메모리 포맷 사용 여부

# ---------- 학습 전용 ----------
TRAIN_BEST_MODEL_SAVE_PATH = Path("/content/drive/MyDrive/capstone-26/best_hitandrun_model.pth")  # 학습 중 가장 좋은 모델을 저장할 경로
TRAIN_BATCH_SIZE = 15  # 학습/검증 DataLoader 배치 크기
TRAIN_NUM_EPOCHS = 5  # 최대 학습 epoch 수
TRAIN_SPLIT_RATIO = 0.8  # 전체 데이터 중 학습 데이터 비율
TRAIN_EARLY_STOPPING_PATIENCE = 10  # 검증 손실 개선이 없을 때 기다릴 epoch 수

# ---------- 단일 영상 예측/CAM 출력 전용 ----------
PREDICT_WEIGHTS_PATH = Path("/content/drive/MyDrive/capstone-26/best_hitandrun_model.pth")  # 단일 영상 예측에 불러올 가중치 파일 경로
PREDICT_VIDEO_PATH = Path("/content/drive/MyDrive/capstone-26/dataset_real/real01.mp4")  # 분석할 단일 영상 파일 경로
PREDICT_TXT_PATH = Path("/content/drive/MyDrive/capstone-26/dataset_real/real01.gdoc")  # 단일 영상에 대응하는 bbox txt 파일 경로
PREDICT_OUTPUT_DIR = Path("/content/drive/MyDrive/capstone-26/predict_result")  # CAM 예측 결과 영상 저장 폴더 경로
PREDICT_INFER_BATCH_SIZE = 16  # 단일 영상 슬라이딩 윈도우 추론 배치 크기
PREDICT_WINDOW_STRIDE = 3  # 단일 영상 추론 시 슬라이딩 윈도우 이동 간격(프레임)

# ---------- 랜덤영상 정확도 예측 전용 ----------
EVAL_WEIGHTS_PATH = Path("/content/drive/MyDrive/capstone-26/hitandrun_model_best.pth")  # 랜덤영상 정확도 예측에 불러올 가중치 파일 경로
EVAL_FOLDER_PATH = Path("/content/drive/MyDrive/capstone-26/dataset")  # 랜덤으로 샘플링할 영상/txt 폴더 경로
EVAL_NUM_SAMPLES = 50  # 랜덤으로 평가할 영상 수(None이면 전체 평가)
EVAL_INFER_BATCH_SIZE = 32  # 랜덤영상 정확도 예측 추론 배치 크기
EVAL_WINDOW_STRIDE = 3  # 랜덤영상 정확도 예측 시 슬라이딩 윈도우 이동 간격(프레임)

# ============================================================# Colab용 환경설정
# 필요한 경우 opencv 설치 (Colab에는 기본적으로 설치되어 있으나 확인용)
!pip install opencv-python

# ============================================================# 실제데이터 라벨링용
# 영상에서 차량의 id와 바운딩 박스 좌표가 반환

# 1. YOLOv8 패키지 설치 (Colab 환경)
!pip install -q ultralytics

import cv2
import matplotlib.pyplot as plt
from ultralytics import YOLO

# ==========================================
# ⚙️ 설정: 실제 영상 경로를 입력하세요
# ==========================================
video_path = '/content/drive/MyDrive/capstone-26/dataset_real/real01.mp4' # 분석할 실제 비디오 경로
txt_output_path = video_path.replace('.mp4', '.txt') # 동일한 이름의 txt 파일로 자동 저장

# 2. 사전 학습된 YOLOv8 모델 로드 (가볍고 빠른 nano 버전)
model = YOLO('yolov8n.pt')

# 3. 비디오에서 첫 프레임 추출
cap = cv2.VideoCapture(video_path)
ret, frame = cap.read()
if not ret:
    print("❌ 비디오를 읽을 수 없습니다. 경로를 다시 확인해 주세요.")
    cap.release()
else:
    cap.release()

    # BGR을 RGB로 변환 (시각화용)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 4. YOLO 모델로 차량 객체 탐지 수행
    print("🔍 영상에서 차량을 탐지하는 중입니다...")
    results = model(frame_rgb, verbose=False)

    # COCO 데이터셋 기준 차량 관련 클래스 ID (2: car, 5: bus, 7: truck)
    vehicle_classes = [2, 5, 7]
    detected_cars = []

    # 5. 좌표 추출 및 텍스트 파일 저장 (논문 형식)
    with open(txt_output_path, 'w') as f:
        car_id = 0
        for r in results:
            boxes = r.boxes
            for box in boxes:
                cls = int(box.cls[0])
                if cls in vehicle_classes: # 탐지된 객체가 차량인 경우만
                    # 좌표 추출 및 정수 변환
                    x1, y1, x2, y2 = map(int, box.xyxy[0])

                    # 논문 포맷 작성: car,ID,x_min,y_min,x_max,y_max
                    line = f"car,{car_id},{x1},{y1},{x2},{y2}\n"
                    f.write(line)

                    # 시각화를 위해 리스트에 저장
                    detected_cars.append((car_id, x1, y1, x2, y2))
                    car_id += 1

    print(f"\n✅ 라벨링 텍스트 파일 자동 생성 완료: {txt_output_path}")
    print(f"총 {len(detected_cars)}대의 차량이 감지되었습니다.\n")

    # 6. 사용자 확인을 위한 시각화 (어떤 차가 몇 번 ID인지 확인)
    plt.figure(figsize=(16, 9))
    plt.imshow(frame_rgb)
    ax = plt.gca()

    for cid, x1, y1, x2, y2 in detected_cars:
        # 붉은색 바운딩 박스 그리기
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        # 차량 ID 표시
        plt.text(x1, y1-10, f'ID: {cid}', color='white', fontsize=12, fontweight='bold', backgroundcolor='red')

    plt.title("Detected Vehicles and Auto-assigned IDs", fontsize=15)
    plt.axis('off')
    plt.show()

    print("💡 [사용 방법]")
    print("위 이미지에서 감시(예측)하고 싶은 타겟 차량의 'ID 번호'를 확인하세요.")
    print("이후 3D-CNN 예측 코드의 'target_id'에 해당 번호를 입력하고 돌리시면 됩니다!")
# ============================================================# [데이터 전처리 및 Custom Dataset 셀]

import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
import random
import torchvision.transforms.functional as TF
from PIL import Image

# ==========================================
# [1단계] 데이터 전처리 및 Custom Dataset (데이터 증강 적용 완료)
# ==========================================
class HitAndRunDataset(Dataset):
    def __init__(self, data_dir, clip_length=30, r_value=1.0, resize=(224, 224)):
        self.data_dir = data_dir
        self.clip_length = clip_length
        self.r_value = r_value
        self.resize = resize
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)

        self.file_names = sorted(f.rsplit('.', 1)[0] for f in os.listdir(data_dir) if f.endswith('.mp4'))
        self.samples = self._build_index()

    def __len__(self):
        return len(self.samples)

    def _build_index(self):
        samples = []
        for file_name in self.file_names:
            mp4_path = os.path.join(self.data_dir, f"{file_name}.mp4")
            txt_path = os.path.join(self.data_dir, f"{file_name}.txt")
            bboxes, action = self._parse_annotation(txt_path)

            if action is not None:
                class_str = action[0]
                target_id = int(float(action[1]))
                start_f = int(float(action[2]))
            else:
                class_str = 'S'
                target_id = 0
                start_f = 0

            if target_id not in bboxes:
                target_id = next(iter(bboxes), 0)
            target_bbox = bboxes.get(target_id, [0, 0, 224, 224])
            label = 1 if class_str == 'A' else 0

            samples.append({
                'file_name': file_name,
                'mp4_path': mp4_path,
                'label': label,
                'start_f': start_f,
                'bbox': target_bbox,
            })
        return samples

    def _parse_annotation(self, txt_path):
        bboxes = {}
        action = None
        if os.path.exists(txt_path):
            with open(txt_path, 'r') as f:
                for line in f:
                    parts = line.strip().split(',')
                    if not parts or len(parts) < 2:
                        continue
                    if parts[0] == 'car' and len(parts) >= 6:
                        bboxes[int(parts[1])] = [int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])]
                    elif parts[0] in ['A', 'S']:
                        action = parts
        return bboxes, action

    def _crop_and_pad(self, frame, bbox, r):
        h, w, _ = frame.shape
        x_min, y_min, x_max, y_max = bbox

        veh_w, veh_h = (x_max - x_min) * r, (y_max - y_min) * r
        cx, cy = x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2

        square_size = int(max(veh_w, veh_h))
        new_x_min, new_y_min = cx - square_size // 2, cy - square_size // 2
        new_x_max, new_y_max = cx + square_size // 2, cy + square_size // 2

        pad_left, pad_top = max(0, -new_x_min), max(0, -new_y_min)
        pad_right, pad_bottom = max(0, new_x_max - w), max(0, new_y_max - h)

        valid_x_min, valid_y_min = max(0, new_x_min), max(0, new_y_min)
        valid_x_max, valid_y_max = min(w, new_x_max), min(h, new_y_max)
        cropped_frame = frame[valid_y_min:valid_y_max, valid_x_min:valid_x_max]

        if pad_left > 0 or pad_top > 0 or pad_right > 0 or pad_bottom > 0:
            cropped_frame = np.pad(
                cropped_frame,
                ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                mode='constant',
                constant_values=0,
            )

        return cv2.resize(cropped_frame, self.resize, interpolation=cv2.INTER_LINEAR)

    def _apply_augmentation(self, frames):
        """논문에 등장하는 Data Augmentation 기법 실시간(온라인) 적용"""
        # 1. Horizontal Flip (50% 확률)
        do_hflip = random.random() > 0.5

        # 2. Random Rotation (-10 ~ 10도 사이)
        do_rot = random.random() > 0.5
        angle = random.uniform(-10, 10) if do_rot else 0

        # 3. Color Jitter (밝기, 대비 미세 조절)
        do_jitter = random.random() > 0.5
        brightness = random.uniform(0.9, 1.1)
        contrast = random.uniform(0.9, 1.1)

        aug_frames = []
        for frame in frames:
            # OpenCV Numpy 배열을 PIL 이미지로 변환하여 증강 적용
            img = Image.fromarray(frame)

            if do_hflip:
                img = TF.hflip(img)
            if do_rot:
                img = TF.rotate(img, angle)
            if do_jitter:
                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)

            aug_frames.append(np.array(img))

        return aug_frames

    def _frames_to_tensor(self, frames):
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        video_tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
        return (video_tensor - self.mean) / self.std

    def __getitem__(self, idx):
        sample = self.samples[idx]
        cap = cv2.VideoCapture(sample['mp4_path'])
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, sample['start_f'])

        for _ in range(self.clip_length):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self._crop_and_pad(frame, sample['bbox'], self.r_value))
        cap.release()

        while len(frames) < self.clip_length:
            frames.append(frames[-1] if frames else np.zeros((self.resize[1], self.resize[0], 3), dtype=np.uint8))

        # --- 🚀 핵심: 모델에 들어가기 직전에 무작위 데이터 증강 적용 ---
        frames = self._apply_augmentation(frames)

        return self._frames_to_tensor(frames), torch.tensor(sample['label'], dtype=torch.long)
# ============================================================# [모델 설계 셀]

# ==========================================
# [2단계] I3D 기반 3D-CNN 모델 설계
# ==========================================
import torch
import torch.nn as nn

class InceptionModule3D(nn.Module):
    def __init__(self, in_channels, n1x1, n3x3_reduce, n3x3, n5x5_reduce, n5x5, pool_proj):
        super(InceptionModule3D, self).__init__()
        # Branch 1: 1x1x1
        self.branch1 = nn.Sequential(
            nn.Conv3d(in_channels, n1x1, kernel_size=1),
            nn.ReLU(inplace=True)
        )
        # Branch 2: 1x1x1 -> 3x3x3
        self.branch2 = nn.Sequential(
            nn.Conv3d(in_channels, n3x3_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(n3x3_reduce, n3x3, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Branch 3: 1x1x1 -> 3x3x3 (논문의 5x5 확장 대체 구조)
        self.branch3 = nn.Sequential(
            nn.Conv3d(in_channels, n5x5_reduce, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(n5x5_reduce, n5x5, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)
        )
        # Branch 4: MaxPool3d -> 1x1x1 (수학적 결합을 위해 논문의 의심스러운 Stride를 1로 정상화)
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, pool_proj, kernel_size=1),
            nn.ReLU(inplace=True)
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

        # 1. 초기 특징 추출 레이어 (Figure 11 맨 하단부)
        self.conv1 = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(7, 7, 7), stride=(2, 2, 2), padding=(3, 3, 3)),
            nn.ReLU(inplace=True)
        )
        self.maxpool1 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.conv2 = nn.Sequential(
            nn.Conv3d(64, 64, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 192, kernel_size=3, padding=1),
            nn.ReLU(inplace=True)

        self.maxpool2 = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        # 2. 첫 번째 블록: Inception 2개 연속 배치
        self.inception3a = InceptionModule3D(192, 64, 96, 128, 16, 32, 32)   # Out: 256
        self.inception3b = InceptionModule3D(256, 128, 128, 192, 32, 96, 64) # Out: 480
        self.maxpool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)     # Stride=2,2,2 다운샘플링

        # 3. 두 번째 블록: Inception 5개 연속 배치
        self.inception4a = InceptionModule3D(480, 192, 96, 208, 16, 48, 64)  # Out: 512
        self.inception4b = InceptionModule3D(512, 160, 112, 224, 24, 64, 64) # Out: 512
        self.inception4c = InceptionModule3D(512, 128, 128, 256, 24, 64, 64) # Out: 512
        self.inception4d = InceptionModule3D(512, 112, 144, 288, 32, 64, 64) # Out: 528
        self.inception4e = InceptionModule3D(528, 256, 160, 320, 32, 128, 128) # Out: 832
        self.maxpool4 = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=(1, 2, 2), padding=1) # Stride=1,2,2 다운샘플링

        # 4. 세 번째 블록: Inception 2개 연속 배치 (최종 레이어 완성)
        self.inception5a = InceptionModule3D(832, 256, 160, 320, 32, 128, 128) # Out: 832
        self.inception5b = InceptionModule3D(832, 384, 192, 384, 48, 128, 128) # Out: 1024

        # 5. 탑 다운 풀링 및 출력 헤드 (3D-CAM 구현을 위한 1x1x1 Conv 설계)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(p=0.4)
        self.head_conv = nn.Conv3d(1024, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.conv2(x)
        x = self.maxpool2(x)

        # Block 1
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool3(x)

        # Block 2
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool4(x)

        # Block 3
        x = self.inception5a(x)
        x = self.inception5b(x) # 최종 CAM을 위한 Target Feature Map 단계

        x = self.avg_pool(x)
        x = self.dropout(x)
        x = self.head_conv(x)
        return x.squeeze(-1).squeeze(-1).squeeze(-1) # (Batch, num_classes)

# ============================================================# [학습 전용 셀]

# ==========================================
# [3단계] 조기 종료 및 학습 루프 실행부
# ==========================================
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
            print(f'검증 손실 감소 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 모델 저장 중...')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss

def _make_loader(dataset, batch_size, shuffle, device):
    num_workers = min(4, os.cpu_count() or 2)
    kwargs = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers,
        'pin_memory': device.type == 'cuda',
    }
    if num_workers > 0:
        kwargs.update({'persistent_workers': True, 'prefetch_factor': 2})
    return DataLoader(dataset, **kwargs)

def train_model(data_dir, batch_size=15, num_epochs=5, clip_length=30, r_value=1.0, resize=(224, 224), save_path='best_model.pth', train_split_ratio=0.8, early_stopping_patience=10, use_amp=True, use_channels_last=True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 사용 중인 디바이스: {device}")

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    full_dataset = HitAndRunDataset(data_dir=data_dir, clip_length=clip_length, r_value=r_value, resize=resize)
    train_size = int(train_split_ratio * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = random_split(full_dataset, [train_size, val_size])

    train_loader = _make_loader(train_subset, batch_size=batch_size, shuffle=True, device=device)
    val_loader = _make_loader(val_subset, batch_size=batch_size, shuffle=False, device=device)

    model = HitAndRun3DCNN(num_classes=2).to(device)
    channels_last_enabled = use_channels_last and device.type == 'cuda'
    if channels_last_enabled:
        model = model.to(memory_format=torch.channels_last_3d)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.00001)
    amp_enabled = use_amp and device.type == 'cuda'
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    early_stopping = EarlyStopping(patience=early_stopping_patience, path=save_path)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if channels_last_enabled:
                inputs = inputs.contiguous(memory_format=torch.channels_last_3d)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=amp_enabled):
                outputs = model(inputs)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            train_loss += loss.item() * inputs.size(0)

        avg_train_loss = train_loss / len(train_loader.dataset)

        model.eval()
        val_loss = 0.0
        correct = 0
        with torch.inference_mode():
            for inputs, labels in val_loader:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                if channels_last_enabled:
                    inputs = inputs.contiguous(memory_format=torch.channels_last_3d)

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
            print("🛑 조기 종료 조건 충족. 학습을 중단합니다.")
            break

    model.load_state_dict(torch.load(save_path, map_location=device))
    return model

# --- 실제 학습을 시작하는 부분 ---
if os.path.exists(DATA_DIR):
    print(f"데이터 디렉토리 확인 완료: {DATA_DIR}. 학습을 시작합니다!")
    trained_model = train_model(
        data_dir=DATA_DIR,
        batch_size=TRAIN_BATCH_SIZE,
        num_epochs=TRAIN_NUM_EPOCHS,
        clip_length=CLIP_LENGTH,
        r_value=R_VALUE,
        resize=RESIZE,
        save_path=TRAIN_BEST_MODEL_SAVE_PATH,
        train_split_ratio=TRAIN_SPLIT_RATIO,
        early_stopping_patience=TRAIN_EARLY_STOPPING_PATIENCE,
        use_amp=USE_AMP,
        use_channels_last=USE_CHANNELS_LAST,
    )
else:
    print(f"⚠️ 데이터 폴더를 찾을 수 없습니다: {DATA_DIR}")

# ============================================================# [단일 영상 예측/CAM 출력 전용 셀]

import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import os

# ==========================================
# 1. CAM 출력을 위한 특징 맵 추출기 (Hook)
# ==========================================
activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def frames_to_video_tensor(frames):
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
    return (tensor - mean) / std

# ==========================================
# 2. 정사각형 패딩 + 라벨 프리 + 전체 화면 CAM 예측 함수
# ==========================================
def predict_hit_and_run_final(model, video_path, txt_path, target_id=0, r_value=1.0, resize=(224, 224), output_dir=None, infer_batch_size=16, window_stride=3):
    video_path = str(video_path)
    txt_path = str(txt_path)
    if output_dir is None:
        output_dir = PREDICT_OUTPUT_DIR
    output_dir = Path(output_dir)
    print(f"[{os.path.basename(video_path)}] 전체 화면 분석 시작...")
    device = next(model.parameters()).device
    model.eval()
    window_stride = max(1, int(window_stride))

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    # 훅(Hook) 대상을 맨 마지막 계층인 inception5b로 타겟팅해서 3D Feature Map을 추적
    handle = loaded_model.inception5b.register_forward_hook(get_activation('inception5b'))

    try:
        bboxes = {}
        if not os.path.exists(txt_path):
            print(f"⚠️ 텍스트 파일을 찾을 수 없습니다: {txt_path}")
            return

        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6 and parts[0] == 'car':
                    bboxes[int(parts[1])] = [int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5])]

        if target_id not in bboxes:
            print(f"⚠️ ID {target_id}번 차량의 좌표가 없습니다. 존재하는 ID: {list(bboxes.keys())}")
            return

        target_bbox = bboxes[target_id]

        def crop_square_and_pad(frame, bbox, r):
            h, w, _ = frame.shape
            x_min, y_min, x_max, y_max = bbox
            vw, vh = (x_max - x_min) * r, (y_max - y_min) * r
            cx, cy = x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2
            side = int(max(vw, vh))
            nx1, ny1 = cx - side // 2, cy - side // 2
            nx2, ny2 = cx + side // 2, cy + side // 2
            v_x1, v_y1 = max(0, nx1), max(0, ny1)
            v_x2, v_y2 = min(w, nx2), min(h, ny2)
            p_l, p_t = max(0, -nx1), max(0, -ny1)
            p_r, p_b = max(0, nx2 - w), max(0, ny2 - h)
            cropped = frame[v_y1:v_y2, v_x1:v_x2]
            if p_l > 0 or p_t > 0 or p_r > 0 or p_b > 0:
                cropped = np.pad(cropped, ((p_t, p_b), (p_l, p_r), (0, 0)), mode='constant')
            return cv2.resize(cropped, resize, interpolation=cv2.INTER_LINEAR), (nx1, ny1, nx2, ny2)

        cap = cv2.VideoCapture(video_path)
        original_full_frames = []
        processed_frames = []

        ret, first_frame = cap.read()
        if not ret:
            return
        _, (rx1, ry1, rx2, ry2) = crop_square_and_pad(first_frame, target_bbox, r_value)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_full_frames.append(frame_rgb)
            processed, _ = crop_square_and_pad(frame_rgb, target_bbox, r_value)
            processed_frames.append(processed)
        cap.release()

        if not processed_frames:
            return

        while len(processed_frames) < 30:
            processed_frames.append(processed_frames[-1])
            original_full_frames.append(original_full_frames[-1])

        full_video_tensor = frames_to_video_tensor(processed_frames)
        orig_h, orig_w = original_full_frames[0].shape[:2]

        out_path = output_dir / f'final_{os.path.basename(video_path)}'
        os.makedirs(out_path.parent, exist_ok=True)
        out_video = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (orig_w, orig_h))

        v_nx1, v_ny1 = max(0, rx1), max(0, ry1)
        v_nx2, v_ny2 = min(orig_w, rx2), min(orig_h, ry2)

        for i in range(min(29, len(original_full_frames))):
            final_frame = cv2.cvtColor(original_full_frames[i], cv2.COLOR_RGB2BGR)
            cv2.rectangle(final_frame, (v_nx1, v_ny1), (v_nx2, v_ny2), (0, 255, 0), 2)
            out_video.write(final_frame)
        next_frame_to_write = min(29, len(original_full_frames))

        print(f"🔍 배치 슬라이딩 윈도우 추론 및 히트맵 생성 중... (stride={window_stride})")
        num_windows = full_video_tensor.size(1) - 29
        window_starts = list(range(0, num_windows, window_stride))
        with torch.inference_mode():
            for batch_start in range(0, len(window_starts), infer_batch_size):
                batch_window_starts = window_starts[batch_start:batch_start + infer_batch_size]
                clips = torch.stack([full_video_tensor[:, i:i+30, :, :] for i in batch_window_starts])
                clips = clips.to(device, non_blocking=True)
                if device.type == 'cuda':
                    clips = clips.contiguous(memory_format=torch.channels_last_3d)

                outputs = model(clips)
                probs = F.softmax(outputs, dim=1)
                pred_classes = outputs.argmax(dim=1)
                feat_maps = activation['inception5b']

                for offset, window_idx in enumerate(batch_window_starts):
                    frame_idx = window_idx + 29
                    pred_class = int(pred_classes[offset].item())
                    conf = probs[offset, pred_class].item() * 100

                    feat_map = feat_maps[offset]
                    weight = model.head_conv.weight[pred_class]
                    cam = torch.sum(weight * feat_map, dim=0)
                    cam = F.relu(cam)
                    cam_2d = torch.mean(cam, dim=0).cpu().numpy()
                    cam_2d = (cam_2d - cam_2d.min()) / (cam_2d.max() - cam_2d.min() + 1e-8)

                    heatmap = cv2.applyColorMap(np.uint8(255 * cam_2d), cv2.COLORMAP_JET)
                    heatmap = cv2.resize(heatmap, (rx2-rx1, ry2-ry1))
                    heatmap_valid = heatmap[v_ny1-ry1:(v_ny1-ry1)+(v_ny2-v_ny1), v_nx1-rx1:(v_nx1-rx1)+(v_nx2-v_nx1)]

                    for skipped_idx in range(next_frame_to_write, frame_idx):
                        skipped_frame = cv2.cvtColor(original_full_frames[skipped_idx], cv2.COLOR_RGB2BGR)
                        cv2.rectangle(skipped_frame, (v_nx1, v_ny1), (v_nx2, v_ny2), (0, 255, 0), 2)
                        out_video.write(skipped_frame)

                    final_frame = cv2.cvtColor(original_full_frames[frame_idx], cv2.COLOR_RGB2BGR)
                    roi = final_frame[v_ny1:v_ny2, v_nx1:v_nx2]

                    if pred_class == 1:
                        final_frame[v_ny1:v_ny2, v_nx1:v_nx2] = cv2.addWeighted(roi, 0.6, heatmap_valid, 0.4, 0)
                        color, label_text = (0, 0, 255), f"Accident ({conf:.1f}%)"
                    else:
                        color, label_text = (0, 255, 0), f"Normal ({conf:.1f}%)"

                    cv2.rectangle(final_frame, (v_nx1, v_ny1), (v_nx2, v_ny2), color, 3)
                    cv2.putText(final_frame, label_text, (50, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
                    out_video.write(final_frame)
                    next_frame_to_write = frame_idx + 1

        for skipped_idx in range(next_frame_to_write, len(original_full_frames)):
            skipped_frame = cv2.cvtColor(original_full_frames[skipped_idx], cv2.COLOR_RGB2BGR)
            cv2.rectangle(skipped_frame, (v_nx1, v_ny1), (v_nx2, v_ny2), (0, 255, 0), 2)
            out_video.write(skipped_frame)

        out_video.release()
        print(f"✨ 분석 완료! 결과 파일: {out_path}")
    finally:
        handle.remove()

# ==========================================
# 3. [핵심] 가중치 불러오기 및 실행부
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = HitAndRun3DCNN(num_classes=MODEL_NUM_CLASSES).to(device)
if device.type == 'cuda':
    loaded_model = loaded_model.to(memory_format=torch.channels_last_3d)

try:
    loaded_model.load_state_dict(torch.load(PREDICT_WEIGHTS_PATH, map_location=device))
    print("✅ 성공적으로 학습된 가중치(best_hitandrun_model.pth)를 불러왔습니다!")

    predict_hit_and_run_final(
        loaded_model,
        PREDICT_VIDEO_PATH,
        PREDICT_TXT_PATH,
        target_id=TARGET_ID,
        r_value=R_VALUE,
        resize=RESIZE,
        output_dir=PREDICT_OUTPUT_DIR,
        infer_batch_size=PREDICT_INFER_BATCH_SIZE,
        window_stride=PREDICT_WINDOW_STRIDE,
    )

except FileNotFoundError:
    print(f"⚠️ 에러: {PREDICT_WEIGHTS_PATH} 경로에 가중치 파일이 없습니다.")
    print("경로가 정확한지 확인하시거나, 먼저 학습 코드를 실행하여 가중치 파일을 생성해 주세요.")

# ============================================================# [랜덤영상 정확도 예측 전용 셀]

import os
import cv2
import torch
import numpy as np
import torchvision.transforms as transforms
import torch.nn.functional as F
import random
from tqdm import tqdm

# ==========================================
# 1. 평가 전용 함수 정의
# ==========================================
def evaluate_folder_accuracy(model, folder_path, target_id=0, r_value=1.0, resize=(224, 224), num_samples=None, infer_batch_size=32, window_stride=3):
    print(f"📂 [{folder_path}] 폴더의 영상 평가를 준비합니다...")
    device = next(model.parameters()).device
    model.eval()
    window_stride = max(1, int(window_stride))

    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(3, 1, 1, 1)

    def crop_square_and_pad(frame, bbox, r):
        h, w, _ = frame.shape
        x_min, y_min, x_max, y_max = bbox
        vw, vh = (x_max - x_min) * r, (y_max - y_min) * r
        cx, cy = x_min + (x_max - x_min) // 2, y_min + (y_max - y_min) // 2
        side = int(max(vw, vh))
        nx1, ny1 = cx - side // 2, cy - side // 2
        nx2, ny2 = cx + side // 2, cy + side // 2
        v_x1, v_y1 = max(0, nx1), max(0, ny1)
        v_x2, v_y2 = min(w, nx2), min(h, ny2)
        p_l, p_t = max(0, -nx1), max(0, -ny1)
        p_r, p_b = max(0, nx2 - w), max(0, ny2 - h)
        cropped = frame[v_y1:v_y2, v_x1:v_x2]
        if p_l > 0 or p_t > 0 or p_r > 0 or p_b > 0:
            cropped = np.pad(cropped, ((p_t, p_b), (p_l, p_r), (0, 0)), mode='constant')
        return cv2.resize(cropped, resize, interpolation=cv2.INTER_LINEAR)

    def frames_to_tensor(frames):
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        video_tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
        return (video_tensor - mean) / std

    all_files = os.listdir(folder_path)
    all_files_set = set(all_files)
    mp4_files = [f for f in all_files if f.endswith('.mp4')]
    valid_pairs = [mp4 for mp4 in mp4_files if f"{mp4.rsplit('.', 1)[0]}.txt" in all_files_set]

    if not valid_pairs:
        print("⚠️ 평가할 수 있는 영상-텍스트 짝이 없습니다.")
        return

    if num_samples is not None:
        actual_samples = min(num_samples, len(valid_pairs))
        selected_files = random.sample(valid_pairs, actual_samples)
        print(f"🎲 총 {len(valid_pairs)}개의 영상 쌍 중에서 {actual_samples}개를 랜덤으로 추출하여 평가합니다.")
    else:
        selected_files = valid_pairs
        print(f"전체 {len(valid_pairs)}개의 영상 쌍을 모두 평가합니다.")

    total_videos = 0
    correct_preds = 0
    wrong_list = []

    with torch.inference_mode():
        for mp4_file in tqdm(selected_files, desc="평가 진행률"):
            base_name = mp4_file.rsplit('.', 1)[0]
            video_path = os.path.join(folder_path, mp4_file)
            txt_path = os.path.join(folder_path, f"{base_name}.txt")

            parts = base_name.split('_')
            is_accident_gt = len(parts) >= 2 and len(parts[1]) == 2 and parts[1][1] == 'A'
            gt_label = 1 if is_accident_gt else 0

            bboxes = {}
            with open(txt_path, 'r') as f:
                for line in f:
                    l_parts = line.strip().split(',')
                    if len(l_parts) >= 6 and l_parts[0] == 'car':
                        bboxes[int(l_parts[1])] = [int(l_parts[2]), int(l_parts[3]), int(l_parts[4]), int(l_parts[5])]

            if target_id not in bboxes:
                if not bboxes:
                    continue
                target_bbox = next(iter(bboxes.values()))
            else:
                target_bbox = bboxes[target_id]

            cap = cv2.VideoCapture(video_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(crop_square_and_pad(frame_rgb, target_bbox, r_value))
            cap.release()

            while len(frames) < 30:
                frames.append(frames[-1] if frames else np.zeros((resize[1], resize[0], 3), dtype=np.uint8))

            full_video_tensor = frames_to_tensor(frames)
            predicted_label = 0
            num_windows = len(frames) - 29
            window_starts = list(range(0, num_windows, window_stride))

            for batch_start in range(0, len(window_starts), infer_batch_size):
                batch_window_starts = window_starts[batch_start:batch_start + infer_batch_size]
                clips = torch.stack([full_video_tensor[:, i:i+30, :, :] for i in batch_window_starts])
                clips = clips.to(device, non_blocking=True)
                if device.type == 'cuda':
                    clips = clips.contiguous(memory_format=torch.channels_last_3d)

                outputs = model(clips)
                pred_classes = outputs.argmax(dim=1)
                if (pred_classes == 1).any().item():
                    predicted_label = 1
                    break

            total_videos += 1
            if predicted_label == gt_label:
                correct_preds += 1
            else:
                wrong_list.append({
                    "file": mp4_file,
                    "gt": "Accident(충돌)" if gt_label == 1 else "Normal(정상)",
                    "pred": "Accident(충돌)" if predicted_label == 1 else "Normal(정상)",
                })

    accuracy = (correct_preds / total_videos) * 100 if total_videos > 0 else 0
    print("\n" + "="*50)
    print("📊 [모델 성능 평가 결과]")
    print("="*50)
    print(f"총 평가 영상 수 : {total_videos} 개")
    print(f"정답 맞춘 수    : {correct_preds} 개")
    print(f"최종 Accuracy   : {accuracy:.2f}%")
    print("="*50)

    if wrong_list:
        print("\n❌ [오답 노트 (틀린 영상 리스트)]")
        for w in wrong_list:
            print(f" - {w['file']} (실제: {w['gt']}  |  모델예측: {w['pred']})")
    else:
        print("\n🎉 모든 영상을 완벽하게 맞췄습니다!")

# ==========================================
# 3. [핵심] 랜덤영상 정확도 예측 실행부
# ==========================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
loaded_model = HitAndRun3DCNN(num_classes=MODEL_NUM_CLASSES).to(device)
if device.type == 'cuda':
    loaded_model = loaded_model.to(memory_format=torch.channels_last_3d)

try:
    loaded_model.load_state_dict(torch.load(EVAL_WEIGHTS_PATH, map_location=device))
    print("✅ 성공적으로 학습된 모델 가중치를 불러왔습니다!")
    evaluate_folder_accuracy(
        loaded_model,
        EVAL_FOLDER_PATH,
        target_id=TARGET_ID,
        r_value=R_VALUE,
        resize=RESIZE,
        num_samples=EVAL_NUM_SAMPLES,
        infer_batch_size=EVAL_INFER_BATCH_SIZE,
        window_stride=EVAL_WINDOW_STRIDE,
    )

except FileNotFoundError:
    print(f"⚠️ 에러: {EVAL_WEIGHTS_PATH} 경로에 가중치 파일이 없습니다. 먼저 모델을 학습시켜주세요.")

