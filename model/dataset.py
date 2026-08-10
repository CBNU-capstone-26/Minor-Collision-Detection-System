import os
import cv2
import numpy as np
import torch
import random
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image

import config


class HitAndRunDataset(Dataset):
    def __init__(
        self,
        data_dir=config.DATA_DIR,
        clip_length=config.CLIP_LENGTH,
        r_value=config.R_VALUE,
        resize=config.RESIZE,
        augment=True,
    ):
        self.data_dir = data_dir
        self.clip_length = clip_length
        self.r_value = r_value
        self.resize = resize
        # augment=True → 학습용(랜덤 증강 적용), False → 검증용(증강 없음, 결정적)
        self.augment = augment
        # 정규화 통계는 사전학습(S3D Kinetics-400)과 동일하게 config에서 일괄 관리
        self.mean = torch.tensor(
            config.NORM_MEAN, dtype=torch.float32).view(3, 1, 1, 1)
        self.std = torch.tensor(
            config.NORM_STD, dtype=torch.float32).view(3, 1, 1, 1)

        self.file_names = sorted(f.rsplit('.', 1)[0] for f in os.listdir(
            data_dir) if f.endswith('.mp4'))
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
            target_bbox = bboxes.get(
                target_id, [0, 0, self.resize[0], self.resize[1]])
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
                        bboxes[int(parts[1])] = [int(parts[2]), int(
                            parts[3]), int(parts[4]), int(parts[5])]
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

        pad_left = max(0, -new_x_min)
        pad_top = max(0, -new_y_min)
        pad_right = max(0, new_x_max - w)
        pad_bottom = max(0, new_y_max - h)

        valid_x_min = max(0, new_x_min)
        valid_y_min = max(0, new_y_min)
        valid_x_max = min(w, new_x_max)
        valid_y_max = min(h, new_y_max)
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
        # 실제 주차장 CCTV 환경 대응 데이터 증강
        # 변환 파라미터는 클립당 1회 샘플링해 모든 프레임에 동일 적용 (시간적 일관성 유지)
        # ※ 센서 노이즈만 프레임별로 새로 샘플링 (실제 노이즈는 프레임마다 다름)

        do_hflip = random.random() > 0.5

        # 밝기·대비·채도·색조: 조명 변화, 날씨, 카메라 색감 차이 시뮬레이션
        do_jitter = random.random() > 0.5
        brightness = random.uniform(0.6, 1.4)   # 어두운 지하주차장 ~ 밝은 외부
        contrast = random.uniform(0.7, 1.3)   # 저대비(흐린 날) ~ 고대비(직사광)
        saturation = random.uniform(0.7, 1.3)   # 카메라별 채도 특성 차이
        hue = random.uniform(-0.1, 0.1)  # 카메라 화이트밸런스·조명 색온도 차이

        # 야간 적외선(IR) CCTV: 사실상 무채색 영상 → 색 없이도 동작 판별하도록 학습
        do_gray = random.random() < 0.15
        # 저화질 CCTV: 초점 흐림/압축 열화(블러), 센서 노이즈(야간 게인↑) 시뮬레이션
        do_blur = random.random() < 0.2
        do_noise = random.random() < 0.2
        noise_sigma = random.uniform(2.0, 8.0)

        aug_frames = []
        for frame in frames:
            img = Image.fromarray(frame)
            if do_hflip:
                img = TF.hflip(img)
            if do_jitter:
                img = TF.adjust_brightness(img, brightness)
                img = TF.adjust_contrast(img, contrast)
                img = TF.adjust_saturation(img, saturation)
                img = TF.adjust_hue(img, hue)
            if do_gray:
                img = TF.rgb_to_grayscale(img, num_output_channels=3)
            if do_blur:
                img = TF.gaussian_blur(img, kernel_size=3)
            arr = np.array(img)
            if do_noise:
                noise = np.random.normal(0.0, noise_sigma, arr.shape)
                arr = np.clip(arr.astype(np.float32) + noise,
                              0, 255).astype(np.uint8)
            aug_frames.append(arr)

        return aug_frames

    def _frames_to_tensor(self, frames):
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        video_tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
        return (video_tensor - self.mean) / self.std

    def __getitem__(self, idx):
        sample = self.samples[idx]
        start_f = sample['start_f']
        bbox = sample['bbox']

        if self.augment:
            # ① 시간 오프셋 지터: 학습 클립이 항상 '충돌 시작 = 윈도 첫 프레임'이면
            #    추론(슬라이딩 윈도)에서 충돌이 윈도 중간에 걸릴 때 분포가 어긋난다.
            #    시작점을 0~10프레임 앞으로 당겨 충돌 위치를 윈도 내에서 다양화.
            start_f = max(0, start_f - random.randint(0, 10))
            # ② bbox 지터: 서비스에서는 사용자가 마우스로 대충 박스를 그린다.
            #    GT 좌표 그대로만 학습하면 손그림 박스와 분포가 어긋나므로
            #    중심 이동(±5%)·크기 배율(0.9~1.15)로 부정확한 박스를 시뮬레이션.
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            cx = (x1 + x2) // 2 + int(bw * random.uniform(-0.05, 0.05))
            cy = (y1 + y2) // 2 + int(bh * random.uniform(-0.05, 0.05))
            s = random.uniform(0.9, 1.15)
            hw, hh = int(bw * s / 2), int(bh * s / 2)
            bbox = [cx - hw, cy - hh, cx + hw, cy + hh]

        cap = cv2.VideoCapture(sample['mp4_path'])
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)

        for _ in range(self.clip_length):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self._crop_and_pad(
                frame, bbox, self.r_value))
        cap.release()

        while len(frames) < self.clip_length:
            frames.append(
                frames[-1] if frames else np.zeros((self.resize[1], self.resize[0], 3), dtype=np.uint8))

        # 증강은 학습 세트에만 적용 (검증 세트는 결정적이어야 val loss가 안정됨)
        if self.augment:
            frames = self._apply_augmentation(frames)

        return self._frames_to_tensor(frames), torch.tensor(sample['label'], dtype=torch.long)
