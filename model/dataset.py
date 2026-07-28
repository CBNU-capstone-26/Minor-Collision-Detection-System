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
    ):
        self.data_dir = data_dir
        self.clip_length = clip_length
        self.r_value = r_value
        self.resize = resize
        self.mean = torch.tensor(
            [0.485, 0.456, 0.406], dtype=torch.float32).view(3, 1, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225],
                                dtype=torch.float32).view(3, 1, 1, 1)

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
        # 조명 환경·날씨·카메라 색감 차이로 인한 도메인 갭을 줄이기 위해
        # 색상 계열 증강을 중심으로 구성. 모든 프레임에 동일한 변환 적용 (시간적 일관성 유지)

        do_hflip = random.random() > 0.5

        # 밝기·대비·채도·색조: 조명 변화, 날씨, 카메라 색감 차이 시뮬레이션
        do_jitter = random.random() > 0.5
        brightness = random.uniform(0.6, 1.4)   # 어두운 지하주차장 ~ 밝은 외부
        contrast = random.uniform(0.7, 1.3)   # 저대비(흐린 날) ~ 고대비(직사광)
        saturation = random.uniform(0.7, 1.3)   # 카메라별 채도 특성 차이
        hue = random.uniform(-0.1, 0.1)  # 카메라 화이트밸런스·조명 색온도 차이

        # 날씨 증강: 전체 학습 클립 중 약 5%에만 적용 (장대비 2.5%, 함박눈 2.5%)
        # 프레임마다 입자 위치를 조금씩 이동시켜 실제 날씨처럼 시간적 움직임을 만든다.
        weather_rand = random.random()
        weather_type = None
        if weather_rand < 0.025:
            weather_type = "rain"
        elif weather_rand < 0.05:
            weather_type = "snow"

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
            aug_frames.append(np.array(img))

        if weather_type == "rain":
            aug_frames = self._apply_rain(aug_frames)
        elif weather_type == "snow":
            aug_frames = self._apply_snow(aug_frames)

        return aug_frames

    def _apply_rain(self, frames):
        if not frames:
            return frames

        h, w = frames[0].shape[:2]
        # 장대비 느낌을 위해 빗줄기 수·길이·두께를 키운다.
        num_drops = max(25, (h * w) // 1600)
        length = random.randint(18, 32)
        speed = random.randint(8, 14)
        slant = random.randint(-6, 6)
        thickness = random.choice([1, 2, 2])
        drops = [
            [random.randint(0, w - 1), random.randint(-h, h - 1)]
            for _ in range(num_drops)
        ]

        rainy_frames = []
        for frame_idx, frame in enumerate(frames):
            out = frame.copy()
            overlay = out.copy()
            for x, y in drops:
                y_shifted = (y + frame_idx * speed) % (h + length) - length
                x_shifted = int((x + frame_idx * slant) % w)
                cv2.line(
                    overlay,
                    (x_shifted, int(y_shifted)),
                    (int(np.clip(x_shifted + slant, 0, w - 1)),
                     int(np.clip(y_shifted + length, 0, h - 1))),
                    (200, 215, 230),
                    thickness,
                    cv2.LINE_AA,
                )
            out = cv2.addWeighted(overlay, 0.52, out, 0.48, 0)
            out = cv2.convertScaleAbs(out, alpha=0.86, beta=-8)
            rainy_frames.append(out)
        return rainy_frames

    def _apply_snow(self, frames):
        if not frames:
            return frames

        h, w = frames[0].shape[:2]
        # 함박눈 느낌을 위해 눈송이 수와 크기를 키운다.
        num_flakes = max(20, (h * w) // 2000)
        speed = random.uniform(1.5, 3.2)
        wind = random.uniform(-1.4, 1.4)
        flakes = [
            [random.uniform(0, w - 1), random.uniform(-h, h - 1),
             random.choice([2, 2, 3, 3, 4])]
            for _ in range(num_flakes)
        ]

        snowy_frames = []
        for frame_idx, frame in enumerate(frames):
            out = frame.copy()
            overlay = out.copy()
            for x, y, radius in flakes:
                y_shifted = (y + frame_idx * speed) % (h + radius) - radius
                x_shifted = (x + frame_idx * wind) % w
                cv2.circle(
                    overlay,
                    (int(x_shifted), int(y_shifted)),
                    int(radius),
                    (245, 245, 245),
                    -1,
                    cv2.LINE_AA,
                )
            out = cv2.addWeighted(overlay, 0.48, out, 0.52, 0)
            snowy_frames.append(out)
        return snowy_frames

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
            frames.append(self._crop_and_pad(
                frame, sample['bbox'], self.r_value))
        cap.release()

        while len(frames) < self.clip_length:
            frames.append(
                frames[-1] if frames else np.zeros((self.resize[1], self.resize[0], 3), dtype=np.uint8))

        frames = self._apply_augmentation(frames)

        return self._frames_to_tensor(frames), torch.tensor(sample['label'], dtype=torch.long)
