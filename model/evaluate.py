import os
import cv2
import torch
import numpy as np
import random
from tqdm import tqdm

import config
from device_utils import is_cuda_like, is_channels_last_3d_supported


def evaluate_folder_accuracy(
    model,
    folder_path=config.EVAL_FOLDER_PATH,
    target_id=config.TARGET_ID,
    r_value=config.R_VALUE,
    resize=config.RESIZE,
    clip_length=config.CLIP_LENGTH,
    num_samples=config.EVAL_NUM_SAMPLES,
    infer_batch_size=config.EVAL_INFER_BATCH_SIZE,
    window_stride=config.EVAL_WINDOW_STRIDE,
):
    folder_path = str(folder_path)
    print(f"[{folder_path}] 폴더의 영상 평가를 준비합니다...")
    device = next(model.parameters()).device
    model.eval()
    window_stride = max(1, int(window_stride))

    if is_cuda_like(device):
        torch.backends.cudnn.benchmark = True

    mean = torch.tensor([0.485, 0.456, 0.406],
                        dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225],
                       dtype=torch.float32).view(3, 1, 1, 1)

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
            cropped = np.pad(
                cropped, ((p_t, p_b), (p_l, p_r), (0, 0)), mode='constant')
        return cv2.resize(cropped, resize, interpolation=cv2.INTER_LINEAR)

    def frames_to_tensor(frames):
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        video_tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
        return (video_tensor - mean) / std

    all_files = os.listdir(folder_path)
    all_files_set = set(all_files)
    mp4_files = [f for f in all_files if f.endswith('.mp4')]
    valid_pairs = [
        mp4 for mp4 in mp4_files if f"{mp4.rsplit('.', 1)[0]}.txt" in all_files_set]

    if not valid_pairs:
        print("평가할 수 있는 영상-텍스트 짝이 없습니다.")
        return

    if num_samples is not None:
        actual_samples = min(num_samples, len(valid_pairs))
        selected_files = random.sample(valid_pairs, actual_samples)
        print(f"총 {len(valid_pairs)}개의 영상 쌍 중에서 {actual_samples}개를 랜덤으로 추출하여 평가합니다.")
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
            is_accident_gt = len(parts) >= 2 and len(
                parts[1]) == 2 and parts[1][1] == 'A'
            gt_label = 1 if is_accident_gt else 0

            bboxes = {}
            with open(txt_path, 'r') as f:
                for line in f:
                    l_parts = line.strip().split(',')
                    if len(l_parts) >= 6 and l_parts[0] == 'car':
                        bboxes[int(l_parts[1])] = [int(l_parts[2]), int(
                            l_parts[3]), int(l_parts[4]), int(l_parts[5])]

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
                frames.append(crop_square_and_pad(
                    frame_rgb, target_bbox, r_value))
            cap.release()

            while len(frames) < clip_length:
                frames.append(
                    frames[-1] if frames else np.zeros((resize[1], resize[0], 3), dtype=np.uint8))

            # ⑦ 영상 텐서를 처음부터 GPU에 상주 — 배치마다 .to(device) 전송 제거
            full_video_tensor = frames_to_tensor(frames).to(device)
            predicted_label = 0
            num_windows = len(frames) - (clip_length - 1)
            window_starts = list(range(0, num_windows, window_stride))

            for batch_start in range(0, len(window_starts), infer_batch_size):
                batch_window_starts = window_starts[batch_start:batch_start +
                                                    infer_batch_size]
                # 이미 GPU에 있는 텐서 슬라이싱 — .to() 호출 없음
                clips = torch.stack(
                    [full_video_tensor[:, i:i+clip_length, :, :] for i in batch_window_starts])
                if is_channels_last_3d_supported(device):
                    clips = clips.contiguous(
                        memory_format=torch.channels_last_3d)

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
    print("\n" + "=" * 50)
    print("[모델 성능 평가 결과]")
    print("=" * 50)
    print(f"총 평가 영상 수 : {total_videos} 개")
    print(f"정답 맞춘 수    : {correct_preds} 개")
    print(f"최종 Accuracy   : {accuracy:.2f}%")
    print("=" * 50)

    if wrong_list:
        print("\n[오답 노트 (틀린 영상 리스트)]")
        for w in wrong_list:
            print(f" - {w['file']} (실제: {w['gt']}  |  모델예측: {w['pred']})")
    else:
        print("\n모든 영상을 완벽하게 맞췄습니다!")
