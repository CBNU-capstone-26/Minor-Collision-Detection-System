import os
import queue
import threading
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path

import config
from device_utils import is_cuda_like, is_channels_last_3d_supported


activation = {}


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def _frames_to_video_tensor(frames):
    mean = torch.tensor([0.485, 0.456, 0.406],
                        dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225],
                       dtype=torch.float32).view(3, 1, 1, 1)
    arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
    tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
    return (tensor - mean) / std


def _crop_square_and_pad(frame, bbox, r, resize):
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
    return cv2.resize(cropped, resize, interpolation=cv2.INTER_LINEAR), (nx1, ny1, nx2, ny2)


def _video_writer_worker(q, out_video):
    """VideoWriter를 별도 스레드에서 실행 — 추론과 디스크 I/O를 병렬화"""
    while True:
        frame = q.get()
        if frame is None:
            break
        out_video.write(frame)


def _draw_state_label(frame, state):
    """좌상단에 현재 클래스 상태(S/A)를 표시한다.

    state=0 → S (정상, 초록)
    state=1 → A (충돌, 빨강)
    """
    label = "A" if state == 1 else "S"
    color = (0, 0, 255) if state == 1 else (0, 255, 0)
    # 배경 사각형으로 가독성 확보
    cv2.rectangle(frame, (10, 10), (180, 90), (0, 0, 0), -1)
    cv2.putText(frame, label, (20, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)


def predict_hit_and_run_final(
    model,
    video_path=config.PREDICT_VIDEO_PATH,
    txt_path=config.PREDICT_TXT_PATH,
    target_id=config.TARGET_ID,
    r_value=config.R_VALUE,
    resize=config.RESIZE,
    clip_length=config.CLIP_LENGTH,
    output_dir=config.PREDICT_OUTPUT_DIR,
    infer_batch_size=config.PREDICT_INFER_BATCH_SIZE,
    window_stride=config.PREDICT_WINDOW_STRIDE,
):
    """단일 영상에 대해 슬라이딩 윈도우 추론 + CAM 합성 영상을 출력한다.

    Returns:
        tuple(Path, list[dict]):
            - 출력 영상 경로
            - 이벤트 목록. 각 항목: {'start_frame': int, 'end_frame': int}
              (한 영상에 복수의 물피도주 이벤트가 존재할 때 모두 반환)
    """
    video_path = str(video_path)
    txt_path = str(txt_path)
    output_dir = Path(output_dir)

    print(f"[{os.path.basename(video_path)}] 전체 화면 분석 시작...")
    device = next(model.parameters()).device
    model.eval()
    window_stride = max(1, int(window_stride))

    if is_cuda_like(device):
        torch.backends.cudnn.benchmark = True

    handle = model.inception5b.register_forward_hook(
        get_activation('inception5b'))
    out_video = None
    write_queue = None
    writer_thread = None

    # 다중 이벤트 추적용 상태 변수
    events = []          # [{'start_frame': int, 'end_frame': int}, ...]
    display_state = 0    # 현재 표시 상태: 0=S(정상), 1=A(충돌)
    prev_state = 0       # 직전 윈도우 예측 (S→A 전환 감지용)
    event_start = None   # 진행 중인 이벤트의 시작 프레임

    try:
        if not os.path.exists(txt_path):
            print(f"텍스트 파일을 찾을 수 없습니다: {txt_path}")
            return None, []

        bboxes = {}
        with open(txt_path, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 6 and parts[0] == 'car':
                    bboxes[int(parts[1])] = [
                        int(parts[2]), int(parts[3]),
                        int(parts[4]), int(parts[5]),
                    ]

        if target_id not in bboxes:
            print(f"ID {target_id}번 차량의 좌표가 없습니다. 존재하는 ID: {list(bboxes.keys())}")
            return None, []

        target_bbox = bboxes[target_id]

        cap = cv2.VideoCapture(video_path)
        original_full_frames = []
        processed_frames = []

        ret, first_frame = cap.read()
        if not ret:
            return None, []
        _, (rx1, ry1, rx2, ry2) = _crop_square_and_pad(
            first_frame, target_bbox, r_value, resize)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            original_full_frames.append(frame_rgb)
            processed, _ = _crop_square_and_pad(
                frame_rgb, target_bbox, r_value, resize)
            processed_frames.append(processed)
        cap.release()

        if not processed_frames:
            return None, []

        while len(processed_frames) < clip_length:
            processed_frames.append(processed_frames[-1])
            original_full_frames.append(original_full_frames[-1])

        full_video_tensor = _frames_to_video_tensor(
            processed_frames).to(device)

        orig_h, orig_w = original_full_frames[0].shape[:2]
        out_path = output_dir / f'final_{os.path.basename(video_path)}'
        os.makedirs(out_path.parent, exist_ok=True)
        out_video = cv2.VideoWriter(
            str(out_path), cv2.VideoWriter_fourcc(*'mp4v'), 30.0, (orig_w, orig_h))

        write_queue = queue.Queue(maxsize=64)
        writer_thread = threading.Thread(
            target=_video_writer_worker,
            args=(write_queue, out_video),
            daemon=True,
        )
        writer_thread.start()

        v_nx1, v_ny1 = max(0, rx1), max(0, ry1)
        v_nx2, v_ny2 = min(orig_w, rx2), min(orig_h, ry2)

        # 첫 (clip_length - 1)개 프레임: 아직 예측 전 → 상태 S로 출력
        for i in range(min(clip_length - 1, len(original_full_frames))):
            f = cv2.cvtColor(original_full_frames[i], cv2.COLOR_RGB2BGR)
            cv2.rectangle(f, (v_nx1, v_ny1), (v_nx2, v_ny2), (0, 255, 0), 2)
            _draw_state_label(f, display_state)
            write_queue.put(f)
        next_frame_to_write = min(clip_length - 1, len(original_full_frames))

        print(f"배치 슬라이딩 윈도우 추론 및 히트맵 생성 중... (stride={window_stride})")
        num_windows = full_video_tensor.size(1) - (clip_length - 1)
        window_starts = list(range(0, num_windows, window_stride))

        with torch.inference_mode():
            for batch_start in range(0, len(window_starts), infer_batch_size):
                batch_window_starts = window_starts[batch_start:batch_start +
                                                    infer_batch_size]

                clips = torch.stack(
                    [full_video_tensor[:, i:i + clip_length, :, :] for i in batch_window_starts])
                if is_channels_last_3d_supported(device):
                    clips = clips.contiguous(
                        memory_format=torch.channels_last_3d)

                outputs = model(clips)
                probs = F.softmax(outputs, dim=1)
                pred_classes = outputs.argmax(dim=1)
                feat_maps = activation['inception5b']

                for offset, window_idx in enumerate(batch_window_starts):
                    # 이 윈도우의 마지막 프레임에 오버레이를 그림
                    frame_idx = window_idx + clip_length - 1
                    pred_class = int(pred_classes[offset].item())
                    conf = probs[offset, pred_class].item() * 100

                    # ── 이벤트 상태 전환 감지 ──────────────────────────────
                    if prev_state == 0 and pred_class == 1:
                        # S → A: 새 충돌 이벤트 시작
                        event_start = window_idx
                    elif prev_state == 1 and pred_class == 0:
                        # A → S: 충돌 이벤트 종료
                        if event_start is not None:
                            events.append({
                                'start_frame': event_start,
                                'end_frame': frame_idx - 1,
                            })
                            event_start = None
                    prev_state = pred_class
                    display_state = pred_class
                    # ────────────────────────────────────────────────────────

                    # ⑤ CAM 연산 GPU 유지
                    feat_map = feat_maps[offset]
                    weight = model.head_conv.weight[pred_class]
                    cam = F.relu(torch.sum(weight * feat_map, dim=0))
                    cam_2d = torch.mean(cam, dim=0)
                    cam_min, cam_max = cam_2d.min(), cam_2d.max()
                    cam_2d = (cam_2d - cam_min) / (cam_max - cam_min + 1e-8)
                    cam_np = (cam_2d * 255).byte().cpu().numpy()

                    heatmap = cv2.applyColorMap(cam_np, cv2.COLORMAP_JET)
                    heatmap = cv2.resize(heatmap, (rx2 - rx1, ry2 - ry1))
                    heatmap_valid = heatmap[
                        v_ny1 - ry1:(v_ny1 - ry1) + (v_ny2 - v_ny1),
                        v_nx1 - rx1:(v_nx1 - rx1) + (v_nx2 - v_nx1),
                    ]

                    # 스킵된 프레임: 직전 display_state 그대로 표시
                    for skipped_idx in range(next_frame_to_write, frame_idx):
                        sf = cv2.cvtColor(
                            original_full_frames[skipped_idx], cv2.COLOR_RGB2BGR)
                        cv2.rectangle(sf, (v_nx1, v_ny1),
                                      (v_nx2, v_ny2), (0, 255, 0), 2)
                        _draw_state_label(sf, display_state)
                        write_queue.put(sf)

                    # 현재 예측 프레임
                    final_frame = cv2.cvtColor(
                        original_full_frames[frame_idx], cv2.COLOR_RGB2BGR)
                    roi = final_frame[v_ny1:v_ny2, v_nx1:v_nx2]

                    if pred_class == 1:
                        final_frame[v_ny1:v_ny2, v_nx1:v_nx2] = cv2.addWeighted(
                            roi, 0.6, heatmap_valid, 0.4, 0)
                        bbox_color = (0, 0, 255)
                        conf_text = f"Accident ({conf:.1f}%)"
                    else:
                        bbox_color = (0, 255, 0)
                        conf_text = f"Normal ({conf:.1f}%)"

                    cv2.rectangle(final_frame, (v_nx1, v_ny1),
                                  (v_nx2, v_ny2), bbox_color, 3)
                    # 신뢰도는 bbox 상단에 작게 표시
                    cv2.putText(final_frame, conf_text,
                                (v_nx1, max(v_ny1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
                    # 좌상단 클래스 상태 (S / A)
                    _draw_state_label(final_frame, pred_class)

                    write_queue.put(final_frame)
                    next_frame_to_write = frame_idx + 1

        # 마지막 예측 이후 남은 프레임: 마지막 display_state 유지
        for skipped_idx in range(next_frame_to_write, len(original_full_frames)):
            sf = cv2.cvtColor(
                original_full_frames[skipped_idx], cv2.COLOR_RGB2BGR)
            cv2.rectangle(sf, (v_nx1, v_ny1), (v_nx2, v_ny2), (0, 255, 0), 2)
            _draw_state_label(sf, display_state)
            write_queue.put(sf)

        # 영상 마지막 시점까지 A 상태가 유지된 경우 이벤트 닫기
        if prev_state == 1 and event_start is not None:
            events.append({
                'start_frame': event_start,
                'end_frame': len(original_full_frames) - 1,
            })

        write_queue.put(None)
        writer_thread.join()
        out_video.release()
        out_video = None

        # 결과 출력
        print(f"\n분석 완료! 결과 파일: {out_path}")
        if events:
            print(f"감지된 충돌 이벤트: {len(events)}건")
            for i, ev in enumerate(events, 1):
                print(f"  이벤트 {i}: 시작 프레임 {ev['start_frame']}, "
                      f"종료 프레임 {ev['end_frame']}")
        else:
            print("감지된 충돌 이벤트 없음 (Normal)")

        return out_path, events

    finally:
        handle.remove()
        if writer_thread is not None and writer_thread.is_alive():
            try:
                write_queue.put_nowait(None)
            except queue.Full:
                pass
            writer_thread.join(timeout=5)
        if out_video is not None:
            out_video.release()
