import os
import queue
import shutil
import subprocess
import threading
import time
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path

import config
from device_utils import is_cuda_like, is_channels_last_3d_supported, predict_batch_size
from infer_runtime import get_runner, is_accident

# 시스템 ffmpeg(H.264, libx264) 경로입니다. 있으면 클립을 어디서든 재생되는 mp4로 만듭니다.
_FFMPEG = shutil.which("ffmpeg")
# 시스템 ffmpeg가 없으면(macOS, Windows 기본 상태) imageio-ffmpeg에 들어 있는 ffmpeg를 씁니다.
# 클립 변환에 ffmpeg가 없으면 분석 전체가 실패하므로 requirements에 넣어 둡니다. 이 ffmpeg도 libx264를 포함합니다.
if not _FFMPEG:
    try:
        import imageio_ffmpeg
        _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        pass

activation = {}

# 진행률(0.0~1.0)은 단계마다 실제 진행 상황으로 보고해, 화면이 추정값을 쓰다 실제값으로 바뀌며 막대가 뒤로 가는 일을 막습니다.
# 남은 시간(초)은 실제로 잰 속도(묶음 하나 추론 시간, 클립 프레임 처리 시간)로 계산하고,
# 진행률은 경과 / (경과 + 남은 시간)으로 냅니다. 단계마다 고정 몫을 주면 실제 시간과 크게 어긋나기 때문입니다.
# 추론 속도를 재기 전(영상 읽기, 광학흐름)에는 아래 작은 고정값만 씁니다.
PROG_DECODE_END = 0.02    # 프레임 디코딩과 크롭
PROG_FLOW_END = 0.03      # 광학흐름(충격 시점 계산용)
# 3D-CNN이 끝나는 진행률의 잠정값입니다. 훑어보기가 끝나 의심 구간을 알면, 남은 막대를 '자세히 보기'와
# '사고 클립 만들기'에 걸릴 시간 비율로 다시 나눕니다(클립 만들기가 99% 근처에 몰려 보이지 않게).
PROG_INFER_END = 0.90
# 클립 렌더링 속도를 재기 전에 쓰는 추정값입니다. 프레임당 시간을 디코딩과 크롭 시간의 이 배수로 봅니다
# (원본 해상도 디코딩, 히트맵 합성, 인코딩, H.264 변환).
RENDER_COST_PER_DECODE = 8.0
# 사고 구간 CAM 클립 렌더링


def _open_clip_writer(dest_stem, fps, size):
    """클립을 mp4v 임시본에 그릴 VideoWriter를 엽니다. (writer, rendered_path)를 돌려줍니다.
    브라우저 호환 변환(H.264, mp4)은 _finalize_clip이 맡습니다.
    """
    rendered = f"{dest_stem}.render.mp4"
    writer = cv2.VideoWriter(rendered, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    return writer, rendered


def _finalize_clip(rendered_path, dest_stem):
    """mp4v 임시본을 H.264 mp4(yuv420p, faststart)로 다시 인코딩해 최종 경로(str)를 돌려줍니다.

    ffmpeg(libx264)가 꼭 필요합니다(시스템 것을 먼저 쓰고, 없으면 imageio-ffmpeg). H.264 mp4로 통일해야
    브라우저와 OS(크롬, 파이어폭스, 사파리, iOS)를 가리지 않고 재생됩니다. ffmpeg가 없으면 분명한 오류를 냅니다.
    """
    if not _FFMPEG:
        try:
            os.remove(rendered_path)
        except OSError:
            pass
        raise RuntimeError(
            "ffmpeg가 설치되어 있지 않습니다 — 브라우저 호환 클립(H.264) 생성에 필요합니다. "
            "pip install imageio-ffmpeg 로 설치하거나(requirements 포함), "
            "시스템에 설치하세요: sudo apt install -y ffmpeg (또는 brew install ffmpeg)")
    final = f"{dest_stem}.mp4"
    subprocess.run(
        [_FFMPEG, "-y", "-loglevel", "error", "-i", rendered_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", final],
        check=True)
    os.remove(rendered_path)
    return final


def make_activation_hook(name):
    """forward hook입니다. 그 층의 출력(특징맵)을 activation[name]에 담습니다."""
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def _frames_to_video_tensor(frames):
    # 정규화 통계는 학습과 같아야 하므로 config에서 함께 관리합니다(Kinetics-400).
    mean = torch.tensor(config.NORM_MEAN,
                        dtype=torch.float32).view(3, 1, 1, 1)
    std = torch.tensor(config.NORM_STD,
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
    """VideoWriter를 별도 스레드에서 돌려 추론과 디스크 쓰기를 병렬로 합니다."""
    while True:
        frame = q.get()
        if frame is None:
            break
        out_video.write(frame)


def _state_bar_height(frame_h):
    """영상 아래에 붙일 상태 띠 높이입니다(영상 높이의 약 6%, 최소 36px).
    H.264(yuv420p) 변환은 가로와 세로가 짝수여야 하므로 전체 높이가 짝수가 되게 맞춥니다.
    """
    bar_h = max(36, int(round(frame_h * 0.06)))
    if (frame_h + bar_h) % 2:
        bar_h += 1
    return bar_h


def _add_state_bar(frame, state):
    """프레임 아래에 상태 띠(S/A 표시)를 붙인 새 프레임을 돌려줍니다.

    띠를 영상 밖에 붙여 영상 내용을 가리지 않습니다.
    state=0이면 S NORMAL(초록), state=1이면 A ACCIDENT(빨강)입니다.
    OpenCV 글꼴은 한글을 그리지 못해 영어로 씁니다(서버에 한글 글꼴이 없어도 되게).
    """
    h, w = frame.shape[:2]
    bar_h = _state_bar_height(h)
    bar = np.full((bar_h, w, 3), (24, 24, 24), dtype=np.uint8)
    color = (40, 40, 220) if state == 1 else (60, 170, 60)       # BGR: 빨강, 초록
    label, word = ("A", "ACCIDENT") if state == 1 else ("S", "NORMAL")
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = max(4, bar_h // 6)
    thick = max(1, bar_h // 20)
    # 왼쪽: 색 배지와 흰 글자(A/S)
    side = bar_h - 2 * pad
    x0 = 2 * pad
    cv2.rectangle(bar, (x0, pad), (x0 + side, pad + side), color, -1)
    scale = (side * 0.62) / cv2.getTextSize(label, font, 1.0, thick)[0][1]
    (tw, th), _ = cv2.getTextSize(label, font, scale, thick + 1)
    cv2.putText(bar, label, (x0 + (side - tw) // 2, pad + (side + th) // 2),
                font, scale, (255, 255, 255), thick + 1, cv2.LINE_AA)
    # 오른쪽: 상태 단어
    wscale = (bar_h * 0.40) / cv2.getTextSize(word, font, 1.0, thick)[0][1]
    (_, wh), _ = cv2.getTextSize(word, font, wscale, thick)
    cv2.putText(bar, word, (x0 + side + 2 * pad, (bar_h + wh) // 2),
                font, wscale, color, thick, cv2.LINE_AA)
    return np.vstack([frame, bar])


def _draw_collision_cam(frame, cam, square, box, min_rel=config.PREDICT_CLIP_CAM_MIN,
                        alpha=config.PREDICT_CLIP_CAM_ALPHA, gamma=config.PREDICT_CLIP_CAM_GAMMA):
    """사고 판정 근거가 강한 곳(충돌 부위)에만 CAM 히트맵을 칠합니다(frame을 바로 고칩니다).

    cam: 정사각형 크롭의 7x7 칸별 'A - S' 로짓입니다. 칸 평균이 곧 판정 로짓 차이라, 클수록 그 칸이 사고 쪽 근거입니다.
    square: 정사각형 크롭의 원본 좌표 (x1, y1, x2, y2)이며 영상 밖으로 나간 부분도 포함합니다.
    box: 칠할 박스의 원본 좌표 (x1, y1, x2, y2)이며 영상 안으로 자른 것입니다.
    박스 안에서 가장 강한 값의 min_rel배 이상인 곳만, 강할수록 진하게(최대 alpha, 빠르기 gamma) 칠하고 나머지는 원본 그대로 둡니다.
    색은 바깥부터 하늘색, 초록, 노랑, 가운데 빨강인 고리입니다(충돌 지점에 작은 무지개 고리).
    JET의 아주 파란 끝(0~0.2)은 쓰지 않습니다. 가장자리가 짙은 파랑이면 칠한 곳이 멍처럼 보이기 때문입니다.
    """
    sx1, sy1, sx2, sy2 = square
    bx1, by1, bx2, by2 = box
    if bx2 <= bx1 or by2 <= by1:
        return
    big = cv2.resize(np.asarray(cam, dtype=np.float32), (sx2 - sx1, sy2 - sy1), interpolation=cv2.INTER_CUBIC)
    # 박스 부분만 씁니다. 반올림으로 정사각형보다 1px 넘칠 수 있어 자른 뒤 박스 크기에 맞춥니다.
    part = big[max(0, by1 - sy1):max(0, by2 - sy1), max(0, bx1 - sx1):max(0, bx2 - sx1)]
    if part.size == 0:
        return
    part = cv2.resize(part, (bx2 - bx1, by2 - by1))
    peak = float(part.max())
    if peak <= 0:
        return                      # 박스 안에 사고 쪽 근거가 없으면 칠하지 않습니다.
    t = np.clip((part / peak - min_rel) / (1 - min_rel), 0, 1)     # 기준선은 0, 가장 강한 곳은 1
    a = (t ** gamma * alpha)[..., None]
    color = cv2.applyColorMap(((0.2 + 0.8 * t) * 255).astype(np.uint8), cv2.COLORMAP_JET)
    roi = frame[by1:by2, bx1:bx2]
    frame[by1:by2, bx1:bx2] = (roi * (1 - a) + color * a).astype(np.uint8)


def predict_full_video(
    model,
    video_path=config.PREDICT_VIDEO_PATH,
    txt_path=config.PREDICT_TXT_PATH,
    target_id=config.TARGET_ID,
    r_value=config.R_VALUE,
    resize=config.RESIZE,
    clip_length=config.CLIP_LENGTH,
    output_dir=config.PREDICT_OUTPUT_DIR,
    infer_batch_size=None,
    window_stride=config.PREDICT_WINDOW_STRIDE,
    display_offset=config.PREDICT_DISPLAY_OFFSET,
):
    """영상 하나에 슬라이딩 윈도우 추론을 하고 CAM을 합성한 영상을 출력합니다.

    Returns:
        tuple(Path, list[dict]):
            출력 영상 경로
            이벤트 목록. 각 항목은 {'start_frame': int, 'end_frame': int}이며, 이벤트가 여러 개면 모두 돌려줍니다.
    """
    video_path = str(video_path)
    txt_path = str(txt_path)
    output_dir = Path(output_dir)

    print(f"[{os.path.basename(video_path)}] 전체 화면 분석 시작...")
    device = next(model.parameters()).device
    # 배치 크기입니다. 주지 않으면 장치에 맞는 값(CPU 1, GPU 8, config)을 씁니다.
    infer_batch_size = infer_batch_size or predict_batch_size(device)
    model.eval()
    window_stride = max(1, int(window_stride))

    if is_cuda_like(device):
        torch.backends.cudnn.benchmark = True

    handle = model.cam_layer.register_forward_hook(
        make_activation_hook('cam_layer'))
    out_video = None
    write_queue = None
    writer_thread = None

    # 이벤트 여러 개를 추적하는 상태 변수
    events = []          # [{'start_frame': int, 'end_frame': int}, ...]
    display_state = 0    # 현재 표시 상태: 0은 S(정상), 1은 A(충돌)
    prev_state = 0       # 직전 묶음 예측(S에서 A로 바뀌는 것을 감지)
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
        out_stem = str(out_path.with_suffix(""))
        out_video, out_rendered = _open_clip_writer(
            out_stem, 30.0, (orig_w, orig_h + _state_bar_height(orig_h)))

        write_queue = queue.Queue(maxsize=64)
        writer_thread = threading.Thread(
            target=_video_writer_worker,
            args=(write_queue, out_video),
            daemon=True,
        )
        writer_thread.start()

        # CAM 오버레이와 박스를 정사각형 크롭이 아니라 원본 박스 크기로 그립니다.
        # 변수 이름 v_n*는 그대로 두고 좌표를 원본 박스 기준으로 정해, 이후 그리기와 히트맵 자르기가 모두 원본 박스 영역을 씁니다.
        v_nx1, v_ny1 = max(0, target_bbox[0]), max(0, target_bbox[1])
        v_nx2, v_ny2 = min(orig_w, target_bbox[2]), min(orig_h, target_bbox[3])

        # 묶음 판정은 '묶음 시작 + display_offset' 프레임에 그립니다(config 주석 참고).
        display_offset = max(0, min(int(display_offset), clip_length - 1))
        # 처음 display_offset개 프레임은 아직 판정을 그릴 묶음이 없어 S 상태로 출력합니다.
        for i in range(min(display_offset, len(original_full_frames))):
            f = cv2.cvtColor(original_full_frames[i], cv2.COLOR_RGB2BGR)
            cv2.rectangle(f, (v_nx1, v_ny1), (v_nx2, v_ny2), (0, 255, 0), 2)
            write_queue.put(_add_state_bar(f, display_state))
        next_frame_to_write = min(display_offset, len(original_full_frames))

        print(f"배치 슬라이딩 윈도우 추론 및 히트맵 생성 중... (stride={window_stride})")
        num_windows = full_video_tensor.size(1) - (clip_length - 1)
        window_starts = list(range(0, num_windows, window_stride))

        with torch.inference_mode():
            for batch_start in range(0, len(window_starts), infer_batch_size):
                batch_window_starts = window_starts[batch_start:batch_start +
                                                    infer_batch_size]

                clips = torch.stack(
                    [full_video_tensor[:, i:i + clip_length, :, :] for i in batch_window_starts])
                if is_channels_last_3d_supported(device) and config.USE_CHANNELS_LAST:
                    clips = clips.contiguous(
                        memory_format=torch.channels_last_3d)

                outputs = model(clips)
                probs = F.softmax(outputs, dim=1)
                pred_classes = is_accident(outputs).long()     # P(A) > config.A_THRESHOLD
                feat_maps = activation['cam_layer']

                for offset, window_idx in enumerate(batch_window_starts):
                    # 이벤트 경계는 묶음 끝(frame_idx) 기준이고, 판정과 CAM 표시는 disp_idx에 그립니다.
                    frame_idx = window_idx + clip_length - 1
                    disp_idx = window_idx + display_offset
                    pred_class = int(pred_classes[offset].item())
                    conf = probs[offset, pred_class].item() * 100

                    # 이벤트 상태 전환 감지
                    if prev_state == 0 and pred_class == 1:
                        # S에서 A: 새 충돌 이벤트 시작
                        event_start = window_idx
                    elif prev_state == 1 and pred_class == 0:
                        # A에서 S: 충돌 이벤트 종료
                        if event_start is not None:
                            events.append({
                                'start_frame': event_start,
                                'end_frame': frame_idx - 1,
                            })
                            event_start = None
                    prev_state = pred_class
                    display_state = pred_class

                    # CAM 계산은 장치(GPU)에서 그대로 합니다.
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

                    # 건너뛴 프레임은 직전 display_state를 그대로 표시합니다.
                    for skipped_idx in range(next_frame_to_write, disp_idx):
                        sf = cv2.cvtColor(
                            original_full_frames[skipped_idx], cv2.COLOR_RGB2BGR)
                        cv2.rectangle(sf, (v_nx1, v_ny1),
                                      (v_nx2, v_ny2), (0, 255, 0), 2)
                        write_queue.put(_add_state_bar(sf, display_state))

                    # 현재 예측 프레임
                    final_frame = cv2.cvtColor(
                        original_full_frames[disp_idx], cv2.COLOR_RGB2BGR)
                    roi = final_frame[v_ny1:v_ny2, v_nx1:v_nx2]

                    if pred_class == 1:
                        # 반올림으로 생긴 1px 차이를 흡수하도록 히트맵을 박스 영역 크기에 맞춥니다.
                        hm = cv2.resize(heatmap_valid, (roi.shape[1], roi.shape[0]))
                        final_frame[v_ny1:v_ny2, v_nx1:v_nx2] = cv2.addWeighted(
                            roi, 0.6, hm, 0.4, 0)
                        bbox_color = (0, 0, 255)
                        conf_text = f"Accident ({conf:.1f}%)"
                    else:
                        bbox_color = (0, 255, 0)
                        conf_text = f"Normal ({conf:.1f}%)"

                    cv2.rectangle(final_frame, (v_nx1, v_ny1),
                                  (v_nx2, v_ny2), bbox_color, 3)
                    # 신뢰도는 박스 위쪽에 작게 표시합니다.
                    cv2.putText(final_frame, conf_text,
                                (v_nx1, max(v_ny1 - 10, 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, bbox_color, 2)
                    # 클래스 상태(S/A)는 영상 아래 띠에 표시합니다.
                    write_queue.put(_add_state_bar(final_frame, pred_class))
                    next_frame_to_write = disp_idx + 1

        # 마지막 예측 이후 남은 프레임은 마지막 display_state를 유지합니다.
        for skipped_idx in range(next_frame_to_write, len(original_full_frames)):
            sf = cv2.cvtColor(
                original_full_frames[skipped_idx], cv2.COLOR_RGB2BGR)
            cv2.rectangle(sf, (v_nx1, v_ny1), (v_nx2, v_ny2), (0, 255, 0), 2)
            write_queue.put(_add_state_bar(sf, display_state))

        # 영상 마지막까지 A 상태가 이어진 경우 이벤트를 닫습니다.
        if prev_state == 1 and event_start is not None:
            events.append({
                'start_frame': event_start,
                'end_frame': len(original_full_frames) - 1,
            })

        write_queue.put(None)
        writer_thread.join()
        out_video.release()
        out_video = None
        # 브라우저 호환 최종본(H.264 mp4)으로 변환합니다.
        out_path = _finalize_clip(out_rendered, out_stem)

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


def fps_scaled_event_params(fps, coarse_step=None, min_span=None, clip_length=None, enabled=None):
    """거친 탐색 간격과 이벤트 최소 길이를 영상 fps에 맞춥니다(config.PREDICT_FPS_SCALED_POSTPROC).

    두 값은 30fps(config.PREDICT_POSTPROC_REF_FPS) 기준 프레임 수입니다. 최소 길이는 (묶음 길이 - 2) + 연속 A 묶음 수인데,
    묶음 길이 몫은 묶음이 늘 clip_length 프레임이라 그대로 두고 연속 묶음 수만 fps 비율로 바꿉니다(1 이하면 필터를 끈 값이라 그대로).
    서비스와 postproc_check가 함께 씁니다. (거친 탐색 간격, 최소 길이) 프레임을 돌려줍니다.
    """
    coarse_step = config.PREDICT_COARSE_STEP if coarse_step is None else coarse_step
    min_span = config.PREDICT_MIN_EVENT_SPAN_FRAMES if min_span is None else min_span
    clip_length = config.CLIP_LENGTH if clip_length is None else clip_length
    enabled = config.PREDICT_FPS_SCALED_POSTPROC if enabled is None else enabled
    if not enabled or not fps or fps <= 0:
        return int(coarse_step), int(min_span)
    s = float(fps) / float(config.PREDICT_POSTPROC_REF_FPS)
    coarse_step = max(1, int(round(coarse_step * s)))
    base = clip_length - 2
    run = min_span - base
    if run > 1:
        min_span = base + max(1, int(round(run * s)))
    return coarse_step, int(min_span)


def predict_events_and_clips(
    model,
    video_path,
    bbox,
    output_dir,
    r_value=config.R_VALUE,
    resize=config.RESIZE,
    clip_length=config.CLIP_LENGTH,
    infer_batch_size=None,
    window_stride=config.PREDICT_WINDOW_STRIDE,
    display_offset=config.PREDICT_DISPLAY_OFFSET,
    display_start_offset=config.PREDICT_DISPLAY_START_OFFSET,
    display_end_offset=config.PREDICT_DISPLAY_END_OFFSET,
    cam_overlay=config.PREDICT_CLIP_CAM_OVERLAY,
    clip_pad_frames=config.PREDICT_CLIP_PAD_FRAMES,
    progress_callback=None,
    use_coarse_scan=config.PREDICT_USE_COARSE_SCAN,
    coarse_step=config.PREDICT_COARSE_STEP,
    coarse_prob=config.PREDICT_COARSE_PROB,
    coarse_pad=config.PREDICT_COARSE_PAD,
    render_clips=True,
    stats=None,
    clip_name=None,
):
    """웹 서비스용입니다. 대상 차량(bbox) 하나에 대해 사고 의심 구간만 찾고, 그 구간만 짧은 CAM 오버레이 클립으로 만듭니다.
    전체 길이 결과 영상을 다시 만들지 않으므로 몇 시간짜리 영상에도 효율적입니다.

    Args:
        model: 불러온 HitAndRun3DCNN
        video_path (str|Path): 원본 영상 경로
        bbox (tuple|list): 대상 차량 좌표 (xmin, ymin, xmax, ymax), 원본 해상도 기준
        output_dir (str|Path): 클립 저장 폴더
        clip_pad_frames (int): 구간 앞뒤로 덧붙일 여유 프레임 수
        display_start_offset, display_end_offset (int): 클립 A 띠는 첫 A 묶음 시작 + start부터 마지막 A 묶음 시작 + end까지입니다.
            띠가 config.PREDICT_DISPLAY_MIN_SEC보다 짧으면(짧은 반응) 양 끝 모두 display_offset을 씁니다(config 주석 참고).
        use_coarse_scan (bool): True면 거친 탐색(coarse_step 간격) 뒤 P(A) > coarse_prob인 묶음 주변
            coarse_pad 안만 촘촘히 봅니다. False면 전체를 봅니다.
        clip_name (str|None): 클립 파일 이름 앞부분('<clip_name>_event1.mp4')입니다. 없으면 영상 파일 이름을 씁니다.
            서비스는 분석마다 다른 이름을 주어, 같은 영상을 다시 분석해도 이전 클립을 덮어쓰지 않습니다.
        render_clips (bool): False면 CAM 클립을 만들지 않고 이벤트 목록만 돌려줍니다(clip_path=None).
            평가에서 서비스와 같은 경로로 이벤트를 뽑을 때 씁니다. 렌더링은 평가에 필요 없고 가장 느린 단계입니다.
        stats (dict|None): 주면 단계별 처리 시간(초)과 묶음 수를 채워 줍니다(평가의 속도 측정용).
            키: frames, decode_sec, flow_sec, infer_sec, render_sec, windows_total, windows_eval

    Returns:
        list[dict]: 이벤트 목록. 각 항목은 다음과 같습니다.
            {
                'start_frame', 'end_frame',
                'start_sec', 'end_sec',
                'crash_prob',        # 구간 대표 확률(0~1)
                'clip_path',         # 만든 CAM 클립 경로(Path)
            }
    """
    video_path = str(video_path)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    window_stride = max(1, int(window_stride))
    display_offset = max(0, min(int(display_offset), clip_length - 1))
    display_start_offset = max(0, min(int(display_start_offset), clip_length - 1))
    display_end_offset = max(0, min(int(display_end_offset), clip_length - 1))

    device = next(model.parameters()).device
    # 배치 크기입니다. 주지 않으면 장치에 맞는 값(CPU 1, GPU 8, config)을 씁니다.
    infer_batch_size = infer_batch_size or predict_batch_size(device)
    model.eval()
    if is_cuda_like(device):
        torch.backends.cudnn.benchmark = True

    target_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

    # 3D-CNN 실행기입니다. 서버 컴퓨터에 맞는 가장 빠른 방식을 처음 한 번만 고릅니다(infer_runtime.py).
    # CAM 히트맵도 같은 실행기가 판정과 함께 내줍니다(runner.with_cam). 클립을 만들지 않으면(평가) 받지 않습니다.
    runner = get_runner(model)
    cam_overlay = cam_overlay and render_clips
    if use_coarse_scan and coarse_prob >= config.A_THRESHOLD:
        print(f"⚠️ 거친 탐색 기준 {coarse_prob:g} 이 판정 기준 A_THRESHOLD {config.A_THRESHOLD:g} 보다 낮지 않아 "
              "약한 충돌 신호를 흘릴 수 있습니다(config.PREDICT_COARSE_PROB)")
    # 영상을 처음부터 끝까지 한 번 읽으며 바로 판정합니다(스트리밍).
    # 최근 프레임 몇십 장만 메모리에 두고,
    #     거친 탐색 묶음(coarse_step 간격)은 프레임이 다 들어오는 대로 판정하고,
    #     의심 묶음 주변(coarse_pad 안)의 묶음은 그 프레임이 버퍼에 남아 있을 때 판정합니다.
    # 판정하는 묶음의 집합과 배치를 묶는 순서는 영상 전체를 쌓아 두고 판정할 때와 같아 결과가 같습니다.
    #     거친 묶음: 영상 순서대로 묶고, 끝 묶음이 격자에 없으면 맨 뒤에 붙입니다.
    #     촘촘한 묶음: 전체를 정렬한 순서로 묶습니다. 앞으로 나올 의심 묶음은 '아직 판정하지 않은 거친 묶음 - coarse_pad'
    #     이후에만 생기므로, 그보다 앞선 묶음은 순서가 확정됩니다.
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # 거친 탐색 간격과 최소 길이는 30fps 기준 값이라 이 영상 fps에 맞춥니다(config.PREDICT_FPS_SCALED_POSTPROC).
    coarse_step, min_span = fps_scaled_event_params(fps, coarse_step, config.PREDICT_MIN_EVENT_SPAN_FRAMES,
                                                    clip_length)

    ret, first_frame = cap.read()
    if not ret:
        cap.release()
        return []
    orig_h, orig_w = first_frame.shape[:2]
    _, (rx1, ry1, rx2, ry2) = _crop_square_and_pad(
        first_frame, target_bbox, r_value, resize)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # 원본 박스 영역(프레임 경계로 자름)입니다. CAM 오버레이와 박스를 사용자가 지정한 실제 박스 크기로 그릴 때 씁니다.
    bx1, by1 = max(0, target_bbox[0]), max(0, target_bbox[1])
    bx2, by2 = min(orig_w, target_bbox[2]), min(orig_h, target_bbox[3])

    # 피해 차량 박스를 224 크롭 좌표로 바꿔 충격 시점 계산 전용 영역으로 씁니다.
    # 정사각형 크롭 여백을 지나는 가해 차량의 움직임에 시점이 끌려가지 않게 하려는 것입니다.
    _cw = max(1, rx2 - rx1)
    _ch = max(1, ry2 - ry1)
    _sx, _sy = resize[0] / _cw, resize[1] / _ch
    _roi = (
        max(0, min(resize[0] - 1, int((bx1 - rx1) * _sx))),
        max(0, min(resize[1] - 1, int((by1 - ry1) * _sy))),
        max(1, min(resize[0], int((bx2 - rx1) * _sx))),
        max(1, min(resize[1], int((by2 - ry1) * _sy))),
    )

    def _report(frac=None, eta=None, eta_min=False, stage=None, count=None, index=None):
        """진행률(0~1), 지금 단계, 단계의 남은 시간을 보고합니다. 화면은 진행률과 단계 이름을 보여 줍니다.

        stage: decode(영상 불러오기), motion(차량 움직임 계산), coarse(영상 전체 훑어보기, count는 지금까지 찾은 의심 지점 수),
               refine(의심 구간 자세히 보기, index/count), scan(거친 탐색 없이 전부 보기), events(사고 구간 정리),
               render(사고 클립 만들기, index/count)
        진행률은 뒤로 가지 않습니다. 100%는 작업 완료(SUCCESS)로만 표시되도록 0.99에서 멈춥니다.
        eta는 지금 단계의 남은 시간(초)이며, 모르면 None입니다.
        """
        if progress_callback is None:
            return
        if eta is not None:
            eta = max(0.0, float(eta))
        progress_callback(min(0.99, frac or 0.0), eta, eta_min, stage, count, index)

    events = []          # [{'start_frame','end_frame'}, ...]
    # 표시 프레임(묶음 시작 + display_offset)에서 (칸별 'A - S' 로짓 7x7 또는 None, prob)로 가는 사전입니다(사고로 예측한 묶음만).
    accident_overlays = {}
    # window_idx에서 pred_class와 P(A)로 가는 사전입니다(3D-CNN을 실제로 돌린 묶음만).
    window_pred = {}
    window_prob_a = {}

    _mean = np.asarray(config.NORM_MEAN, dtype=np.float32).reshape(3, 1, 1)
    _std = np.asarray(config.NORM_STD, dtype=np.float32).reshape(3, 1, 1)
    # 프레임 번호에서 정규화한 224 크롭 (3, H, W) float32 텐서로 가는 버퍼입니다. 아직 필요한 최근 프레임만 둡니다.
    # 버퍼에 넣을 때 한 번만 정규화해, 1프레임씩 밀며 보는 촘촘한 묶음에서 같은 프레임을 반복해 바꾸지 않습니다.
    # 버퍼는 영상 길이와 상관없이 최대 백수십 장(약 0.1GB)입니다. 계산은 같은 값에 같은 순서라 입력은 비트까지 같습니다.
    # 버린 프레임의 텐서는 pool에 모아 새 프레임에 덮어씁니다. 프레임마다 새 메모리를 잡으면 의심 구간이 적은 영상이 오히려 느려집니다.
    # 계산은 numpy(스레드 하나)로 합니다. torch로 하면 여러 스레드가 영상 풀기와 광학흐름(OpenCV)과 다퉈 전체가 느려집니다.
    buf = {}
    pool = []

    def _normalize(rgb):
        t = pool.pop() if pool else torch.empty((3,) + rgb.shape[:2], dtype=torch.float32)
        a = t.numpy()                                       # 같은 메모리를 numpy로 봅니다.
        np.copyto(a, rgb.transpose(2, 0, 1), casting='unsafe')
        np.divide(a, np.float32(255.0), out=a)
        np.subtract(a, _mean, out=a)
        np.divide(a, _std, out=a)
        return t

    def _clip(w):
        """묶음 w의 입력 텐서 (3, T, H, W)를 돌려줍니다."""
        return torch.stack([buf[i] for i in range(w, w + clip_length)], dim=1)

    speed = {"n": 0, "sec": 0.0, "first": True}
    infer_sec = [0.0]       # 속도 측정용: 첫 배치까지 포함한 3D-CNN 추론 총시간
    infer_prog = [PROG_FLOW_END]

    def _run_batch(batch_window_starts, clips_cpu):
        """묶음 배치를 3D-CNN에 넣어 예측, P(A), CAM을 채웁니다."""
        t_batch = time.time()
        with torch.inference_mode():
            clips = torch.stack(clips_cpu).to(device)
            if is_channels_last_3d_supported(device) and config.USE_CHANNELS_LAST:
                clips = clips.contiguous(memory_format=torch.channels_last_3d)
            if cam_overlay:
                outputs, cams = runner.with_cam(clips)
            else:
                outputs, cams = runner(clips), None
            probs = F.softmax(outputs, dim=1)
            pred_classes = is_accident(outputs).long()         # P(A) > config.A_THRESHOLD
            for offset, window_idx in enumerate(batch_window_starts):
                # 판정과 CAM은 묶음 끝이 아니라 '묶음 시작 + display_offset' 프레임에 그립니다.
                # 묶음 끝에 붙이면 A 표시가 실제 충돌보다 늦기 때문입니다(config 주석 참고).
                disp_idx = window_idx + display_offset
                pred_class = int(pred_classes[offset].item())
                prob = probs[offset, pred_class].item()
                window_pred[window_idx] = pred_class
                window_prob_a[window_idx] = probs[offset, 1].item()
                if pred_class == 1:
                    # 히트맵은 클립을 만들 때 그립니다(_draw_collision_cam). 여기서는 칸별 'A - S' 로짓 49개만 남겨,
                    # 사고 구간이 길어도 박스 크기 그림을 쌓지 않습니다.
                    cam_as = (None if cams is None
                              else (cams[offset, 1] - cams[offset, 0]).float().cpu().numpy())
                    accident_overlays[disp_idx] = (cam_as, prob)
        dt = time.time() - t_batch
        infer_sec[0] += dt
        if speed["first"]:
            speed["first"] = False
        else:
            speed["n"] += len(batch_window_starts)
            speed["sec"] += dt

    k = max(1, int(round(coarse_step / window_stride))) if use_coarse_scan else 1
    grid = k * window_stride                 # 거친 묶음 간격(프레임)
    refine_on = k > 1
    coarse_q = []        # 프레임이 다 들어와 판정을 기다리는 거친 묶음(영상 순서)
    next_grid = 0        # 다음에 들어올 거친 묶음 시작
    coarse_list = []     # 판정한 거친 묶음(출력용)
    hot = []             # P(A) > coarse_prob인 거친 묶음
    refine_pending = set()   # 촘촘히 볼 묶음 중 아직 판정하지 않은 것
    refine_held = []     # 순서가 확정돼 배치 짝을 기다리는 묶음(묶음 시작, 입력 텐서)
    refine_done = []     # 판정한 촘촘한 묶음(출력용)

    def _hot_regions():
        """의심 구간 수입니다. 서로 이어진 '촘촘히 볼 범위' 덩어리 수(화면의 '의심 지점 N곳')입니다."""
        n, end = 0, None
        for h in hot:
            lo = max(0, h - coarse_pad)
            if end is None or lo > end:
                n += 1
            end = h + coarse_pad
        return n

    def _flush_coarse(final=False):
        while coarse_q and (len(coarse_q) >= infer_batch_size or final):
            ws = coarse_q[:infer_batch_size]
            del coarse_q[:len(ws)]
            _run_batch(ws, [_clip(w) for w in ws])
            coarse_list.extend(ws)
            for w in ws:
                if refine_on and window_prob_a[w] > coarse_prob:
                    hot.append(w)
                    lo = max(0, w - coarse_pad)
                    lo += (-lo) % window_stride        # 묶음 시작 격자에 맞춥니다.
                    for r in range(lo, w + coarse_pad + 1, window_stride):
                        if r % grid != 0 and r not in window_pred:
                            refine_pending.add(r)

    def _future_coarse(cur_frame):
        """앞으로 판정할 거친 묶음의 하한입니다. 대기열 첫 묶음, 다음 격자 묶음, 그리고 아직 모르는
        영상 끝 묶음(지금 들어온 마지막 묶음 이후)이 될 수 있는 가장 이른 위치입니다.
        """
        return min(coarse_q[0] if coarse_q else next_grid, next_grid,
                   max(0, cur_frame - clip_length + 1))

    def _flush_refine(cur_frame, final=False, last_start=None):
        """순서가 확정되고 프레임이 준비된 촘촘한 묶음을 정렬 순서대로 두 개씩 판정합니다."""
        if final:
            ready_limit = last_start
            order_limit = float("inf")
        else:
            ready_limit = cur_frame - clip_length + 1          # 이 묶음까지 프레임이 다 들어왔습니다.
            order_limit = _future_coarse(cur_frame) - coarse_pad   # 이보다 앞선 묶음은 순서가 확정됐습니다.
        for w in sorted(refine_pending):
            if w > ready_limit or w >= order_limit:
                break
            refine_pending.discard(w)
            refine_held.append((w, _clip(w)))
            if len(refine_held) == infer_batch_size:
                ws = [x[0] for x in refine_held]
                _run_batch(ws, [x[1] for x in refine_held])
                refine_done.extend(ws)
                refine_held.clear()
        if final and refine_held:
            ws = [x[0] for x in refine_held]
            _run_batch(ws, [x[1] for x in refine_held])
            refine_done.extend(ws)
            refine_held.clear()

    # 충격 시점용 광학흐름입니다. 읽는 대로 같은 간격(flow_step)으로 계산하고 숫자만 남깁니다.
    flow_step = max(1, int(config.PREDICT_FLOW_SAMPLE_STEP))
    flow_mag_l, flow_roi_l = [], []
    flow_ok = True
    prev_gray = None
    qx1, qy1, qx2, qy2 = _roi

    total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    decode_sec = flow_sec = 0.0
    t_stream = time.time()
    n_read = 0
    last_frame = None
    max_buf = [0]
    stage_name = 'coarse' if k > 1 else 'scan'
    _report(0.0, stage='decode')
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        processed, _ = _crop_square_and_pad(frame_rgb, target_bbox, r_value, resize)
        i = n_read
        buf[i] = _normalize(processed)
        last_frame = buf[i]
        n_read += 1
        decode_sec += time.time() - t0

        # 광학흐름(flow_step 간격 프레임 쌍)은 이벤트 안에서 충격 시점(움직임이 가장 큰 프레임)을 고르는 데만 씁니다.
        # 크롭 전체(mag)와 피해 차량 박스 영역(mag_roi) 두 줄을 만들고 시점은 박스 쪽으로 봅니다.
        # 크롭 여백을 지나는 가해 차량의 움직임이 피해 차량의 미세한 흔들림보다 커서, 크롭 전체로 잡으면 접촉이 아니라 가해 차량 진입을 가리키기 때문입니다.
        t0 = time.time()
        flow_mag_l.append(0.0)
        flow_roi_l.append(0.0)
        if flow_ok and i % flow_step == 0:
            try:
                gray = cv2.cvtColor(processed, cv2.COLOR_RGB2GRAY)
                if prev_gray is not None:
                    flow = cv2.calcOpticalFlowFarneback(
                        prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
                    fmag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    m = float(fmag.mean())
                    sub = fmag[qy1:qy2, qx1:qx2]
                    mr = float(sub.mean()) if sub.size else m
                    for j in range(i - flow_step + 1, i + 1):
                        flow_mag_l[j] = m
                        flow_roi_l[j] = mr
                prev_gray = gray
            except Exception:
                flow_ok = False   # 실패하면 충격 시점은 이벤트 시작으로 대신합니다.
        flow_sec += time.time() - t0

        # 프레임이 다 들어온 거친 묶음을 대기열에 넣고, 배치가 차면 판정합니다.
        while next_grid + clip_length - 1 <= i:
            coarse_q.append(next_grid)
            next_grid += grid
        _flush_coarse()
        if refine_on:
            _flush_refine(i)

        # 더 이상 필요 없는 오래된 프레임은 버립니다. 앞으로 판정할 거친 묶음과 그 주변(pad 안)의 촘촘한 묶음,
        # 아직 판정하지 않은 촘촘한 묶음에 필요한 프레임만 남깁니다.
        need = _future_coarse(i)
        if refine_on:
            need = max(0, need - coarse_pad)
            if refine_pending:
                need = min(need, min(refine_pending))
        for old in [f for f in buf if f < need]:
            pool.append(buf.pop(old))       # 판정에 넘긴 묶음은 이미 복사본이라 덮어써도 됩니다(_clip).
        max_buf[0] = max(max_buf[0], len(buf))

        # 진행률은 영상을 읽은 비율입니다(남은 시간은 남은 프레임 수와 지금까지의 프레임당 시간으로 계산).
        # 긴 영상에서도 막대가 멈춰 보이지 않게 10프레임마다 알립니다(실제 기록 빈도는 prediction_job이 0.4초 단위로 조절합니다).
        if total_frames_hint > 0 and n_read % 10 == 0:
            frac = min(1.0, n_read / total_frames_hint)
            infer_prog[0] = PROG_FLOW_END + (PROG_INFER_END - PROG_FLOW_END) * frac
            per = (time.time() - t_stream) / n_read
            _report(infer_prog[0], eta=(total_frames_hint - n_read) * per,
                    stage=stage_name, count=_hot_regions() if k > 1 else None)
    cap.release()

    if n_read == 0:
        return []
    real_frame_count = n_read
    # 묶음 길이보다 짧은 영상은 마지막 프레임을 반복해 묶음 하나를 채웁니다.
    n_total = n_read
    while n_total < clip_length:
        buf[n_total] = last_frame
        n_total += 1
        while next_grid + clip_length - 1 <= n_total - 1:
            coarse_q.append(next_grid)
            next_grid += grid
    decode_sec_per_frame = decode_sec / max(1, n_read)

    num_windows = n_total - (clip_length - 1)
    window_starts = list(range(0, max(0, num_windows), window_stride))
    last_start = window_starts[-1]
    # 영상 끝 묶음이 거친 격자에 없으면 거친 묶음으로 맨 뒤에 붙입니다.
    if last_start % grid != 0:
        refine_pending.discard(last_start)
        coarse_q.append(last_start)
    _flush_coarse(final=True)
    if refine_on:
        _report(infer_prog[0], stage='refine', count=_hot_regions())
        _flush_refine(None, final=True, last_start=last_start)
    infer_prog[0] = PROG_INFER_END
    buf.clear()

    flow_mag = flow_mag_roi = None
    if flow_ok and n_total >= clip_length + 2:
        flow_mag = np.asarray(flow_mag_l, dtype=np.float32)
        flow_mag_roi = np.asarray(flow_roi_l, dtype=np.float32)

    _report(infer_prog[0], stage='events')
    print(f"[2단계] 거친 탐색 {len(coarse_list)}개(간격 {grid}) → "
          f"의심 {len(hot)}개 주변 {len(refine_done)}개 추가 | 전체 "
          f"{len(window_starts)} 윈도우 중 3D-CNN 평가 {len(window_pred)}개 "
          f"({100.0 * len(window_pred) / max(1, len(window_starts)):.0f}%) "
          f"| 영상 {real_frame_count}프레임 {fps:.1f}fps · 최소 길이 {min_span}, 버퍼 최대 {max_buf[0]}장")

    # 전체 묶음을 순서대로 상태 기계(S, A, S)로 훑어 이벤트 구간을 정합니다.
    # 평가하지 않은 묶음은 사고가 아닌 것(0)으로 봅니다.
    prev_state = 0
    event_start = None
    for window_idx in window_starts:
        frame_idx = window_idx + clip_length - 1
        pred_class = window_pred.get(window_idx, 0)
        if prev_state == 0 and pred_class == 1:
            event_start = window_idx
        elif prev_state == 1 and pred_class == 0:
            if event_start is not None:
                events.append({'start_frame': event_start,
                               'end_frame': frame_idx - 1})
                event_start = None
        prev_state = pred_class

    # 영상 끝까지 A 상태가 이어진 경우 이벤트를 닫습니다.
    if prev_state == 1 and event_start is not None:
        events.append({'start_frame': event_start,
                       'end_frame': real_frame_count - 1})

    # 이벤트 후처리: 병합한 뒤 길이 필터를 겁니다(한 충돌이 여러 개로 쪼개지는 것을 막습니다).
    # 먼저 병합해야 한 충돌의 조각들이 하나로 합쳐져 길이를 되찾고, 그 뒤에 남은 단발 깜빡임만 길이 필터로 지워집니다.
    _n_raw = len(events)
    # 병합 간격은 초 단위 설정을 이 영상의 fps로 프레임으로 바꿉니다.
    merge_gap_frames = int(round(fps * config.PREDICT_EVENT_MERGE_GAP_SEC))
    if events:
        events.sort(key=lambda e: e['start_frame'])
        merged = [events[0]]
        for ev in events[1:]:
            gap = ev['start_frame'] - merged[-1]['end_frame']
            if gap <= merge_gap_frames:
                merged[-1]['end_frame'] = max(
                    merged[-1]['end_frame'], ev['end_frame'])
            else:
                merged.append(ev)
        events = [e for e in merged
                  if (e['end_frame'] - e['start_frame']) >= min_span]
    if _n_raw != len(events):
        print(f"[이벤트 후처리] {_n_raw}개 → 병합·길이필터 후 {len(events)}개")

    # 두 번째 단계: 이벤트 구간 프레임만 영상에서 다시 읽어 CAM 클립을 그립니다.
    # 원본 프레임을 메모리에 들고 있지 않고, 사고 구간마다 cap.set으로 찾아가 읽습니다.
    results = []
    base_name = clip_name or os.path.splitext(os.path.basename(video_path))[0]
    render_cap = (cv2.VideoCapture(video_path)
                  if (events and render_clips) else None)
    # 클립 렌더링 남은 시간은 남은 프레임 수와 프레임당 시간으로 계산합니다(재기 전에는 디코딩 기준 추정).
    render_total = sum(
        min(real_frame_count - 1, e['end_frame'] + clip_pad_frames)
        - max(0, e['start_frame'] - clip_pad_frames) + 1
        for e in events) if render_clips else 0
    render_done = [0]
    render_idx = [0]        # 지금 만드는 클립 번호(1부터)
    render_start = infer_prog[0]   # 클립 만들기는 지금 진행률에서 99%까지 씁니다.
    t_render = time.time()

    def _report_render():
        if render_done[0] >= 15:
            per = (time.time() - t_render) / render_done[0]
        else:
            per = decode_sec_per_frame * RENDER_COST_PER_DECODE
        frac = render_start + (0.99 - render_start) * render_done[0] / max(1, render_total)
        _report(frac, eta=(render_total - render_done[0]) * per, stage='render',
                count=len(events), index=render_idx[0])

    try:
        for ev_idx, ev in enumerate(events, 1):
            render_idx[0] = ev_idx
            _report_render()
            start_f = ev['start_frame']
            end_f = ev['end_frame']
            clip_start = max(0, start_f - clip_pad_frames)
            clip_end = min(real_frame_count - 1, end_f + clip_pad_frames)

            # 구간 대표 확률은 구간 안 사고 프레임 확률의 최댓값입니다.
            probs_in_event = [p for f, (_, p) in accident_overlays.items()
                              if start_f <= f <= end_f]
            crash_prob = max(probs_in_event) if probs_in_event else None

            clip_path = None
            raw_clip_path = None
            if render_clips:
                clip_stem = str(output_dir / f'{base_name}_event{ev_idx}')
                writer, rendered_path = _open_clip_writer(
                    clip_stem, fps, (orig_w, orig_h + _state_bar_height(orig_h)))
                # 원본 사고 클립은 같은 프레임 구간을 아무것도 합성하지 않고 그대로 저장합니다.
                # 사용자가 박스와 상태 표시 없는 원본을 따로 내려받을 수 있게 하려는 것입니다.
                raw_stem = f'{clip_stem}_raw'
                raw_writer, raw_rendered = _open_clip_writer(raw_stem, fps, (orig_w, orig_h))


                # 사고 구간 시작 프레임으로 이동한 뒤 차례로 디코딩합니다.
                # A로 그리는 구간은 (이벤트 안 첫 A 묶음 시작 + display_start_offset)부터 (마지막 A 묶음 시작 + display_end_offset)까지입니다(config 주석 참고).
                # 반응이 짧아 띠가 PREDICT_DISPLAY_MIN_SEC보다 짧으면(시작 > 끝 포함) 양 끝 모두 display_offset으로 그립니다.
                # CAM은 '묶음 시작 + display_offset' 프레임에 붙여 두고 띠 안에서 가장 최근 것을 씁니다.
                a_wins = sorted(w for w, c in window_pred.items() if c == 1 and start_f <= w <= end_f)
                a_first = a_last = None
                if a_wins:
                    a_first = a_wins[0] + display_start_offset
                    a_last = a_wins[-1] + display_end_offset + window_stride - 1
                    if a_last - a_first + 1 < max(1, int(round(config.PREDICT_DISPLAY_MIN_SEC * fps))):
                        a_first = a_wins[0] + display_offset
                        a_last = a_wins[-1] + display_offset + window_stride - 1
                render_cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
                last_cam = None
                # 박스 선 굵기는 해상도에 맞춥니다(1080p 이하 2px, 1440p 3px, 4K 4px).
                # 1px는 H.264(yuv420p, 색 정보가 절반 해상도)로 압축하면 빨간 선이 흐려지므로 최소 2px입니다.
                box_thick = max(2, int(round(orig_h / 540)))
                for f in range(clip_start, clip_end + 1):
                    ret, frame = render_cap.read()  # 이미 BGR입니다.
                    if not ret:
                        break
                    raw_writer.write(frame)   # 그리기 전에 원본부터 저장합니다.
                    if f in accident_overlays and accident_overlays[f][0] is not None:
                        last_cam = accident_overlays[f][0]
                    in_event = a_first is not None and a_first <= f <= a_last
                    if in_event:
                        if last_cam is not None:
                            # 충돌 부위에만 히트맵을 칠하고, 박스 테두리(사고 빨강, 정상 초록)는 그 위에 그립니다.
                            _draw_collision_cam(frame, last_cam, (rx1, ry1, rx2, ry2), (bx1, by1, bx2, by2))
                        cv2.rectangle(frame, (bx1, by1),
                                      (bx2, by2), (0, 0, 255), box_thick, cv2.LINE_AA)
                        state = 1
                    else:
                        cv2.rectangle(frame, (bx1, by1),
                                      (bx2, by2), (0, 255, 0), box_thick, cv2.LINE_AA)
                        state = 0
                    # S/A는 차량을 가리지 않도록 영상 밖(아래 띠)에 표시합니다.
                    writer.write(_add_state_bar(frame, state))
                    render_done[0] += 1
                    if render_done[0] % 15 == 0:
                        _report_render()
                writer.release()
                raw_writer.release()
                # 브라우저 호환 최종본(H.264 mp4)으로 변환합니다.
                clip_path = _finalize_clip(rendered_path, clip_stem)
                raw_clip_path = _finalize_clip(raw_rendered, raw_stem)

            # 사고 시점 확정: 이벤트 구간 안 광학흐름 최대점(충격 순간)을 사고 시점으로 봅니다.
            # 묶음 시작은 실제 충돌보다 체계적으로 이르고, 단순히 움직임이 처음 커지는 곳은 다가오는 차의 움직임을 먼저 잡습니다.
            # 레이블과 대조했을 때 흐름 최대점이 충돌 구간 안에 가장 잘 들어왔습니다.
            # 시점은 피해 차량 박스 영역의 흐름으로 정하고, 없으면 크롭 전체로 대신합니다.
            _mag_for_impact = (flow_mag_roi if flow_mag_roi is not None
                               else flow_mag)
            impact_f = start_f
            if _mag_for_impact is not None and end_f >= start_f:
                seg = _mag_for_impact[
                    start_f:min(end_f + 1, len(_mag_for_impact))]
                if seg.size:
                    impact_f = start_f + int(np.argmax(seg))

            results.append({
                # start_*는 사고 발생(충격) 시점이며 화면 타임라인과 DB 기록에 씁니다.
                'start_frame': impact_f,
                'start_sec': impact_f / fps,
                'end_frame': end_f,
                'end_sec': end_f / fps,
                # 참고용: 모델이 A로 판정한 구간 전체(클립 렌더링 범위 기준)
                'event_start_frame': start_f,
                'event_end_frame': end_f,
                'crash_prob': crash_prob,
                'clip_path': clip_path,
                'raw_clip_path': raw_clip_path,   # 합성 없는 원본 사고 클립(이름은 클립 + '_raw')
                # 클립 첫 프레임이 원본의 이 프레임입니다. 재생 화면이 원본 시간에 맞춰 CAM 클립을 겹쳐 보여 줄 때 씁니다.
                'clip_start_frame': clip_start,
                'clip_start_sec': clip_start / fps,
            })
    finally:
        if render_cap is not None:
            render_cap.release()

    _report(0.99, eta=0.0, stage='render' if events and render_clips else 'events',
            count=len(events) if events and render_clips else None,
            index=len(events) if events and render_clips else None)
    if stats is not None:
        stats.update(frames=real_frame_count, decode_sec=decode_sec, flow_sec=flow_sec,
                     infer_sec=infer_sec[0], render_sec=time.time() - t_render,
                     windows_total=len(window_starts), windows_eval=len(window_pred),
                     backend=runner.name)
    print(f"[predict_events_and_clips] {len(results)}건 사고구간 클립 생성")
    return results
