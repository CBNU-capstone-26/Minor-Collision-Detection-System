import os
import queue
import shutil
import subprocess
import threading
import cv2
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path

import config
from device_utils import is_cuda_like, is_channels_last_3d_supported

# 시스템 ffmpeg(H.264/libx264) 경로 — 있으면 클립을 어디서든 재생 가능한 mp4로 만든다.
_FFMPEG = shutil.which("ffmpeg")


activation = {}

# 진행도(0.0~1.0) 구간 배분 — 단계마다 '실제' 진행도를 보고해 UI가 가짜 추정을
# 쓰지 않게 한다(가짜 추정 → 실제값 전환 시 진행바가 뒤로 점프하는 문제 방지).
PROG_DECODE_END = 0.25    # 프레임 디코딩/크롭
PROG_PRESCREEN_END = 0.30  # 광학흐름 사전선별
PROG_INFER_END = 0.95     # S3D 윈도우 추론
# 0.95~1.0 = 사고구간 CAM 클립 렌더링


def _open_clip_writer(dest_stem, fps, size):
    """클립을 mp4v 임시본에 렌더할 VideoWriter를 연다. 반환: (writer, rendered_path).

    최종 브라우저 호환 변환(H.264/mp4)은 _finalize_clip이 담당한다.
    """
    rendered = f"{dest_stem}.render.mp4"
    writer = cv2.VideoWriter(rendered, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    return writer, rendered


def _finalize_clip(rendered_path, dest_stem):
    """mp4v 임시본을 H.264/mp4(yuv420p+faststart)로 재인코딩해 최종 경로(str)를 반환.

    시스템 ffmpeg(libx264)가 **필수**다. H.264/mp4로 통일해야 브라우저·OS(크롬·파폭·
    사파리·iOS)를 가리지 않고 재생되기 때문. ffmpeg가 없으면 명확한 에러를 낸다.
    """
    if not _FFMPEG:
        try:
            os.remove(rendered_path)
        except OSError:
            pass
        raise RuntimeError(
            "ffmpeg가 설치되어 있지 않습니다 — 브라우저 호환 클립(H.264) 생성에 필요합니다. "
            "시스템에 설치하세요: sudo apt install -y ffmpeg (또는 brew install ffmpeg)")
    final = f"{dest_stem}.mp4"
    subprocess.run(
        [_FFMPEG, "-y", "-loglevel", "error", "-i", rendered_path,
         "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-movflags", "+faststart", final],
        check=True)
    os.remove(rendered_path)
    return final


def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook


def _frames_to_video_tensor(frames):
    # 정규화 통계는 학습과 동일해야 함 — config에서 일괄 관리(S3D Kinetics-400)
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
    cv2.rectangle(frame, (10, 10), (90, 65), (0, 0, 0), -1)
    cv2.putText(frame, label, (20, 57),
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
        out_stem = str(out_path.with_suffix(""))
        out_video, out_rendered = _open_clip_writer(
            out_stem, 30.0, (orig_w, orig_h))

        write_queue = queue.Queue(maxsize=64)
        writer_thread = threading.Thread(
            target=_video_writer_worker,
            args=(write_queue, out_video),
            daemon=True,
        )
        writer_thread.start()

        # CAM 오버레이·박스를 정사각 크롭이 아니라 '원본 bbox' 크기로 그린다.
        # (변수명 v_n*는 유지하되 좌표를 원본 bbox 기준으로 정의 — 이후 rectangle/
        #  putText/heatmap 슬라이싱이 모두 자동으로 원본 bbox 영역을 사용하게 됨)
        v_nx1, v_ny1 = max(0, target_bbox[0]), max(0, target_bbox[1])
        v_nx2, v_ny2 = min(orig_w, target_bbox[2]), min(orig_h, target_bbox[3])

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
                        # 반올림 1px 차이를 흡수하도록 히트맵을 bbox 영역 크기에 맞춤
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
        # 브라우저 호환 최종본(H.264/mp4)으로 변환
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


def _flow_prescreen_suspicious(
    processed_frames,
    clip_length,
    roi=None,
    threshold_factor=config.PREDICT_FLOW_THRESHOLD_FACTOR,
    pad=config.PREDICT_FLOW_PRESCREEN_PAD,
    sample_step=config.PREDICT_FLOW_SAMPLE_STEP,
    max_ratio=config.PREDICT_FLOW_MAX_SUSPICIOUS_RATIO,
):
    """[2단계 1단계] 피해차 크롭(processed_frames, 224 RGB)에서 옵티컬 플로우
    크기 시계열을 만들어 '사고 의심 프레임'의 불리언 배열을 반환한다.

    - 이미 RAM에 있는 크롭 프레임을 쓰므로 추가 영상 디코딩이 없다.
    - 크롭은 r=1이라 피해차가 대부분을 차지 → 크롭 전체 평균 플로우가 사실상
      마스크 플로우. 접근/통과 차량이 크롭에 들어오면 스파이크 → 고재현율.
    - 임계값 τ=median+factor·std(강건). τ 초과 지점을 ±pad로 팽창.
    - 의심 비율이 max_ratio를 넘거나 계산 실패 시 '전체 True'로 폴백(정확도 우선).

    Args:
        roi: (x1,y1,x2,y2) — 224 크롭 좌표계에서의 '피해차 bbox' 영역.
            주면 이 영역만의 플로우 시계열(mag_roi)을 함께 만든다.

    Returns:
        (suspicious, mag_crop, mag_roi)
        suspicious: np.bool_ 배열 (len == len(processed_frames)) — 사전선별용
        mag_crop  : 크롭 전체 평균 플로우 시계열 (사전선별 판단에 사용)
        mag_roi   : 피해차 bbox 영역만의 플로우 시계열 (충격 '시점' 판정에 사용)
                    roi가 없거나 계산 실패 시 None

    ⚠️ 두 시계열을 분리하는 이유(실측 근거):
       크롭은 정사각형이라 차량 위아래에 여백이 생기고, **가해차량이 그 여백을
       통과할 때의 모션이 피해차의 미세 흔들림을 압도**한다. 크롭 전체 argmax로
       충격 시점을 잡으면 접촉이 아니라 '가해차량 진입'을 가리킨다
       (실차 CCTV 실측: bbox 한정 피크와 -52 / +105 프레임까지 어긋남).
       → 선별은 넓게(고재현율), 시점 판정은 좁게(bbox 한정).
    """
    n = len(processed_frames)
    all_true = np.ones(n, dtype=bool)
    if n < clip_length + 2:
        return all_true, None, None  # 너무 짧으면 전체 스캔
    try:
        grays = [cv2.cvtColor(f, cv2.COLOR_RGB2GRAY) for f in processed_frames]
        step = max(1, int(sample_step))
        mag = np.zeros(n, dtype=np.float32)
        mag_roi = np.zeros(n, dtype=np.float32) if roi is not None else None
        if roi is not None:
            rx1, ry1, rx2, ry2 = roi
        prev_i = 0
        for i in range(step, n, step):
            flow = cv2.calcOpticalFlowFarneback(
                grays[prev_i], grays[i], None, 0.5, 3, 15, 3, 5, 1.2, 0)
            fmag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            m = float(fmag.mean())
            mag[prev_i + 1:i + 1] = m  # 구간에 동일값 채움
            if mag_roi is not None:
                sub = fmag[ry1:ry2, rx1:rx2]
                mag_roi[prev_i + 1:i + 1] = float(sub.mean()) if sub.size else m
            prev_i = i
        med = float(np.median(mag))
        std = float(mag.std())
        tau = med + threshold_factor * std
        suspicious = mag > tau
        if suspicious.any():
            # ±pad 프레임 팽창(스파이크로 끝나는 윈도우까지 평가되도록)
            idx = np.where(suspicious)[0]
            dilated = np.zeros(n, dtype=bool)
            for j in idx:
                dilated[max(0, j - pad):min(n, j + pad + 1)] = True
            suspicious = dilated
        ratio = suspicious.mean()
        if ratio == 0.0 or ratio > max_ratio:
            return all_true, mag, mag_roi  # 후보 없음/과다 → 전체 스캔 폴백
        return suspicious, mag, mag_roi
    except Exception:
        return all_true, None, None  # 실패하면 안전하게 전체 스캔


def predict_events_and_clips(
    model,
    video_path,
    bbox,
    output_dir,
    r_value=config.R_VALUE,
    resize=config.RESIZE,
    clip_length=config.CLIP_LENGTH,
    infer_batch_size=config.PREDICT_INFER_BATCH_SIZE,
    window_stride=config.PREDICT_WINDOW_STRIDE,
    clip_pad_frames=15,
    progress_callback=None,
    use_flow_prescreen=config.PREDICT_USE_FLOW_PRESCREEN,
):
    """웹 서비스용: 단일 대상 차량(bbox)에 대해 사고 의심 구간만 탐지하고,
    각 구간에 대해서만 짧은 CAM 오버레이 클립을 생성한다.

    전체 길이 결과 영상을 재생성하지 않으므로 멀티-아워 영상에도 효율적이다.

    Args:
        model: 로드된 HitAndRun3DCNN
        video_path (str|Path): 원본 영상 경로
        bbox (tuple|list): 대상 차량 좌표 (xmin, ymin, xmax, ymax) — 원본 해상도 기준
        output_dir (str|Path): 클립 저장 폴더
        clip_pad_frames (int): 구간 앞뒤로 덧붙일 여유 프레임 수

    Returns:
        list[dict]: 이벤트 목록. 각 항목:
            {
                'start_frame', 'end_frame',
                'start_sec', 'end_sec',
                'crash_prob',        # 구간 대표 확률 (0~1)
                'clip_path',         # 생성된 CAM 클립 경로 (Path)
            }
    """
    video_path = str(video_path)
    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    window_stride = max(1, int(window_stride))

    device = next(model.parameters()).device
    model.eval()
    if is_cuda_like(device):
        torch.backends.cudnn.benchmark = True

    target_bbox = [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])]

    handle = model.inception5b.register_forward_hook(
        get_activation('inception5b'))
    try:
        # ── 1패스: 추론용 224×224 프레임만 메모리에 보관 (원본 프레임은 보관 X) ──
        # 원본 해상도 프레임을 전부 RAM에 들면 영상 길이×해상도에 비례해 메모리가
        # 폭증하므로, 추론에 필요한 224×224 프레임만 유지한다. 원본 프레임은
        # 클립 렌더링 단계에서 "사고 구간만" 영상에서 다시 읽는다.
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        processed_frames = []

        ret, first_frame = cap.read()
        if not ret:
            cap.release()
            return []
        orig_h, orig_w = first_frame.shape[:2]
        _, (rx1, ry1, rx2, ry2) = _crop_square_and_pad(
            first_frame, target_bbox, r_value, resize)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # 디코딩 단계도 실제 진행도로 보고한다(0~PROG_DECODE_END).
        # 이게 없으면 긴 영상에서 '진행도 없음' 구간이 길어져 UI가 가짜 추정을
        # 쓰게 되고, 이후 실제 진행도가 오면 진행바가 뒤로 점프한다.
        total_frames_hint = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        _decoded = 0
        _last_decode_report = -1
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            processed, _ = _crop_square_and_pad(
                frame_rgb, target_bbox, r_value, resize)
            processed_frames.append(processed)
            _decoded += 1
            if progress_callback is not None and total_frames_hint > 0:
                frac = PROG_DECODE_END * min(1.0, _decoded / total_frames_hint)
                step = int(frac * 100)
                if step != _last_decode_report:   # 정수 %마다만 보고
                    _last_decode_report = step
                    progress_callback(frac)
        cap.release()

        if not processed_frames:
            return []

        real_frame_count = len(processed_frames)
        while len(processed_frames) < clip_length:
            processed_frames.append(processed_frames[-1])

        full_video_tensor = _frames_to_video_tensor(processed_frames).to(device)

        # 원본 bbox 영역(프레임 경계로 클램프) — CAM 오버레이·박스를 사용자가 지정한
        # 실제 bbox 크기로 그리기 위해 사용한다(정사각 크롭 크기가 아님).
        bx1, by1 = max(0, target_bbox[0]), max(0, target_bbox[1])
        bx2, by2 = min(orig_w, target_bbox[2]), min(orig_h, target_bbox[3])

        # ── 추론: 윈도우별 예측 + 사고 프레임의 CAM 히트맵 캐싱 ──────────────
        events = []          # [{'start_frame','end_frame'}, ...]
        # frame_idx → (heatmap_bbox, prob) (사고로 예측된 프레임만)
        accident_overlays = {}
        # window_idx → pred_class (S3D를 실제로 실행한 윈도우만)
        window_pred = {}

        num_windows = full_video_tensor.size(1) - (clip_length - 1)
        window_starts = list(range(0, max(0, num_windows), window_stride))

        # [2단계 1단계] 광학흐름 사전선별: '사고 의심 프레임'으로 끝나는 윈도우만
        # S3D로 평가한다. 나머지는 비사고(class 0)로 간주(선별기가 고재현율이므로
        # 실제 충돌은 반드시 스파이크 근처에 있음). 멀티-아워 영상에서 S3D 호출을
        # 대폭 줄인다. 선별 실패/과다 시 helper가 '전체 True'를 돌려 전체스캔 폴백.
        # 피해차 bbox를 224 크롭 좌표계로 환산 → 충격 '시점' 판정 전용 ROI.
        # (정사각 크롭 여백을 지나는 가해차량 모션에 시점이 끌려가지 않게 한다)
        _cw = max(1, rx2 - rx1)
        _ch = max(1, ry2 - ry1)
        _sx, _sy = resize[0] / _cw, resize[1] / _ch
        _roi = (
            max(0, min(resize[0] - 1, int((bx1 - rx1) * _sx))),
            max(0, min(resize[1] - 1, int((by1 - ry1) * _sy))),
            max(1, min(resize[0], int((bx2 - rx1) * _sx))),
            max(1, min(resize[1], int((by2 - ry1) * _sy))),
        )
        # 플로우 시계열은 '충격 시점(피크)' 산출에도 쓰므로 항상 계산한다(224 크롭이라 저렴).
        suspicious, flow_mag, flow_mag_roi = _flow_prescreen_suspicious(
            processed_frames, clip_length, roi=_roi)
        if not use_flow_prescreen:
            suspicious = np.ones(len(processed_frames), dtype=bool)
        n_frames_susp = len(suspicious)
        eval_window_starts = [
            w for w in window_starts
            if suspicious[min(w + clip_length - 1, n_frames_susp - 1)]]
        if not eval_window_starts:
            eval_window_starts = window_starts  # 방어적: 하나도 없으면 전체
        print(f"[2단계] 사전선별: 전체 {len(window_starts)} 윈도우 → "
              f"S3D 평가 {len(eval_window_starts)}개 "
              f"({100.0 * len(eval_window_starts) / max(1, len(window_starts)):.0f}%)")
        if progress_callback is not None:
            progress_callback(PROG_PRESCREEN_END)

        # Phase A: 선별된 윈도우에만 S3D 실행 (예측/확률/CAM 저장)
        with torch.inference_mode():
            for batch_start in range(0, len(eval_window_starts), infer_batch_size):
                # 실제 추론 진행도 보고 (사전선별 종료~추론 종료 구간에 매핑)
                if progress_callback is not None:
                    frac = batch_start / max(1, len(eval_window_starts))
                    progress_callback(
                        PROG_PRESCREEN_END
                        + (PROG_INFER_END - PROG_PRESCREEN_END) * frac)
                batch_window_starts = eval_window_starts[
                    batch_start:batch_start + infer_batch_size]
                clips = torch.stack(
                    [full_video_tensor[:, i:i + clip_length, :, :]
                     for i in batch_window_starts])
                if is_channels_last_3d_supported(device):
                    clips = clips.contiguous(
                        memory_format=torch.channels_last_3d)

                outputs = model(clips)
                probs = F.softmax(outputs, dim=1)
                pred_classes = outputs.argmax(dim=1)
                feat_maps = activation['inception5b']

                for offset, window_idx in enumerate(batch_window_starts):
                    frame_idx = window_idx + clip_length - 1
                    pred_class = int(pred_classes[offset].item())
                    prob = probs[offset, pred_class].item()
                    window_pred[window_idx] = pred_class

                    if pred_class == 1:
                        feat_map = feat_maps[offset]
                        weight = model.head_conv.weight[pred_class]
                        cam = F.relu(torch.sum(weight * feat_map, dim=0))
                        cam_2d = torch.mean(cam, dim=0)
                        cam_min, cam_max = cam_2d.min(), cam_2d.max()
                        cam_2d = (cam_2d - cam_min) / (cam_max - cam_min + 1e-8)
                        cam_np = (cam_2d * 255).byte().cpu().numpy()
                        heatmap = cv2.applyColorMap(cam_np, cv2.COLORMAP_JET)
                        heatmap = cv2.resize(heatmap, (rx2 - rx1, ry2 - ry1))
                        # 정사각 히트맵에서 '원본 bbox'에 해당하는 부분만 잘라 저장
                        heatmap_bbox = heatmap[
                            max(0, by1 - ry1):(by2 - ry1),
                            max(0, bx1 - rx1):(bx2 - rx1),
                        ]
                        accident_overlays[frame_idx] = (heatmap_bbox, prob)

        # Phase B: 전체 윈도우 순서대로 상태머신(S→A→S)으로 이벤트 구간 확정.
        # 평가 안 한 윈도우는 비사고(0)로 간주한다.
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

        # 영상 끝까지 A 상태가 유지된 경우 이벤트 닫기
        if prev_state == 1 and event_start is not None:
            events.append({'start_frame': event_start,
                           'end_frame': real_frame_count - 1})

        # ── 이벤트 후처리: 병합 → 길이필터 (한 충돌이 여러 개로 쪼개지는 것 방지) ──
        # 순서가 중요하다: 먼저 병합해야 '한 충돌의 조각들'이 하나로 합쳐져 길이를
        # 회복하고, 그 뒤 남은 고립된 단발 깜빡임만 길이필터로 제거된다.
        _n_raw = len(events)
        if events:
            events.sort(key=lambda e: e['start_frame'])
            merged = [events[0]]
            for ev in events[1:]:
                gap = ev['start_frame'] - merged[-1]['end_frame']
                if gap <= config.PREDICT_EVENT_MERGE_GAP_FRAMES:
                    merged[-1]['end_frame'] = max(
                        merged[-1]['end_frame'], ev['end_frame'])
                else:
                    merged.append(ev)
            events = [e for e in merged
                      if (e['end_frame'] - e['start_frame'])
                      >= config.PREDICT_MIN_EVENT_SPAN_FRAMES]
        if _n_raw != len(events):
            print(f"[이벤트 후처리] {_n_raw}개 → 병합·길이필터 후 {len(events)}개")

        # ── 2패스: 이벤트 구간 프레임만 영상에서 다시 읽어 CAM 클립 렌더링 ──────
        # 원본 프레임을 RAM에 안 들고, 각 사고 구간만 cap.set으로 탐색해 읽는다.
        results = []
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        render_cap = cv2.VideoCapture(video_path) if events else None
        try:
            for ev_idx, ev in enumerate(events, 1):
                if progress_callback is not None:
                    progress_callback(
                        PROG_INFER_END
                        + (1.0 - PROG_INFER_END) * ((ev_idx - 1) / len(events)))
                start_f = ev['start_frame']
                end_f = ev['end_frame']
                clip_start = max(0, start_f - clip_pad_frames)
                clip_end = min(real_frame_count - 1, end_f + clip_pad_frames)

                # 구간 대표 확률 = 구간 내 사고 프레임 확률의 최댓값
                probs_in_event = [p for f, (_, p) in accident_overlays.items()
                                  if start_f <= f <= end_f]
                crash_prob = max(probs_in_event) if probs_in_event else None

                clip_stem = str(output_dir / f'{base_name}_event{ev_idx}')
                writer, rendered_path = _open_clip_writer(
                    clip_stem, fps, (orig_w, orig_h))


                # 사고 구간 시작 프레임으로 탐색 후 순차 디코딩
                render_cap.set(cv2.CAP_PROP_POS_FRAMES, clip_start)
                last_heatmap = None
                for f in range(clip_start, clip_end + 1):
                    ret, frame = render_cap.read()  # 이미 BGR
                    if not ret:
                        break
                    if f in accident_overlays:
                        last_heatmap = accident_overlays[f][0]
                    in_event = start_f <= f <= end_f
                    if in_event and last_heatmap is not None:
                        roi = frame[by1:by2, bx1:bx2]
                        # 반올림 1px 차이를 흡수하도록 히트맵을 bbox 영역 크기에 정확히 맞춤
                        hm = cv2.resize(last_heatmap, (roi.shape[1], roi.shape[0]))
                        frame[by1:by2, bx1:bx2] = cv2.addWeighted(
                            roi, 0.6, hm, 0.4, 0)
                        cv2.rectangle(frame, (bx1, by1),
                                      (bx2, by2), (0, 0, 255), 3)
                        _draw_state_label(frame, 1)
                    else:
                        cv2.rectangle(frame, (bx1, by1),
                                      (bx2, by2), (0, 255, 0), 2)
                        _draw_state_label(frame, 0)
                    writer.write(frame)
                writer.release()
                # 브라우저 호환 최종본(H.264/mp4)으로 변환
                clip_path = _finalize_clip(rendered_path, clip_stem)

                # ── 사고 '시점' 확정: 이벤트 구간 내 플로우 최대점(=충격 순간) ──
                # 모델은 '윈도우 마지막 프레임=충돌 종료'로 학습돼 윈도우 시작
                # (start_f = F-29)은 체계적으로 이르다. 라벨 검증 결과 플로우
                # 피크가 충돌구간에 5/5 적중(라벨 시작 +3~4프레임)해 가장 정확했다.
                # (naive 상승엣지는 '접근 차량' 모션을 먼저 잡아 최대 -56프레임
                #  빗나가 채택하지 않음)
                # 시점 판정은 '피해차 bbox 한정' 플로우로 한다(없으면 크롭 전체로 폴백)
                _mag_for_impact = (flow_mag_roi if flow_mag_roi is not None
                                   else flow_mag)
                impact_f = start_f
                if _mag_for_impact is not None and end_f >= start_f:
                    seg = _mag_for_impact[
                        start_f:min(end_f + 1, len(_mag_for_impact))]
                    if seg.size:
                        impact_f = start_f + int(np.argmax(seg))

                results.append({
                    # start_* = 사고 발생(충격) 시점 — UI 타임라인/DB 기록용
                    'start_frame': impact_f,
                    'start_sec': impact_f / fps,
                    'end_frame': end_f,
                    'end_sec': end_f / fps,
                    # 참고용: 모델이 A로 판정한 구간 전체(클립 렌더 범위 기준)
                    'event_start_frame': start_f,
                    'event_end_frame': end_f,
                    'crash_prob': crash_prob,
                    'clip_path': clip_path,
                })
        finally:
            if render_cap is not None:
                render_cap.release()

        print(f"[predict_events_and_clips] {len(results)}건 사고구간 클립 생성")
        return results

    finally:
        handle.remove()
