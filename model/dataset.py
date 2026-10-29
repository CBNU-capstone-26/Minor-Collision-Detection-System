import os
import glob
import hashlib
import inspect
import math
from pathlib import Path
import cv2
import numpy as np
import torch
import random
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image

import config


def _open_video(path, threads=None):
    """영상 열기. threads 를 주면 영상 풀기 스레드를 그 수로 제한한다(config.TRAIN_CV2_THREADS —
    로더 작업자 여러 개가 CPU 코어를 나눠 쓸 때). 그렇게 못 열면 기본 방식으로 연다. 풀린 화소는 같다."""
    if threads and hasattr(cv2, 'CAP_PROP_N_THREADS'):
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, int(threads)])
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(path)


def h264_roundtrip(frames, crf, gop):
    """RGB 프레임 목록을 H.264(libx264)로 메모리 안에서 압축했다 풀어 같은 수의 프레임으로 돌려준다.

    실제 CCTV 처럼 키프레임 사이는 앞 프레임과 달라진 곳만 저장해, 가만히 있는 곳은 그대로고 움직이는 곳이 깨진다.
    B 프레임 없음(zerolatency)·스레드 1(로더 작업자가 이미 코어를 나눠 씀). 크기는 짝수여야 한다(yuv420p).
    """
    import io
    import av
    h, w = frames[0].shape[:2]
    buf = io.BytesIO()
    out = av.open(buf, mode='w', format='h264')
    try:
        st = out.add_stream('libx264', rate=30)
        st.width, st.height, st.pix_fmt = w, h, 'yuv420p'
        st.options = {'crf': str(int(crf)), 'preset': 'ultrafast', 'tune': 'zerolatency',
                      'g': str(int(gop)), 'threads': '1'}
        for f in frames:
            for pkt in st.encode(av.VideoFrame.from_ndarray(np.ascontiguousarray(f), format='rgb24')):
                out.mux(pkt)
        for pkt in st.encode():
            out.mux(pkt)
    finally:
        out.close()
    buf.seek(0)
    inp = av.open(buf, mode='r', format='h264')
    try:
        dec = [fr.to_ndarray(format='rgb24') for fr in inp.decode(video=0)]
    finally:
        inp.close()
    if len(dec) != len(frames):
        raise ValueError(f'H.264 복원 프레임 수가 다름({len(dec)} != {len(frames)})')
    return dec


def limit_cv2_threads(n, _worker_id=None):
    """OpenCV 계산(색 변환·크기 줄이기) 스레드 수를 n 으로. 로더를 만들기 전에 부모 프로세스에서 한 번 부르고,
    DataLoader worker_init_fn 으로도 넘긴다(functools.partial(limit_cv2_threads, n)).

    ⚠️ fork 된 작업자 안에서 처음 바꾸면 멈춘다(부모가 OpenCV 스레드를 이미 띄웠을 때 — 2026-10-28 확인).
       그래서 부모에서 먼저 정해 작업자가 물려받게 하고, 작업자에서는 값이 다를 때만(spawn — Windows·macOS) 바꾼다.
    """
    if cv2.getNumThreads() != int(n):
        cv2.setNumThreads(int(n))


class HitAndRunDataset(Dataset):
    def __init__(
        self,
        data_dir=config.DATA_DIR,
        clip_length=config.CLIP_LENGTH,
        r_value=config.R_VALUE,
        resize=config.RESIZE,
        augment=True,
        cache_dir=None,
        span_windows=False,
    ):
        self.data_dir = data_dir
        self.clip_length = clip_length
        self.r_value = r_value
        self.resize = resize
        # augment=True → 학습용(랜덤 증강 적용), False → 검증용(증강 없음, 결정적)
        self.augment = augment
        # 클립 저장본 폴더(검증 전용, config.TRAIN_VAL_CACHE). 증강하면 매번 클립이 달라 쓸 수 없다.
        if augment and cache_dir:
            raise ValueError('클립 저장본은 증강 없는(검증) 데이터셋에만 쓸 수 있습니다')
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache_off = False
        self._lq_off = False         # 저화질 증강이 실패하면(PyAV 없음 등) 이 작업자는 끄고 계속한다
        # 영상 풀기 스레드 수(None = OpenCV 기본). 작업자 쪽 OpenCV 계산 스레드는 train.py 가 맞춘다.
        self.cv2_threads = getattr(config, 'TRAIN_CV2_THREADS', None)
        # A 클립 방식(config.TRAIN_A_MODE). 'span'이면 샘플마다 창 묶음 (K, 3, T, H, W)을
        # 돌려주고(비충돌은 K=1), 로더는 span_collate 로 묶는다.
        a_mode = getattr(config, 'TRAIN_A_MODE', 'pre')
        if a_mode not in ('pre', 'span'):
            raise ValueError(f"TRAIN_A_MODE 는 'pre' 또는 'span' 이어야 합니다: {a_mode!r}")
        self.span = a_mode == 'span'
        # span_windows=True(검증 전용): 'pre' 학습이어도 충돌 샘플을 span 과 같은 창 묶음으로 만든다.
        # 첫 창이 'pre' 의 창과 같아, 기존 지표는 첫 창으로·구간 재현율은 창 전부로 잰다(train.validate).
        # 샘플 수·순서는 그대로다(충돌 영상당 1개). A 슬라이싱(TRAIN_A_SLICE_ENABLED)은 샘플 수가 달라 함께 못 쓴다.
        if span_windows and not self.span and getattr(config, 'TRAIN_A_SLICE_ENABLED', False):
            raise ValueError('span_windows 는 TRAIN_A_SLICE_ENABLED 와 함께 쓸 수 없습니다(학습·검증 샘플 순서가 달라짐)')
        self.span_windows = self.span or bool(span_windows)
        # 정규화 통계는 사전학습(Kinetics-400)과 동일하게 config에서 일괄 관리
        self.mean = torch.tensor(
            config.NORM_MEAN, dtype=torch.float32).view(3, 1, 1, 1)
        self.std = torch.tensor(
            config.NORM_STD, dtype=torch.float32).view(3, 1, 1, 1)

        # data_dir 하위 폴더(rcdata/realdata 등)까지 재귀적으로 mp4를 수집한다.
        # → 최상위에 파일이 있어도(기존 방식) 하위 폴더에 있어도 모두 포함.
        self.mp4_paths = sorted(
            glob.glob(os.path.join(str(data_dir), '**', '*.mp4'), recursive=True))
        self.samples = self._build_index()

    def __len__(self):
        return len(self.samples)

    def _build_index(self):
        samples = []
        for mp4_path in self.mp4_paths:
            # txt는 같은 경로의 확장자만 바꾼 것 (rcdata/realdata 어디든 동일 규칙)
            txt_path = os.path.splitext(mp4_path)[0] + ".txt"
            file_name = os.path.splitext(os.path.basename(mp4_path))[0]
            bboxes, action = self._parse_annotation(txt_path)

            if action is not None:
                class_str = action[0]
                target_id = int(float(action[1]))
                start_f = int(float(action[2]))
                # 충돌 종료 프레임 — A 슬라이싱이 구간을 벗어나지 않도록 쓴다
                end_f = int(float(action[3])) if len(action) > 3 else start_f
            else:
                class_str = 'S'
                target_id = 0
                start_f = 0
                end_f = 0

            if target_id not in bboxes:
                target_id = next(iter(bboxes), 0)
            target_bbox = bboxes.get(
                target_id, [0, 0, self.resize[0], self.resize[1]])
            label = 1 if class_str == 'A' else 0

            # 영상 메타는 한 번만 읽는다(S 슬라이싱의 총프레임 + 지터 환산용 fps).
            total, fps = self._video_meta(mp4_path)

            def _add(sf, wins=None, car=None):
                car = target_id if car is None else car
                sample = {
                    'file_name': file_name,
                    'video_key': mp4_path,   # 분할을 '영상 단위'로 묶기 위한 키
                    'mp4_path': mp4_path,
                    'label': label,
                    'start_f': sf,
                    'bbox': target_bbox if car == target_id else bboxes[car],
                    'car_id': car,           # 크롭 기준 차량 번호
                    'fps': fps,              # 시간 지터를 '초' 기준으로 맞추기 위함
                }
                if wins is not None:
                    sample['win_starts'] = wins  # span: 구간을 덮는 창들의 시작 프레임
                samples.append(sample)

            # S 클립을 만들 차량: 기본은 대상 차량(0번) 하나. 실제 영상 S 는 1번 차량도
            # (config.TRAIN_S_REAL_USE_CAR1). RC S 는 0번만.
            s_cars = [target_id]
            if (label == 0 and getattr(config, 'TRAIN_S_REAL_USE_CAR1', False)
                    and 'realdata' in os.path.normpath(mp4_path).split(os.sep)
                    and 1 in bboxes and target_id != 1):
                s_cars.append(1)

            if label == 1:
                # 충돌(A): 창을 start_f 보다 TRAIN_A_PRE_FRAMES 앞에서 시작해
                # '접근 → 충돌 → 흔들림'이 한 창에 담기게 한다(config 주석 참고).
                pre = int(getattr(config, 'TRAIN_A_PRE_FRAMES', 0))
                a_start = max(0, start_f - pre)
                a_stride = max(1, int(getattr(config, 'TRAIN_A_SLICE_STRIDE', 15)))
                last = end_f - self.clip_length + 1      # 창 끝 == end_f 인 시작점
                if self.span_windows:
                    # span: 접근부터 충돌 종료까지를 덮는 창들을 샘플 하나로 둔다.
                    wins = self._span_starts(a_start, last, total)
                    _add(wins[0], wins)
                elif (getattr(config, 'TRAIN_A_SLICE_ENABLED', False)
                        and last > a_start):
                    # (기본 off) 구간 안에서 창을 밀어 A 클립을 늘리는 경로.
                    # 첫 창만 접근을 담고 나머지는 '충돌 중'이 된다.
                    starts = list(range(a_start, last + 1, a_stride))
                    if starts[-1] != last:
                        starts.append(last)   # 마지막 창은 충돌 종료에 정렬
                    for sf in starts:
                        _add(sf)
                else:
                    _add(a_start)
            elif not getattr(config, 'TRAIN_S_SLICE_ENABLED', False):
                for car in s_cars:
                    _add(start_f, car=car)
            else:
                # 비충돌(S)은 영상 전체를 클립 길이 단위로 잘라 전부 학습에 넣는다.
                # (기존엔 맨 앞 30프레임 1개뿐이라 '차가 지나가지만 충돌 아님'을
                #  학습하지 못했고, 이것이 오탐의 직접 원인이었다 — config 주석 참고)
                stride = max(1, int(getattr(config, 'TRAIN_S_SLICE_STRIDE',
                                            self.clip_length)))
                cap_n = int(getattr(config, 'TRAIN_S_MAX_CLIPS_PER_VIDEO', 0))
                starts = list(range(0, max(1, total - self.clip_length + 1), stride))
                if not starts:
                    starts = [0]
                if cap_n > 0:
                    starts = starts[:cap_n]
                for car in s_cars:
                    for sf in starts:
                        _add(sf, car=car)
        return samples

    def _span_starts(self, a_start, last, total):
        """span 창 시작점들 — 첫 창은 a_start(접근), 마지막 창은 끝이 충돌 종료.

        last : 끝이 충돌 종료에 맞는 창의 시작점. 영상 끝을 넘지 않게 자른다.
        구간이 창 하나에 들어가면(last <= a_start) 창 1개 = 'pre' 방식과 같다.
        그 밖에는 TRAIN_SPAN_WINDOW_STRIDE 이하 간격으로 구간을 고르게 나누고,
        창이 TRAIN_SPAN_MAX_WINDOWS 를 넘으면 간격을 넓혀 그 개수로 덮는다.
        """
        if total > 0:
            last = min(last, total - self.clip_length)
        if last <= a_start:
            return [a_start]
        stride = max(1, int(getattr(config, 'TRAIN_SPAN_WINDOW_STRIDE', 15)))
        max_n = max(2, int(getattr(config, 'TRAIN_SPAN_MAX_WINDOWS', 6)))
        d = last - a_start
        n = min(max_n, 1 + math.ceil(d / stride))
        return [a_start + round(i * d / (n - 1)) for i in range(n)]

    @staticmethod
    def _video_meta(mp4_path):
        """(총 프레임 수, fps). 메타데이터만 읽어 빠르다. 읽기 실패 시 fps는 30 가정."""
        cap = cv2.VideoCapture(mp4_path)
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        cap.release()
        return n, (fps if fps > 1.0 else 30.0)

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

    def _apply_low_quality(self, frames, crop_side):
        """저화질 증강(config.TRAIN_LQ_AUG_PROB, 학습 전용 — 주석은 config). 깨끗한 클립만 H.264 로 압축했다 풀고,
        TRAIN_LQ_SIZE_PROB 비율은 먼저 작게 줄였다 다시 키운다. 값은 클립마다 하나(모든 프레임이 같은 영상)."""
        p = float(getattr(config, 'TRAIN_LQ_AUG_PROB', 0.0) or 0.0)
        if (p <= 0 or self._lq_off or crop_side < getattr(config, 'TRAIN_LQ_MIN_CROP', 128)
                or random.random() >= p):
            return frames
        h, w = frames[0].shape[:2]
        size = None
        if random.random() < getattr(config, 'TRAIN_LQ_SIZE_PROB', 0.5):
            lo, hi = getattr(config, 'TRAIN_LQ_SIZE_RANGE', (48, 128))
            hi = min(hi, int(crop_side * 0.75))      # 원래보다 확실히 작게만
            if hi >= lo:
                size = random.randint(lo // 2, hi // 2) * 2
        crf = random.randint(*getattr(config, 'TRAIN_LQ_CRF_RANGE', (28, 42)))
        gop = random.choice((10, 15, 30, 60))         # CCTV 키프레임 간격(0.3~2초)
        src = [cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames] if size else frames
        try:
            out = h264_roundtrip(src, crf, gop)
        except Exception as e:                        # PyAV·libx264 없음 등 — 이 작업자는 증강 없이 계속
            self._lq_off = True
            print(f'⚠️ 저화질 증강을 쓸 수 없어 끄고 계속합니다: {type(e).__name__}: {e}')
            return frames
        if size:
            out = [cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in out]
        return out

    def _read_frames(self, mp4_path, s0, n_read, bbox, r):
        """s0 부터 n_read 프레임을 읽어 자르고 줄인 uint8 RGB 프레임 목록. 영상이 짧으면 마지막 프레임으로 채운다."""
        cap = _open_video(mp4_path, self.cv2_threads)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, s0)

        for _ in range(n_read):
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(self._crop_and_pad(frame, bbox, r))
        cap.release()

        while len(frames) < n_read:
            frames.append(
                frames[-1] if frames else np.zeros((self.resize[1], self.resize[0], 3), dtype=np.uint8))
        return frames

    # ── 클립 저장본 (검증 전용, config.TRAIN_VAL_CACHE) ──
    # 검증은 증강 없이 매 에포크 같은 클립이라, 자르고 줄인 uint8 프레임을 한 번 저장해 두고 읽는다.
    # 키 = 클립 만드는 코드(_read_frames·_crop_and_pad)·OpenCV 버전 + 영상 파일(경로·크기·수정시각)
    #      + 시작 프레임·길이·박스·r·크기 — 하나라도 바뀌면 다른 파일이 되어 새로 만든다.
    _cache_code = None

    @classmethod
    def _cache_code_key(cls):
        if cls._cache_code is None:
            src = inspect.getsource(cls._read_frames) + inspect.getsource(cls._crop_and_pad)
            cls._cache_code = hashlib.sha1(f'{src}|{cv2.__version__}'.encode()).hexdigest()
        return cls._cache_code

    def _cache_path(self, mp4_path, s0, n_read, bbox, r):
        if self.cache_dir is None or self._cache_off:
            return None
        try:
            st = os.stat(mp4_path)
        except OSError:
            return None
        key = repr((self._cache_code_key(), os.path.abspath(mp4_path), st.st_size, st.st_mtime_ns,
                    int(s0), int(n_read), tuple(int(v) for v in bbox), float(r), tuple(self.resize)))
        return self.cache_dir / f'{hashlib.sha1(key.encode()).hexdigest()}.npy'

    def _cache_load(self, path, n_read):
        if path is None or not path.exists():
            return None
        try:
            arr = np.load(path)
        except (OSError, ValueError):
            return None                       # 읽을 수 없는 파일 — 새로 만들어 덮어쓴다
        if arr.dtype != np.uint8 or arr.shape != (n_read, self.resize[1], self.resize[0], 3):
            return None
        return arr

    def _cache_save(self, path, frames):
        if path is None:
            return
        tmp = path.with_name(f'{path.stem}.{os.getpid()}.tmp')
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(tmp, 'wb') as f:
                np.save(f, np.stack(frames, axis=0))
            os.replace(tmp, path)   # 다 쓴 뒤 이름을 바꿔, 다른 작업자가 쓰다 만 파일을 읽지 않게
        except OSError as e:
            # 디스크가 차는 등 — 이 작업자는 저장 없이 계속한다(검증 결과는 같다)
            self._cache_off = True
            print(f'⚠️ 검증 클립 저장본을 쓰지 못해 저장 없이 계속합니다: {e}')
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def cache_status(self, indices):
        """저장본 상태 (이미 있는 클립 수, 클립 수, 다 만들었을 때 크기 GB) — 학습 시작 때 안내용."""
        have, size = 0, 0
        for i in indices:
            s = self.samples[i]
            starts = list(s.get('win_starts') or [s['start_f']])
            s0, n_read = min(starts), max(starts) - min(starts) + self.clip_length
            p = self._cache_path(s['mp4_path'], s0, n_read, s['bbox'], self.r_value)
            have += bool(p is not None and p.exists())
            size += n_read * self.resize[0] * self.resize[1] * 3
        return have, len(indices), size / 2 ** 30

    def _frames_to_tensor(self, frames):
        arr = np.stack(frames, axis=0).astype(np.float32) / 255.0
        video_tensor = torch.from_numpy(arr).permute(3, 0, 1, 2).contiguous()
        return (video_tensor - self.mean) / self.std

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 창 시작점들. 'pre' 모드와 비충돌 샘플은 창 1개다.
        starts = list(sample.get('win_starts') or [sample['start_f']])
        bbox = sample['bbox']

        if self.augment:
            # ① 시간 오프셋 지터: 학습 클립이 항상 '충돌 시작 = 윈도 첫 프레임'이면
            #    추론(슬라이딩 윈도)에서 충돌이 윈도 중간에 걸릴 때 분포가 어긋난다.
            #    시작점을 조금 앞으로 당겨 충돌 위치를 윈도 내에서 다양화.
            #    ※ 폭을 '초' 기준으로 환산한다. 프레임 수로 고정하면 같은 증강이
            #      fps에 따라 다른 의미가 된다(10프레임 = 30fps 0.33초, 10fps 1.0초).
            #      기본 0.33초는 30fps에서 기존 동작(0~10프레임)과 동일하다.
            #    ※ 양방향(±)으로 흔든다. 한쪽(앞으로만)이면 A 클립의 접근 구간이
            #      TRAIN_A_PRE_FRAMES 보다 계속 길어져 충돌이 창 밖으로 밀린다.
            jitter = int(round(sample.get('fps', 30.0)
                               * getattr(config, 'TRAIN_TIME_JITTER_SEC', 0.33)))
            if jitter > 0:
                # span 창들은 간격을 지킨 채 통째로 민다(맨 앞 창이 0 아래로 가지 않게).
                # 창 1개면 max(0, start_f + d) 와 같다.
                d = max(random.randint(-jitter, jitter), -min(starts))
                starts = [s + d for s in starts]
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

        # r 증강(학습 전용): 클립마다 r을 하나 뽑아 '모든 프레임에 동일 적용'한다.
        # 검증/추론은 항상 self.r_value(=1.0) → 서비스 추론 조건과 일치.
        r = self.r_value
        if (self.augment
                and getattr(config, 'TRAIN_R_AUGMENT_VALUES', ())
                and random.random() < getattr(config, 'TRAIN_R_AUGMENT_PROB', 0.0)):
            r = random.choice(config.TRAIN_R_AUGMENT_VALUES)

        # 첫 창 시작부터 마지막 창 끝까지 한 번에 읽는다(창 1개면 clip_length 프레임).
        # 모든 창이 같은 박스·r·증강을 쓰게 되어, 같은 충돌의 다른 시점일 뿐이 된다.
        s0 = min(starts)
        n_read = max(starts) - s0 + self.clip_length
        cache = self._cache_path(sample['mp4_path'], s0, n_read, bbox, r)
        frames = self._cache_load(cache, n_read)
        if frames is None:
            frames = self._read_frames(sample['mp4_path'], s0, n_read, bbox, r)
            self._cache_save(cache, frames)

        # 증강은 학습 세트에만 적용 (검증 세트는 결정적이어야 val loss가 안정됨)
        if self.augment:
            frames = self._apply_augmentation(frames)
            # 저화질 증강은 색·노이즈 뒤에(찍은 뒤 저장할 때 압축되는 순서). 원본 프레임 크롭 한 변으로 대상을 고른다
            frames = self._apply_low_quality(frames, max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * r)

        video = self._frames_to_tensor(frames)
        label = torch.tensor(sample['label'], dtype=torch.long)
        if not self.span_windows:
            return video, label
        # (K, 3, T, H, W) — 구간을 덮는 창들
        return torch.stack([video[:, s - s0:s - s0 + self.clip_length]
                            for s in starts]), label


def span_collate(batch):
    """span 배치: 샘플마다 창 수가 달라 창들을 이어 붙이고 샘플별 창 수를 함께 준다.

    반환: (창들 (ΣK, 3, T, H, W), 샘플별 창 수 (B,), 라벨 (B,))
    """
    clips = torch.cat([b[0] for b in batch])
    sizes = torch.tensor([b[0].shape[0] for b in batch], dtype=torch.long)
    labels = torch.stack([b[1] for b in batch])
    return clips, sizes, labels
