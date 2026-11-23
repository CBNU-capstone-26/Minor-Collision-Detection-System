import os
import csv
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
    """영상을 엽니다. threads를 주면 영상 풀기 스레드를 그 수로 제한합니다(config.TRAIN_CV2_THREADS,
    로더 작업자 여러 개가 CPU 코어를 나눠 쓸 때). 그렇게 열지 못하면 기본 방식으로 열며, 풀린 화소는 같습니다.
    """
    if threads and hasattr(cv2, 'CAP_PROP_N_THREADS'):
        cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, int(threads)])
        if cap.isOpened():
            return cap
        cap.release()
    return cv2.VideoCapture(path)


def h264_roundtrip(frames, crf, gop):
    """RGB 프레임 목록을 H.264(libx264)로 메모리 안에서 압축했다 풀어 같은 수의 프레임으로 돌려줍니다.

    실제 CCTV처럼 키프레임 사이에는 앞 프레임과 달라진 곳만 저장해, 가만히 있는 곳은 그대로고 움직이는 곳이 깨집니다.
    B 프레임 없이(zerolatency) 스레드 하나로 돌립니다(로더 작업자가 이미 코어를 나눠 씁니다). 크기는 짝수여야 합니다(yuv420p).
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
        # 푸는 쪽도 스레드를 하나만 씁니다. 기본값(자동)은 보이는 CPU 수만큼 스레드를 만들어, CPU가 많이 보이는 컨테이너에서는
        # 작업자 수십 개가 동시에 수백 개씩 만들어 할당된 코어를 넘을 수 있습니다.
        inp.streams.video[0].thread_count = 1
        dec = [fr.to_ndarray(format='rgb24') for fr in inp.decode(video=0)]
    finally:
        inp.close()
    if len(dec) != len(frames):
        raise ValueError(f'H.264 복원 프레임 수가 다름({len(dec)} != {len(frames)})')
    return dec


LIGHT_COLORS = {'white': (255, 255, 255), 'warm': (255, 236, 200), 'cool': (215, 232, 255), 'red': (255, 60, 50)}


def add_light(frames, kind, dark=1.0, peak=150.0, radius=0.35, t0=8, dur=14, direction=1, color='white', y=0.55):
    """크롭 프레임(RGB uint8, 같은 크기)에 움직이는 불빛을 더합니다. 불빛 증강과 robust_check 불빛 변형이 함께 씁니다.

    dark: 먼저 곱할 밝기 배율(밤처럼, 1이면 그대로)
    kind: 'sweep'은 타원 빛 덩어리가 t0부터 dur 프레임 동안 한쪽 끝에서 반대쪽으로 지나갑니다(지나가는 차 전조등).
          'ramp'는 t0부터 dur 프레임 동안 서서히 밝아져 그대로 유지되며, 한쪽이 더 밝습니다(다가오는 차가 비춤).
    peak: 더하는 밝기 최대(0~255), radius: 빛 덩어리 가로 반지름(크롭 한 변 대비), y: 빛 중심 높이(크롭 대비)
    direction: 1이면 왼쪽에서 오른쪽(ramp는 왼쪽이 밝음), -1이면 반대입니다. color: LIGHT_COLORS의 키
    빛 모양은 프레임마다 매끄럽게 이어지고 차체 윤곽은 그대로라, 밝기 변화와 흔들림이 다르다는 것을 가르칩니다.
    """
    n = len(frames)
    h, w = frames[0].shape[:2]
    col = np.array(LIGHT_COLORS.get(color, LIGHT_COLORS['white']), np.float32) / 255.0
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    sx, sy = max(2.0, radius * w), max(2.0, radius * w * 0.6)
    # 크롭이 화면 밖으로 나가 검게 채운 곳(모든 프레임에서 0)은 장면이 아니므로 빛을 받지 않습니다.
    keep = (~np.all(np.stack(frames) == 0, axis=(0, 3)))[..., None].astype(np.float32)
    out = []
    for t, f in enumerate(frames):
        img = f.astype(np.float32) * dark
        u = (t - t0) / max(1, dur)
        if kind == 'sweep' and 0.0 <= u <= 1.0:
            cx = (-sx + u * (w + 2 * sx)) if direction > 0 else (w + sx - u * (w + 2 * sx))
            g = np.exp(-(((xx - cx) / sx) ** 2 + ((yy - y * h) / sy) ** 2) / 2.0)
            amp = peak * math.sin(math.pi * u) ** 0.5           # 들어오고 나갈 때 부드럽게
            img = img + amp * g[..., None] * col * keep
        elif kind == 'ramp' and u >= 0.0:
            s = min(1.0, u)
            s = s * s * (3 - 2 * s)                               # 부드럽게 밝아짐
            grad = (xx / max(1, w - 1)) if direction < 0 else (1.0 - xx / max(1, w - 1))
            img = img + peak * s * (0.35 + 0.65 * grad)[..., None] * col * keep
        out.append(np.clip(img, 0, 255).astype(np.uint8))
    return out


def limit_cv2_threads(n, _worker_id=None):
    """OpenCV 계산(색 변환, 크기 줄이기) 스레드 수를 n으로 정합니다. 로더를 만들기 전에 부모 프로세스에서 한 번 부르고,
    DataLoader worker_init_fn으로도 넘깁니다(functools.partial(limit_cv2_threads, n)).

    주의: fork된 작업자 안에서 처음 바꾸면 멈춥니다(부모가 OpenCV 스레드를 이미 띄웠을 때).
    그래서 부모에서 먼저 정해 작업자가 물려받게 하고, 작업자에서는 값이 다를 때만(spawn 방식인 Windows, macOS) 바꿉니다.
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
        # augment=True면 학습용(무작위 증강), False면 검증용(증강 없음, 결정적)입니다.
        self.augment = augment
        # 클립 저장본 폴더(검증 전용, config.TRAIN_VAL_CACHE)입니다. 증강하면 클립이 매번 달라 쓸 수 없습니다.
        if augment and cache_dir:
            raise ValueError('클립 저장본은 증강 없는(검증) 데이터셋에만 쓸 수 있습니다')
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self._cache_off = False
        self._lq_off = False         # 저화질 증강이 실패하면(PyAV 없음 등) 이 작업자는 끄고 계속합니다.
        # 영상 풀기 스레드 수(None은 OpenCV 기본값)입니다. 작업자 쪽 OpenCV 계산 스레드는 train.py가 맞춥니다.
        self.cv2_threads = getattr(config, 'TRAIN_CV2_THREADS', None)
        # 사고 클립 방식(config.TRAIN_A_MODE)입니다. 'span'이면 샘플마다 묶음 여러 개 (K, 3, T, H, W)를 돌려주고
        # (정상은 K=1), 로더는 span_collate로 묶습니다.
        a_mode = getattr(config, 'TRAIN_A_MODE', 'pre')
        if a_mode not in ('pre', 'span'):
            raise ValueError(f"TRAIN_A_MODE 는 'pre' 또는 'span' 이어야 합니다: {a_mode!r}")
        self.span = a_mode == 'span'
        # span_windows=True(검증 전용)면 'pre' 학습이어도 충돌 샘플을 span과 같은 묶음 여러 개로 만듭니다.
        # 첫 묶음이 'pre'의 묶음과 같아, 기존 지표는 첫 묶음으로, 구간 재현율은 묶음 전부로 잽니다(train.validate).
        # 샘플 수와 순서는 그대로입니다(사고 영상마다 하나). A 슬라이싱(TRAIN_A_SLICE_ENABLED)은 샘플 수가 달라 함께 쓸 수 없습니다.
        if span_windows and not self.span and getattr(config, 'TRAIN_A_SLICE_ENABLED', False):
            raise ValueError('span_windows 는 TRAIN_A_SLICE_ENABLED 와 함께 쓸 수 없습니다(학습·검증 샘플 순서가 달라짐)')
        self.span_windows = self.span or bool(span_windows)
        # 정규화 통계는 사전학습(Kinetics-400)과 같도록 config에서 함께 관리합니다.
        self.mean = torch.tensor(
            config.NORM_MEAN, dtype=torch.float32).view(3, 1, 1, 1)
        self.std = torch.tensor(
            config.NORM_STD, dtype=torch.float32).view(3, 1, 1, 1)
        # 같은 값의 numpy float32입니다. _frames_to_tensor가 배열 하나에서 제자리로 정규화할 때 씁니다.
        self._mean_np = np.asarray(config.NORM_MEAN, dtype=np.float32).reshape(3, 1, 1, 1)
        self._std_np = np.asarray(config.NORM_STD, dtype=np.float32).reshape(3, 1, 1, 1)

        # data_dir 아래 하위 폴더(rcdata, realdata 등)까지 찾아 mp4를 모읍니다. 최상위에 있어도 하위 폴더에 있어도 모두 포함합니다.
        # data_dir은 폴더 여러 개(목록)여도 됩니다(val_recheck가 검증 폴더와 학습 폴더를 함께 읽을 때).
        dirs = data_dir if isinstance(data_dir, (list, tuple)) else [data_dir]
        self.mp4_paths = sorted(
            p for d in dirs for p in glob.glob(os.path.join(str(d), '**', '*.mp4'), recursive=True))
        self.samples = self._build_index()

    def __len__(self):
        return len(self.samples)

    def _build_index(self):
        samples = []
        # 사고 영상의 충돌 전후 정상 클립입니다(config.TRAIN_A_CONTEXT_NEG, 학습 데이터셋만).
        # 맨 뒤에 붙여 앞쪽 샘플 번호를 검증 데이터셋(이 클립을 만들지 않음)과 같게 둡니다. 검증 폴더가 없을 때 두 데이터셋을 같은 번호로 나누기 때문입니다.
        ctx_mode = getattr(config, 'TRAIN_A_CONTEXT_NEG', None) if self.augment else None
        ctx_samples = []
        # 헷갈리는 정상 묶음(config.TRAIN_HARDNEG_CSV, 학습 데이터셋만)도 같은 이유로 맨 뒤에 붙입니다.
        hn_map = self._load_hardneg() if self.augment else {}
        hn_samples = []
        for mp4_path in self.mp4_paths:
            # txt는 같은 경로에서 확장자만 바꾼 파일입니다.
            txt_path = os.path.splitext(mp4_path)[0] + ".txt"
            file_name = os.path.splitext(os.path.basename(mp4_path))[0]
            bboxes, action = self._parse_annotation(txt_path)

            if action is not None:
                class_str = action[0]
                target_id = int(float(action[1]))
                start_f = int(float(action[2]))
                # 충돌 끝 프레임입니다. A 슬라이싱이 구간을 벗어나지 않게 할 때 씁니다.
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

            # 영상 정보는 한 번만 읽습니다(S 슬라이싱의 총 프레임 수와 흔들기 폭 환산용 fps).
            total, fps = self._video_meta(mp4_path)
            hn_samples += self._hardneg_samples(hn_map.get(file_name, ()), mp4_path, file_name, bboxes, fps, total)

            def _add(sf, wins=None, car=None):
                car = target_id if car is None else car
                sample = {
                    'file_name': file_name,
                    'video_key': mp4_path,   # 분할을 영상 단위로 묶기 위한 키
                    'mp4_path': mp4_path,
                    'label': label,
                    'start_f': sf,
                    'bbox': target_bbox if car == target_id else bboxes[car],
                    'car_id': car,           # 크롭 기준 차량 번호
                    'fps': fps,              # 시간 흔들기를 초 기준으로 맞추기 위한 fps
                    'total': total,          # fps 증강이 k배로 읽을 수 있는지 확인용
                }
                if wins is not None:
                    sample['win_starts'] = wins  # span: 구간을 덮는 묶음들의 시작 프레임
                samples.append(sample)

            # 정상 클립을 만들 차량입니다. 기본은 대상 차량(0번) 하나이고, 실제 영상은 1번 차량도 씁니다(config.TRAIN_S_REAL_USE_CAR1).
            # RC는 기본 0번만 쓰고, TRAIN_S_RC_USE_CAR1이면 1번도 씁니다(학습 쪽만 쓰고 검증에서는 train.py가 뺍니다).
            s_cars = [target_id]
            is_real = 'realdata' in os.path.normpath(mp4_path).split(os.sep)
            if (label == 0 and 1 in bboxes and target_id != 1
                    and getattr(config, 'TRAIN_S_REAL_USE_CAR1' if is_real else 'TRAIN_S_RC_USE_CAR1', False)):
                s_cars.append(1)

            if label == 1:
                # 사고(A): 묶음을 start_f보다 TRAIN_A_PRE_FRAMES 앞에서 시작해 접근, 충돌, 흔들림이 한 묶음에 담기게 합니다(config 주석 참고).
                pre = int(getattr(config, 'TRAIN_A_PRE_FRAMES', 0))
                a_start = max(0, start_f - pre)
                a_stride = max(1, int(getattr(config, 'TRAIN_A_SLICE_STRIDE', 15)))
                last = end_f - self.clip_length + 1      # 묶음 끝이 end_f인 시작점
                if self.span_windows:
                    # span: 접근부터 충돌 끝까지를 덮는 묶음들을 샘플 하나로 둡니다.
                    wins = self._span_starts(a_start, last, total)
                    _add(wins[0], wins)
                elif (getattr(config, 'TRAIN_A_SLICE_ENABLED', False)
                        and last > a_start):
                    # 구간 안에서 묶음을 밀어 사고 클립을 늘리는 경로입니다(기본 끔). 첫 묶음만 접근을 담고 나머지는 충돌 중 장면이 됩니다.
                    starts = list(range(a_start, last + 1, a_stride))
                    if starts[-1] != last:
                        starts.append(last)   # 마지막 묶음은 충돌 끝에 맞춥니다.
                    for sf in starts:
                        _add(sf)
                else:
                    _add(a_start)
                if ctx_mode and (is_real or ctx_mode == 'all'):
                    ctx_samples += self._context_neg(mp4_path, file_name, target_id, target_bbox,
                                                     fps, total, start_f, end_f)
            elif not getattr(config, 'TRAIN_S_SLICE_ENABLED', False):
                for car in s_cars:
                    _add(start_f, car=car)
            else:
                # 정상(S)은 영상 전체를 클립 길이 단위로 잘라 모두 학습에 넣습니다. 차가 지나가지만 충돌은 아닌 장면을 배우게 하려는 것입니다.
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
        return samples + ctx_samples + hn_samples

    @staticmethod
    def _load_hardneg():
        """config.TRAIN_HARDNEG_CSV를 읽어 {영상 이름: [(차 번호, 묶음 시작)]}을 돌려줍니다. 끄면 {}입니다."""
        path = getattr(config, 'TRAIN_HARDNEG_CSV', None)
        if not path:
            return {}
        out = {}
        with open(path, encoding='utf-8') as f:
            for r in csv.DictReader(f):
                out.setdefault(r['video'], []).append((int(r['car_id']), int(r['start_f'])))
        return out

    def _hardneg_samples(self, items, mp4_path, file_name, bboxes, fps, total):
        """헷갈리는 정상 묶음(config.TRAIN_HARDNEG_CSV 주석 참고)을 정상 클립으로 TRAIN_HARDNEG_REPEAT번 만듭니다."""
        L = self.clip_length
        j = int(getattr(config, 'TRAIN_HARDNEG_JITTER', 5))
        rep = int(getattr(config, 'TRAIN_HARDNEG_REPEAT', 1))
        out = []
        for car, w in items:
            if car not in bboxes or (total > 0 and w > total - L):
                continue
            hi = w + j if total <= 0 else min(w + j, total - L)
            for _ in range(rep):
                out.append({'file_name': file_name, 'video_key': mp4_path, 'mp4_path': mp4_path, 'label': 0,
                            'start_f': w, 'bbox': bboxes[car], 'car_id': car, 'fps': fps, 'total': total,
                            'hardneg': True, 'start_range': (max(0, w - j), hi)})
        return out

    def _context_neg(self, mp4_path, file_name, car_id, bbox, fps, total, start_f, end_f):
        """사고 영상의 충돌 전후 정상 클립을 만듭니다(config.TRAIN_A_CONTEXT_NEG 주석 참고).
        충돌 전은 묶음 끝이 충돌 시작 - 여유 이하, 충돌 후는 묶음 시작이 충돌 끝 + 여유 이상입니다.
        한쪽에 최대 2개로, 충돌에 가장 가까운 묶음 하나와, 그보다 간격 이상 먼 곳에서 가져올 때마다 무작위로 고르는 묶음 하나('rand_range')입니다.
        묶음 위치 흔들기는 'start_range' 밖으로 나가지 않습니다.
        """
        L = self.clip_length
        m = max(1, int(round(float(getattr(config, 'TRAIN_A_CONTEXT_NEG_MARGIN_SEC', 0.5)) * fps)))
        step = max(1, int(getattr(config, 'TRAIN_A_CONTEXT_NEG_STRIDE', L)))
        out = []

        def _add(side, w, lo, hi, rand=None):
            s = {'file_name': file_name, 'video_key': mp4_path, 'mp4_path': mp4_path, 'label': 0,
                 'start_f': w, 'bbox': bbox, 'car_id': car_id, 'fps': fps, 'total': total,
                 'ctx_neg': side, 'start_range': (lo, hi)}
            if rand is not None:
                s['rand_range'] = rand
            out.append(s)

        pre_hi = start_f - m - L + 1          # 끝이 충돌 시작 - 여유인 묶음(가장 가까움)
        if pre_hi >= 0:
            if pre_hi - step >= 0:
                _add('pre', pre_hi - step, 0, pre_hi, rand=(0, pre_hi - step))
            _add('pre', pre_hi, 0, pre_hi)
        post_lo, post_hi = end_f + m, (total - L if total > 0 else -1)
        if post_lo <= post_hi:
            _add('post', post_lo, post_lo, post_hi)
            if post_lo + step <= post_hi:
                _add('post', post_lo + step, post_lo, post_hi, rand=(post_lo + step, post_hi))
        return out

    def _span_starts(self, a_start, last, total):
        """span 묶음 시작점들을 돌려줍니다. 첫 묶음은 a_start(접근)에서 시작하고, 마지막 묶음은 끝이 충돌 끝입니다.

        last는 끝이 충돌 끝에 맞는 묶음의 시작점이며, 영상 끝을 넘지 않게 자릅니다.
        구간이 묶음 하나에 들어가면(last <= a_start) 묶음 하나로 'pre' 방식과 같습니다.
        그 밖에는 TRAIN_SPAN_WINDOW_STRIDE 이하 간격으로 구간을 고르게 나누고,
        묶음이 TRAIN_SPAN_MAX_WINDOWS를 넘으면 간격을 넓혀 그 개수로 덮습니다.
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
        """(총 프레임 수, fps)를 돌려줍니다. 영상 정보만 읽어 빠르며, 읽지 못하면 fps는 30으로 봅니다."""
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
        # 실제 주차장 CCTV 환경에 맞춘 데이터 증강입니다. 세기와 확률은 config의 '기본 영상 증강 세기'를 따릅니다.
        # 변환 값은 클립마다 한 번 뽑아 모든 프레임에 똑같이 적용해 시간적 일관성을 지킵니다.
        # 노이즈도 클립 내내 같은 잔무늬입니다. 프레임마다 새로 뽑으면 화소가 깜빡여 미세한 흔들림을 덮기 때문입니다.

        do_hflip = random.random() > 0.5

        # 밝기, 대비, 채도, 색조: 조명 변화, 날씨, 카메라 색감 차이를 흉내 냅니다.
        do_jitter = random.random() > 0.5
        brightness = random.uniform(*getattr(config, 'TRAIN_AUG_BRIGHTNESS', (0.4, 1.4)))   # 야간, 지하주차장부터 밝은 바깥까지
        contrast = random.uniform(0.7, 1.3)   # 흐린 날의 낮은 대비부터 직사광의 높은 대비까지
        saturation = random.uniform(0.7, 1.3)   # 카메라마다 다른 채도
        hue = random.uniform(-0.1, 0.1)  # 카메라 화이트밸런스와 조명 색온도 차이

        # 야간 적외선(IR) CCTV는 사실상 무채색이라, 색 없이도 판별하도록 학습합니다.
        do_gray = random.random() < getattr(config, 'TRAIN_AUG_GRAY_PROB', 0.2)
        # 저화질 CCTV의 초점 흐림, 압축 열화(블러), 야간 게인으로 생긴 잔무늬(압축돼 멈춰 있는 잡음)를 흉내 냅니다.
        do_blur = random.random() < 0.2
        do_noise = random.random() < getattr(config, 'TRAIN_AUG_NOISE_PROB', 0.2)
        noise_sigma = random.uniform(*getattr(config, 'TRAIN_AUG_NOISE_SIGMA', (2.0, 8.0)))
        grain = None
        if do_noise:
            # 클립 내내 같은 잔무늬입니다. 흑백 클립은 밝기 잡음만 넣습니다(색 잡음이 끼면 적외선 영상답지 않습니다).
            h, w = frames[0].shape[:2]
            grain = np.random.normal(0.0, noise_sigma, (h, w, 1 if do_gray else 3)).astype(np.float32)

        # 계산 순서는 뒤집기, 밝기와 대비와 채도와 색조, 흑백, 흐림, 잔무늬입니다.
        # PIL 연산이 없는 클립은 PIL을 거치지 않고(뒤집기는 numpy로, 같은 화소), 흐림은 30프레임을 한 번에 합성곱해 빠르게 처리합니다.
        use_pil = do_jitter or do_gray
        aug_frames = []
        for frame in frames:
            if use_pil:
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
                aug_frames.append(np.array(img))
            else:
                aug_frames.append(np.ascontiguousarray(frame[:, ::-1]) if do_hflip else frame)
        if do_blur:
            batch = torch.from_numpy(np.stack(aug_frames)).permute(0, 3, 1, 2)    # (T, 3, H, W) uint8
            batch = TF.gaussian_blur(batch, kernel_size=3)                         # 프레임별 PIL 경로와 같은 커널과 시그마(0.8)
            aug_frames = list(batch.permute(0, 2, 3, 1).contiguous().numpy())
        if grain is not None:
            aug_frames = [np.clip(arr.astype(np.float32) + grain, 0, 255).astype(np.uint8) for arr in aug_frames]

        return aug_frames

    def _apply_light(self, frames):
        """불빛 증강입니다(config.TRAIN_LIGHT_AUG_PROB, 학습 전용, 설명은 config 참고). 사고와 정상 구분 없이 같은 확률로,
        클립마다 훑기나 비추기 중 하나와 세기, 크기, 방향, 색, 시점을 하나씩 뽑습니다. 확률이 0이면 난수를 쓰지 않습니다.
        """
        p = float(getattr(config, 'TRAIN_LIGHT_AUG_PROB', 0.0) or 0.0)
        if p <= 0 or random.random() >= p:
            return frames
        n = len(frames)
        dur = random.randint(6, 30)
        dark = random.uniform(0.25, 0.6) if random.random() < getattr(config, 'TRAIN_LIGHT_DARK_PROB', 0.7) else 1.0
        return add_light(frames, 'sweep' if random.random() < 0.6 else 'ramp', dark=dark,
                         peak=random.uniform(60, 200), radius=random.uniform(0.2, 0.6),
                         t0=random.randint(-dur // 2, n - dur // 2), dur=dur, direction=random.choice((1, -1)),
                         color=random.choices(('white', 'warm', 'cool', 'red'), weights=(4, 3, 2, 1))[0],
                         y=random.uniform(0.3, 0.8))

    def _apply_low_quality(self, frames, crop_side):
        """저화질 증강입니다(config.TRAIN_LQ_AUG_PROB, 학습 전용, 설명은 config 참고). 깨끗한 클립만 H.264로 압축했다 풀고,
        TRAIN_LQ_SIZE_PROB 비율은 먼저 작게 줄였다 다시 키웁니다. 값은 클립마다 하나입니다(모든 프레임이 같은 영상).
        TRAIN_LQ_SMALL이면 작은 크롭(TRAIN_LQ_SMALL_MIN 이상 TRAIN_LQ_MIN_CROP 미만)도 원래 크기로 줄여 압축합니다.
        """
        p = float(getattr(config, 'TRAIN_LQ_AUG_PROB', 0.0) or 0.0)
        small = crop_side < getattr(config, 'TRAIN_LQ_MIN_CROP', 128)
        if (p <= 0 or self._lq_off
                or (small and (not getattr(config, 'TRAIN_LQ_SMALL', False)
                               or crop_side < getattr(config, 'TRAIN_LQ_SMALL_MIN', 32)))
                or random.random() >= p):
            return frames
        h, w = frames[0].shape[:2]
        size = None
        c_lo, c_hi = getattr(config, 'TRAIN_LQ_CRF_RANGE', (28, 42))
        if small:
            # 작은 차는 이미 키운 크롭이라 224에서 압축하면 실제보다 약하므로, 원래 크기(짝수, 16 이상)로 되돌려 압축합니다.
            size = max(16, int(round(crop_side / 2)) * 2)
            crf = random.randint(c_lo, max(c_lo, min(c_hi, getattr(config, 'TRAIN_LQ_SMALL_CRF_MAX', 36))))
        else:
            if random.random() < getattr(config, 'TRAIN_LQ_SIZE_PROB', 0.5):
                lo, hi = getattr(config, 'TRAIN_LQ_SIZE_RANGE', (48, 128))
                hi = min(hi, int(crop_side * 0.75))      # 원래보다 확실히 작게만 줄입니다.
                if hi >= lo:
                    size = random.randint(lo // 2, hi // 2) * 2
            crf = random.randint(c_lo, c_hi)
        gop = random.choice((10, 15, 30, 60))         # CCTV 키프레임 간격(0.3~2초)
        src = [cv2.resize(f, (size, size), interpolation=cv2.INTER_AREA) for f in frames] if size else frames
        try:
            out = h264_roundtrip(src, crf, gop)
        except Exception as e:                        # PyAV나 libx264가 없으면 이 작업자는 증강 없이 계속합니다.
            self._lq_off = True
            print(f'⚠️ 저화질 증강을 쓸 수 없어 끄고 계속합니다: {type(e).__name__}: {e}')
            return frames
        if size:
            out = [cv2.resize(f, (w, h), interpolation=cv2.INTER_LINEAR) for f in out]
        return out

    def _read_frames(self, mp4_path, s0, n_read, bbox, r):
        """s0부터 n_read 프레임을 읽어 자르고 줄인 uint8 RGB 프레임 목록을 돌려줍니다. 영상이 짧으면 마지막 프레임으로 채웁니다."""
        cap = _open_video(mp4_path, self.cv2_threads)
        frames = []
        cap.set(cv2.CAP_PROP_POS_FRAMES, s0)

        for _ in range(n_read):
            ret, frame = cap.read()
            if not ret:
                break
            # 색 변환(BGR에서 RGB)은 채널 순서만 바꾸고 자르기, 여백, 줄이기는 채널마다 같은 계산이라, 자른 뒤에 해도 화소가 같습니다.
            # 전체 화면 대신 224 크롭만 바꿔 클립마다 시간을 줄입니다.
            frames.append(cv2.cvtColor(self._crop_and_pad(frame, bbox, r), cv2.COLOR_BGR2RGB))
        cap.release()

        while len(frames) < n_read:
            frames.append(
                frames[-1] if frames else np.zeros((self.resize[1], self.resize[0], 3), dtype=np.uint8))
        return frames

    # 클립 저장본(검증 전용, config.TRAIN_VAL_CACHE)
    # 검증은 증강 없이 매 에포크 같은 클립이라, 자르고 줄인 uint8 프레임을 한 번 저장해 두고 읽습니다.
    # 키는 클립 만드는 코드(_read_frames, _crop_and_pad), OpenCV 버전, 영상 파일(경로, 크기, 수정 시각),
    # 시작 프레임, 길이, 박스, r, 크기입니다. 하나라도 바뀌면 다른 파일이 되어 새로 만듭니다.
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
            return None                       # 읽을 수 없는 파일이면 새로 만들어 덮어씁니다.
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
            os.replace(tmp, path)   # 다 쓴 뒤 이름을 바꿔, 다른 작업자가 쓰다 만 파일을 읽지 않게 합니다.
        except OSError as e:
            # 디스크가 차는 등의 경우 이 작업자는 저장 없이 계속합니다(검증 결과는 같습니다).
            self._cache_off = True
            print(f'⚠️ 검증 클립 저장본을 쓰지 못해 저장 없이 계속합니다: {e}')
            try:
                tmp.unlink(missing_ok=True)
            except OSError:
                pass

    def cache_status(self, indices):
        """저장본 상태(이미 있는 클립 수, 클립 수, 다 만들었을 때 크기 GB)를 돌려줍니다. 학습 시작 때 안내용입니다."""
        have, size = 0, 0
        for i in indices:
            s = self.samples[i]
            starts = list(s.get('win_starts') or [s['start_f']])
            s0, n_read = min(starts), max(starts) - min(starts) + self.clip_length
            p = self._cache_path(s['mp4_path'], s0, n_read, s['bbox'], self.r_value)
            have += bool(p is not None and p.exists())
            size += n_read * self.resize[0] * self.resize[1] * 3
        return have, len(indices), size / 2 ** 30

    def _fps_aug_stride(self, sample, starts):
        """fps 증강(config.TRAIN_FPS_AUG_PROB, 학습 전용, 설명은 config 참고)으로 몇 장마다 한 장 고를지 정합니다. 하지 않으면 1입니다.
        확률이 0이면 난수를 쓰지 않습니다.
        """
        p = float(getattr(config, 'TRAIN_FPS_AUG_PROB', 0.0) or 0.0)
        # 충돌 전후 정상 클립과 헷갈리는 정상 묶음은 k배로 읽으면 묶음이 충돌 구간까지 늘어나므로 걸지 않습니다(난수도 쓰지 않음).
        if p <= 0 or len(starts) != 1 or sample.get('ctx_neg') or sample.get('hardneg') or random.random() >= p:
            return 1
        fps = float(sample.get('fps', 30.0))
        ks = [k for k in getattr(config, 'TRAIN_FPS_AUG_STRIDES', (2, 3))
              if k > 1 and fps / k >= getattr(config, 'TRAIN_FPS_AUG_MIN_FPS', 8.0)]
        if not ks:
            return 1
        k = random.choice(ks)
        total = int(sample.get('total', 0) or 0)
        if total and total < (self.clip_length - 1) * k + 1:
            return 1                           # 영상이 짧아 k배로 읽지 못하면 그대로 둡니다.
        return k

    def _read_strided(self, sample, start, k, bbox, r):
        """start 근처에서 (clip_length - 1) * k + 1 프레임을 읽어 k장마다 한 장씩 골라 clip_length장을 만듭니다.
        사고 클립은 충돌 시작이 원래처럼 TRAIN_A_PRE_FRAMES번째에 오도록 앞쪽을 k배로 늘립니다.
        """
        n = (self.clip_length - 1) * k + 1
        s0 = start
        if sample['label'] == 1:
            s0 = start - int(getattr(config, 'TRAIN_A_PRE_FRAMES', 0)) * (k - 1)
        total = int(sample.get('total', 0) or 0)
        if total:
            s0 = min(s0, total - n)            # 영상 끝을 넘지 않게 합니다.
        s0 = max(0, s0)
        return self._read_frames(sample['mp4_path'], s0, n, bbox, r)[::k]

    def _frames_to_tensor(self, frames):
        """uint8 RGB 프레임 목록을 정규화한 (3, T, H, W) float32로 바꿉니다.
        255로 나누고, 평균을 빼고, 표준편차로 나누는 계산을 배열 하나에서 제자리로 해 임시 배열과 복사를 줄입니다.
        """
        out = np.empty((3, len(frames)) + frames[0].shape[:2], dtype=np.float32)
        for t, f in enumerate(frames):
            np.copyto(out[:, t], f.transpose(2, 0, 1), casting='unsafe')     # uint8에서 float32로 바꿔도 값은 정확히 같습니다.
        np.divide(out, np.float32(255.0), out=out)
        np.subtract(out, self._mean_np, out=out)
        np.divide(out, self._std_np, out=out)
        return torch.from_numpy(out)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        # 묶음 시작점들입니다. 'pre' 방식과 정상 샘플은 묶음이 하나입니다.
        starts = list(sample.get('win_starts') or [sample['start_f']])
        bbox = sample['bbox']

        if self.augment:
            if 'rand_range' in sample:
                # 충돌 전후 정상의 먼 쪽 클립은 가져올 때마다 범위 안에서 새로 고릅니다(config.TRAIN_A_CONTEXT_NEG).
                starts = [random.randint(*sample['rand_range'])]
            # 1) 시간 오프셋 흔들기: 학습 클립이 늘 같은 위치에서 시작하면, 추론(슬라이딩 윈도우)에서 충돌이 묶음 가운데 걸릴 때 분포가 어긋납니다.
            #    시작점을 조금씩 옮겨 충돌 위치를 묶음 안에서 다양하게 합니다.
            #    폭은 초로 환산합니다. 프레임 수로 고정하면 같은 증강이 fps에 따라 다른 의미가 되기 때문입니다.
            #    앞뒤 양쪽으로 흔듭니다. 앞으로만 흔들면 사고 클립의 접근 구간이 계속 길어져 충돌이 묶음 밖으로 밀립니다.
            jitter = int(round(sample.get('fps', 30.0)
                               * getattr(config, 'TRAIN_TIME_JITTER_SEC', 0.33)))
            if jitter > 0:
                # span 묶음들은 간격을 지킨 채 통째로 밉니다(맨 앞 묶음이 0 아래로 가지 않게). 묶음이 하나면 max(0, start_f + d)와 같습니다.
                d = max(random.randint(-jitter, jitter), -min(starts))
                starts = [s + d for s in starts]
                if 'start_range' in sample:
                    # 충돌 전후 정상 클립은 정한 범위 밖(충돌 쪽)으로 밀리지 않게 합니다(config.TRAIN_A_CONTEXT_NEG).
                    lo, hi = sample['start_range']
                    starts = [min(max(s, lo), hi) for s in starts]
            # 2) 박스 흔들기: 서비스에서는 사용자가 마우스로 대충 박스를 그립니다.
            #    레이블 좌표 그대로만 학습하면 손으로 그린 박스와 분포가 어긋나므로,
            #    중심 이동(5% 안)과 크기 배율(0.9~1.15)로 부정확한 박스를 흉내 냅니다.
            x1, y1, x2, y2 = bbox
            bw, bh = x2 - x1, y2 - y1
            cx = (x1 + x2) // 2 + int(bw * random.uniform(-0.05, 0.05))
            cy = (y1 + y2) // 2 + int(bh * random.uniform(-0.05, 0.05))
            s = random.uniform(0.9, 1.15)
            hw, hh = int(bw * s / 2), int(bh * s / 2)
            bbox = [cx - hw, cy - hh, cx + hw, cy + hh]

        # 크롭 배율입니다. 검증과 서비스는 self.r_value(config.R_VALUE, 1.0)를 씁니다. 학습은 r 증강(config.TRAIN_R_AUGMENT_*)으로
        # 일부 클립만 넓게 자르며, 그 클립의 모든 프레임에 같은 r을 씁니다. 확률이 0이면 난수를 쓰지 않습니다.
        r = self.r_value
        r_p = float(getattr(config, 'TRAIN_R_AUGMENT_PROB', 0.0) or 0.0)
        r_vals = tuple(getattr(config, 'TRAIN_R_AUGMENT_VALUES', ()) or ())
        if self.augment and r_p > 0 and r_vals and random.random() < r_p:
            r = random.choice(r_vals)

        # 첫 묶음 시작부터 마지막 묶음 끝까지 한 번에 읽습니다(묶음이 하나면 clip_length 프레임).
        # 모든 묶음이 같은 박스, r, 증강을 써서 같은 충돌의 다른 시점일 뿐이 됩니다.
        s0 = min(starts)
        k = self._fps_aug_stride(sample, starts) if self.augment else 1
        if k > 1:
            frames = self._read_strided(sample, s0, k, bbox, r)
        else:
            n_read = max(starts) - s0 + self.clip_length
            cache = self._cache_path(sample['mp4_path'], s0, n_read, bbox, r)
            frames = self._cache_load(cache, n_read)
            if frames is None:
                frames = self._read_frames(sample['mp4_path'], s0, n_read, bbox, r)
                self._cache_save(cache, frames)

        # 증강은 학습 세트에만 겁니다(검증 세트는 결정적이어야 검증 손실이 안정됩니다).
        if self.augment:
            # 불빛 증강은 장면에 비치는 빛이라 색, 노이즈, 저화질보다 먼저 겁니다(기본 끔, config.TRAIN_LIGHT_AUG_PROB).
            frames = self._apply_light(frames)
            frames = self._apply_augmentation(frames)
            # 저화질 증강은 찍은 뒤 저장할 때 압축되는 순서에 맞춰 색, 노이즈 뒤에 겁니다. 원본 프레임 크롭 한 변으로 대상을 고릅니다.
            frames = self._apply_low_quality(frames, max(bbox[2] - bbox[0], bbox[3] - bbox[1]) * r)

        video = self._frames_to_tensor(frames)
        label = torch.tensor(sample['label'], dtype=torch.long)
        if not self.span_windows:
            return video, label
        # (K, 3, T, H, W): 구간을 덮는 묶음들
        return torch.stack([video[:, s - s0:s - s0 + self.clip_length]
                            for s in starts]), label


def span_collate(batch):
    """span 배치: 샘플마다 묶음 수가 달라 묶음들을 이어 붙이고 샘플별 묶음 수를 함께 줍니다.

    반환: (묶음들 (K 합계, 3, T, H, W), 샘플별 묶음 수 (B,), 레이블 (B,))
    """
    clips = torch.cat([b[0] for b in batch])
    sizes = torch.tensor([b[0].shape[0] for b in batch], dtype=torch.long)
    labels = torch.stack([b[1] for b in batch])
    return clips, sizes, labels
