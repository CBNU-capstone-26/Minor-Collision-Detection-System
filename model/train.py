import csv
import json
import math
import os
import shutil
from datetime import datetime
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

import config
from dataset import HitAndRunDataset, span_collate
from device_utils import get_device, is_cuda_like, is_channels_last_3d_supported
from hitandrun_model import HitAndRun3DCNN
from runlog import make_run_dir, write_run_info
from train_balance import build as build_balance, resolve_csv


class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='best_model.pth'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float('inf')
        # 최종 파일명 규칙에 쓸 값 — best 가중치가 저장된 에포크(1-index)
        self.best_epoch = None

    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'조기 종료 카운트: {self.counter} / {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch):
        if val_loss < self.val_loss_min:
            print(
                f'판단 지표 개선 ({self.val_loss_min:.6f} --> {val_loss:.6f}). 모델 저장 중...')
            torch.save(model.state_dict(), self.path)
            self.val_loss_min = val_loss
            self.best_epoch = epoch


def _unique_path(path):
    """path 가 이미 있으면 이름 뒤에 _2, _3 … 을 붙인 경로(다른 학습의 결과를 덮어쓰지 않게)."""
    path, n = Path(path), 1
    out = path
    while out.exists():
        n += 1
        out = path.with_name(f'{path.stem}_{n}{path.suffix}')
    return out


def _make_loader(dataset, batch_size, shuffle, device, collate_fn=None, batch_sampler=None):
    # CUDA: 멀티워커 + prefetch 활성화 (① Ryzen 7 5700X 16스레드 기준)
    # CPU : Windows 멀티프로세싱 충돌 방지 및 디바이스 컨텍스트 공유 불가 문제로 단일 프로세스 사용
    if is_cuda_like(device):
        num_workers = min(8, os.cpu_count() or 2)
    else:
        num_workers = 0
    kwargs = {
        'num_workers': num_workers,
        'pin_memory': is_cuda_like(device),
    }
    if batch_sampler is not None:
        # 묶음 구성(train_balance)이 에포크마다 묶음을 직접 짠다 — batch_size·shuffle 대신
        kwargs['batch_sampler'] = batch_sampler
    else:
        kwargs.update({'batch_size': batch_size, 'shuffle': shuffle})
    if collate_fn is not None:
        kwargs['collate_fn'] = collate_fn
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
    learning_rate=None,
    use_amp=config.USE_AMP,
    use_channels_last=config.USE_CHANNELS_LAST,
    dry_run=False,
):
    """dry_run=True 면 데이터 분할·묶음 구성만 출력하고 학습하지 않는다(회차 폴더도 안 만듦)."""
    device = get_device(config.TRAIN_DEVICE_TYPE)
    cuda_like = is_cuda_like(device)
    # 학습률: 따로 안 주면 사전학습(미세조정)과 scratch 의 기본값이 다르다(config 주석).
    # 사전학습이면 백본은 ×0.1 로 보수적으로, scratch 면 지킬 사전학습 가중치가 없으니
    # 백본도 헤드와 같은 학습률로 학습한다.
    if learning_rate is None:
        learning_rate = (config.TRAIN_LEARNING_RATE if config.PRETRAINED
                         else config.TRAIN_SCRATCH_LEARNING_RATE)
    backbone_lr_mult = 0.1 if config.PRETRAINED else 1.0
    print(f"사용 중인 디바이스: {device}")

    if cuda_like:
        torch.backends.cudnn.benchmark = True

    # 학습용(증강 O) / 검증용(증강 X) 데이터셋을 각각 생성해 동일 인덱스로 분할.
    # → 검증 세트는 매 epoch 결정적이라 val loss가 안정되고 best model 선택이 신뢰됨.
    train_dataset = HitAndRunDataset(
        data_dir=data_dir, clip_length=clip_length, r_value=r_value,
        resize=resize, augment=True,
    )
    val_dataset = HitAndRunDataset(
        data_dir=data_dir, clip_length=clip_length, r_value=r_value,
        resize=resize, augment=False,
    )
    # 계층적 분할(stratified): 그룹을 세분화해 각 그룹을 train_split_ratio(8:2)로 나눈다.
    #   rc  : 방향(N/S/L/R) × 클래스(A/S) → 최대 8그룹 (각 방향·클래스가 train/val에 고루 분포)
    #   real: 클래스(A/S)만 → 2그룹 (실제 영상엔 방향 개념이 없음)
    #   → train은 모든 그룹의 학습분(80%)을 합쳐 학습, val은 rc용/real용을 분리해 각각 측정.
    def _domain(mp4_path):
        parts = os.path.normpath(mp4_path).split(os.sep)
        return 'real' if 'realdata' in parts else 'rc'

    def _direction(file_name):
        # 파일명 예: 220510_LA_0001 → 두 번째 토큰의 첫 글자가 방향(N/L/R/S)
        seg = file_name.split('_')
        return seg[1][0] if len(seg) > 1 and seg[1] else '?'

    def _scenario(file_name):
        # 두 번째 토큰의 마지막 글자가 시나리오/클래스 코드
        #   rc: A(충돌)/V(주차)/S(직진)/W(배회),  real: A(충돌)/S(비충돌)
        seg = file_name.split('_')
        return seg[1][-1] if len(seg) > 1 and seg[1] else '?'

    # 층화 그룹: rc = 방향(N/L/R/S) × 시나리오(A/V/S/W) → 최대 16그룹,
    #            real = 시나리오(A/S) → 2그룹. 각 그룹을 train_split_ratio(8:2)로 나눈다.
    #   ※ 실제 학습 라벨은 그대로 txt 기반 이진값(A vs 비A) — 시나리오는 '분할 균형'에만 사용.
    # ⚠️ 분할은 반드시 '영상 단위'로 한다. 비충돌 영상은 한 영상에서 여러 클립이
    #    나오는데(S 슬라이싱), 그 클립들이 train과 val로 갈리면 같은 장면을 이미
    #    학습한 상태로 검증하게 되어 검증 점수가 부풀려진다(데이터 누수).
    groups = {}  # (domain, direction, scenario) -> {video_key: [sample_idx, ...]}
    for i, s in enumerate(train_dataset.samples):
        dom = _domain(s['mp4_path'])
        direction = _direction(s['file_name']) if dom == 'rc' else '-'
        key = (dom, direction, _scenario(s['file_name']))
        vkey = s.get('video_key', s['mp4_path'])
        groups.setdefault(key, {}).setdefault(vkey, []).append(i)

    gen = torch.Generator().manual_seed(42)  # 재현 가능한 분할 (seed 고정)
    train_idx, val_rc_idx, val_real_idx = [], [], []
    split_videos = {'train': [], 'val': []}  # 회차 폴더에 남길 영상 목록(나중에 누수 확인용)
    n_train_vid = n_val_vid = 0
    for (dom, direction, scen), vid_map in sorted(groups.items()):
        vkeys = sorted(vid_map)                      # 영상 목록(그룹 내)
        order = torch.randperm(len(vkeys), generator=gen).tolist()
        shuffled = [vkeys[i] for i in order]
        cut = int(train_split_ratio * len(vkeys))
        for vk in shuffled[:cut]:                    # 영상 통째로 train
            train_idx += vid_map[vk]
            split_videos['train'].append(vk)
        for vk in shuffled[cut:]:                    # 영상 통째로 val
            (val_real_idx if dom == 'real' else val_rc_idx).extend(vid_map[vk])
            split_videos['val'].append(vk)
        n_train_vid += cut
        n_val_vid += len(vkeys) - cut

    print(f"[분할] 영상 {n_train_vid + n_val_vid}개 → train {n_train_vid} / "
          f"val {n_val_vid} (영상 단위, 누수 없음)")
    print(f"[분할] 클립 train {len(train_idx)} / val_rc {len(val_rc_idx)} / "
          f"val_real {len(val_real_idx)}")
    print("  그룹별 개수: " + ", ".join(
        f"{d}{'' if dr == '-' else '-' + dr}-{sc}:{len(v)}"
        for (d, dr, sc), v in sorted(groups.items())))

    # ── A 클립 방식 (config.TRAIN_A_MODE) ──
    # 'span' 이면 충돌 샘플 하나가 '접근~충돌 종료'를 덮는 창들이다. 창을 전부 A로 학습하되
    # 충돌 1건의 무게 합은 1(첫 창 TRAIN_SPAN_FIRST_WEIGHT + 나머지가 남은 무게를 나눔).
    # 회차 폴더·가중치 이름에 _span 을 붙여 'pre' 결과와 섞이지 않게 한다.
    span = train_dataset.span
    run_tag = f"{config.PRETRAIN_TAG}_span" if span else config.PRETRAIN_TAG
    collate = span_collate if span else None
    first_w = float(getattr(config, 'TRAIN_SPAN_FIRST_WEIGHT', 0.5))
    step_cap = int(getattr(config, 'TRAIN_SPAN_MAX_WINDOWS_PER_STEP', None) or batch_size * 2)
    if span:
        print(f"[A 방식] span — 접근~충돌 종료를 창 여러 개로 전부 학습(충돌 1건 무게 1: "
              f"첫 창 {first_w:g} + 나머지 창이 {1 - first_w:g}를 나눔), 비충돌은 창 1개")
        for dom in ('rc', 'real'):
            ws = [s['win_starts'] for s in train_dataset.samples
                  if s['label'] == 1 and _domain(s['mp4_path']) == dom and 'win_starts' in s]
            if not ws:
                continue
            ks = [len(w) for w in ws]
            gap = max((w[i + 1] - w[i] for w in ws for i in range(len(w) - 1)), default=0)
            print(f"  {dom}: 충돌 {len(ws)}개 — 창 수 분포 "
                  f"{dict(sorted((k, ks.count(k)) for k in set(ks)))}, 최대 창 간격 {gap}프레임"
                  + (" ⚠️ 30 이상이라 구간에 빈틈 있음" if gap >= clip_length else ""))
    else:
        print("[A 방식] pre — 충돌 영상당 창 1개(충돌 시작 "
              f"{getattr(config, 'TRAIN_A_PRE_FRAMES', 0)}프레임 전부터)")
    # 실제 영상 S 의 1번 차량 클립 (config.TRAIN_S_REAL_USE_CAR1)
    car1 = [s for s in train_dataset.samples if s.get('car_id') == 1 and s['label'] == 0]
    s_car_info = (f"0번 + 실제 영상 S 는 1번도 (1번 클립 {len(car1)}개, "
                  f"영상 {len({s['video_key'] for s in car1})}개)" if car1 else "0번만")
    print(f"[S 클립 차량] {s_car_info}")

    # ── 학습 묶음 구성 (config.TRAIN_S_MOTION_CSV · TRAIN_SPREAD_A, 기본 끔) ──
    # 학습 쪽만 바꾸고 검증 로더는 그대로다. 켜면 이름에 _bal 을 붙여 기존 학습과 구분한다.
    motion_csv = getattr(config, 'TRAIN_S_MOTION_CSV', None)
    spread_a = bool(getattr(config, 'TRAIN_SPREAD_A', False))
    balance_on = bool(motion_csv) or spread_a
    batch_sampler, balance_lines = None, []
    if balance_on:
        if motion_csv:
            motion_csv = resolve_csv(motion_csv)
            if not motion_csv.exists():
                raise FileNotFoundError(f"TRAIN_S_MOTION_CSV 파일이 없습니다: {motion_csv}")
        batch_sampler, balance_lines = build_balance(
            train_dataset.samples, train_idx, batch_size, csv_path=motion_csv,
            empty_pct=float(getattr(config, 'TRAIN_S_EMPTY_MOTION_PCT', 0.25)),
            empty_keep=float(getattr(config, 'TRAIN_S_EMPTY_KEEP', 1 / 3)),
            spread_a=spread_a)
        run_tag = f"{run_tag}_bal"
        for line in balance_lines:
            print(f"[묶음] {line}")
    else:
        n_a = sum(train_dataset.samples[i]['label'] == 1 for i in train_idx)
        n_s = len(train_idx) - n_a
        nb = math.ceil(len(train_idx) / batch_size)
        print(f"[묶음] 기본(무작위) — 에포크당 충돌 {n_a} · 정상 {n_s} = 1:{n_s / max(n_a, 1):.1f} · "
              f"묶음 {nb}개 · 충돌 없는 묶음 기대 {nb * (n_s / len(train_idx)) ** batch_size:.0f}개")

    if dry_run:
        if batch_sampler is not None:
            for ep in range(3):
                bl = batch_sampler.batches(ep)
                n_a_per = [sum(train_dataset.samples[train_idx[p]]['label'] == 1 for p in b) for b in bl]
                print(f"[점검] {ep + 1}에포크: 묶음 {len(bl)}개 · 클립 {sum(len(b) for b in bl)}개 · "
                      f"묶음당 충돌 {dict(sorted((k, n_a_per.count(k)) for k in set(n_a_per)))} · "
                      f"중복 {sum(len(b) for b in bl) - len({p for b in bl for p in b})}개")
        print("[점검] --dry-run 이라 학습하지 않고 끝냅니다.")
        return None

    train_loader = _make_loader(
        Subset(train_dataset, train_idx), batch_size=batch_size,
        shuffle=True, device=device, collate_fn=collate, batch_sampler=batch_sampler)
    val_rc_loader = _make_loader(
        Subset(val_dataset, val_rc_idx), batch_size=batch_size,
        shuffle=False, device=device, collate_fn=collate) if val_rc_idx else None
    val_real_loader = _make_loader(
        Subset(val_dataset, val_real_idx), batch_size=batch_size,
        shuffle=False, device=device, collate_fn=collate) if val_real_idx else None

    # ── 이번 학습의 회차 폴더 ──────────────────────────────────────
    # outputs/trainlogs/<YYMMDD_HHMM>_<모델>_<ptY|ptN>[_span]/ 에 이번 학습의 산출물을 모은다.
    # 가중치 파일 이름에도 같은 시각을 붙여, 어느 회차에서 나온 가중치인지 맞춰 볼 수 있다.
    run_started = datetime.now()
    run_dir, run_stamp = make_run_dir(
        config.TRAIN_LOG_DIR, config.MODEL_NAME, run_tag, run_started)
    print(f"[로그] 이번 학습 산출물 폴더: {run_dir}")
    # 학습 중 best·후보 가중치(작업용)는 이 회차 폴더 안에 둔다. 회차 폴더는 학습마다 새로
    # 만들어지므로, 같은 조건(ptY끼리 등)의 학습을 동시에 돌려도 서로의 파일을 덮어쓰거나
    # 지우지 않는다. 학습이 끝나면 규칙 파일명으로 weights_dir(= save_path 의 폴더)에 옮긴다.
    # 중간에 멈춘 학습의 best 는 이 폴더의 work_best.pth 에 남는다.
    weights_dir = Path(save_path).parent
    work_best = run_dir / 'work_best.pth'
    # 그때의 설정을 그대로 남긴다 — 나중에 config.py 가 바뀌어도 이 회차 조건을 알 수 있다
    shutil.copy2(config.__file__, run_dir / 'config_snapshot.py')
    # 학습/검증에 쓴 영상 목록 — 평가 영상과 겹치지 않았는지 나중에 확인할 수 있다
    with open(run_dir / 'split.json', 'w', encoding='utf-8') as _f:
        root = Path(data_dir)
        json.dump({
            'data_dir': str(root),
            'n_train_videos': len(split_videos['train']),
            'n_val_videos': len(split_videos['val']),
            'train': sorted(os.path.relpath(v, root) for v in split_videos['train']),
            'val': sorted(os.path.relpath(v, root) for v in split_videos['val']),
        }, _f, ensure_ascii=False, indent=2)
    run_info = [
        ('종류', '학습'),
        ('모델', config.MODEL_NAME),
        ('사전학습', f"{config.PRETRAIN_TAG} ("
                     f"{'Kinetics-400 사전학습' if config.PRETRAINED else 'scratch, 무작위 초기화'})"),
        ('학습률', f"헤드 {learning_rate:g} / 백본 {learning_rate * backbone_lr_mult:g}"),
        ('S 클립 차량', s_car_info),
        ('묶음 구성', ' / '.join(l.strip() for l in balance_lines) if balance_on
                     else '기본(무작위 섞기, 정상 전부)'),
        ('A 클립 방식', (f'span (접근~충돌 종료를 창 여러 개로 전부 학습, 첫 창 무게 {first_w:g})'
                       if span else f"pre (충돌 시작 {getattr(config, 'TRAIN_A_PRE_FRAMES', 0)}"
                            "프레임 전부터 창 1개)")),
        ('시작', run_started.strftime('%Y-%m-%d %H:%M:%S')),
        ('종료', '(학습 중이거나 중간에 멈춤)'),
        ('최종 가중치', '(학습이 끝나면 기록됨)'),
        ('작업 중 best 가중치', f'{work_best} (학습이 끝나면 최종 가중치로 옮겨짐)'),
        ('데이터 폴더', str(data_dir)),
        ('영상 수', f"학습 {len(split_videos['train'])} / 검증 {len(split_videos['val'])}"),
    ]
    write_run_info(run_dir, run_info)

    # ── 학습 곡선 로그 (CSV) ───────────────────────────────────────
    # 파일명에도 모델·사전학습여부·시각을 넣어, 회차 폴더 밖으로 꺼내도 구분된다.
    log_path = run_dir / (
        f"trainlog_{config.MODEL_NAME}_{run_tag}_{run_stamp}.csv")
    with open(log_path, 'w', newline='', encoding='utf-8') as _f:
        csv.writer(_f).writerow([
            'epoch', 'lr', 'train_loss',
            'val_rc_loss', 'val_rc_acc', 'val_rc_recallA', 'val_rc_precA',
            'val_real_loss', 'val_real_acc', 'val_real_recallA', 'val_real_precA',
            # ↓ 클래스별 평균 손실과 균형 손실, best 판단 지표, 이 에포크 저장 여부
            'val_rc_lossA', 'val_rc_lossS', 'val_rc_bal',
            'val_real_lossA', 'val_real_lossS', 'val_real_bal',
            'monitor', 'saved_best', 'saved_cand',
        ])
    print(f"[로그] 학습 곡선 CSV: {log_path}")

    def _csv(v):
        """None(해당 val 셋 없음/분모 0)은 빈칸으로 기록."""
        return '' if v is None else f'{v:.6f}'

    # 학습 시에는 Kinetics-400 사전학습 가중치로 초기화 (config.PRETRAINED)
    model = HitAndRun3DCNN(
        num_classes=num_classes, pretrained=config.PRETRAINED).to(device)

    # AMP, GradScaler: CUDA/ROCm 공통 지원
    # channels_last_3d: NVIDIA CUDA 전용 (ROCm 미지원)
    amp_enabled = use_amp and cuda_like
    channels_last_enabled = use_channels_last and is_channels_last_3d_supported(
        device)

    if channels_last_enabled:
        model = model.to(memory_format=torch.channels_last_3d)

    # S 슬라이싱으로 A:S 불균형이 커지면 모델이 S로 치우칠 수 있어, 필요 시
    # config.TRAIN_CLASS_WEIGHTS로 가중치를 준다(기본 None = 기존과 동일).
    _cw = getattr(config, 'TRAIN_CLASS_WEIGHTS', None)
    if _cw:
        criterion = nn.CrossEntropyLoss(
            weight=torch.tensor(_cw, dtype=torch.float32, device=device))
        print(f"[손실] 클래스 가중치 적용: {tuple(_cw)}")
    else:
        criterion = nn.CrossEntropyLoss()
    # 미세조정 표준 관행: 사전학습된 백본(features)은 낮은 LR(×0.1)로 보수적으로,
    # 새로 초기화된 분류 헤드(head_conv)는 기본 LR로 학습한다(scratch 면 둘 다 기본 LR).
    # AdamW = decoupled weight decay. 과적합 억제용 정규화는 여기서 건다.
    # (Adam + weight_decay 는 L2가 적응적 LR에 왜곡되어 의도대로 동작하지 않는다)
    #
    # 파라미터를 (사전학습/신규) × (decay/no-decay) 4그룹으로 나눈다.
    #  · 사전학습된 부분은 LR ×0.1 로 보수적으로 (기존 동작 유지, scratch 면 ×1)
    #  · ⚠️ model.named_parameters() 를 순회해야 한다. features/head_conv 만
    #    나열하면 그 사이에 있는 모듈(x3d 의 head_pool 0.97M)이 통째로 빠져
    #    학습되지 않는다.
    #  · BN scale/bias 와 모든 bias 는 weight decay 제외 — 표준 관행.
    #    BN 의 γ 가 0 으로 끌려가면 해당 채널 출력이 죽는다. 3D CNN 은 BN 텐서가
    #    154~220 개라 영향이 크다.
    _norm_ids = set()
    for _m in model.modules():
        if isinstance(_m, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d,
                           nn.GroupNorm, nn.LayerNorm)):
            _norm_ids.update(id(p) for p in _m.parameters(recurse=False))

    _g = {'new_d': [], 'new_n': [], 'pre_d': [], 'pre_n': []}
    for _name, _p in model.named_parameters():
        if not _p.requires_grad:
            continue
        # 새로 초기화되는 분류 헤드만 기본 LR, 나머지(사전학습분)는 ×0.1
        _kind = 'new' if _name.startswith('head_conv') else 'pre'
        _decay = 'n' if (id(_p) in _norm_ids or _name.endswith('bias')) else 'd'
        _g[f'{_kind}_{_decay}'].append(_p)

    _wd = config.TRAIN_WEIGHT_DECAY
    print(f"[옵티마이저] 백본 {sum(p.numel() for p in _g['pre_d'])/1e6:.2f}M(decay) + "
          f"{sum(p.numel() for p in _g['pre_n'])/1e3:.1f}K(no-decay) LR {learning_rate * backbone_lr_mult:g} / "
          f"헤드 {sum(p.numel() for p in _g['new_d'] + _g['new_n'])/1e3:.1f}K LR {learning_rate:g}, "
          f"weight_decay={_wd} ({'사전학습' if config.PRETRAINED else 'scratch'})")
    optimizer = optim.AdamW([
        {'params': _g['pre_d'], 'lr': learning_rate * backbone_lr_mult, 'weight_decay': _wd},
        {'params': _g['pre_n'], 'lr': learning_rate * backbone_lr_mult, 'weight_decay': 0.0},
        {'params': _g['new_n'], 'lr': learning_rate, 'weight_decay': 0.0},
        {'params': _g['new_d'], 'lr': learning_rate, 'weight_decay': _wd},
    ])

    # ── 안전한 학습 안정화 장치 (모델 구조·손실 불변, 성능 저하 없음) ──
    # 1) ReduceLROnPlateau: val loss가 정체되면 LR을 절반으로 낮춰 수렴을 돕는다.
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3)
    # 2) gradient clipping: 그래디언트 노름 상한(폭주/진동 억제). 필요 시 조정.
    grad_clip_norm = 1.0

    if amp_enabled:
        scaler = torch.amp.GradScaler("cuda")

    early_stopping = EarlyStopping(
        patience=early_stopping_patience, path=work_best)

    # ── best 판단 지표 / 후보 저장 (config 의 TRAIN_MONITOR_* 주석 참고) ──
    # 판단 지표 = val_real '클래스 균형 손실'의 최근 N에포크 평균. best 저장·LR 감소·
    # 조기종료가 모두 이 값 하나로 움직인다.
    smooth_n = max(1, int(getattr(config, 'TRAIN_MONITOR_SMOOTH', 3)))
    topk = max(1, int(getattr(config, 'TRAIN_TOPK_CANDIDATES', 3)))
    min_gap = max(1, int(getattr(config, 'TRAIN_TOPK_MIN_GAP', 3)))
    bal_hist = []
    cands = []   # [{'epoch', 'score', 'path'}] — 서로 min_gap 이상 떨어진 상위 topk
    # 후보도 회차 폴더 안에 둔다(새 폴더라 이전 학습의 파일과 섞일 일이 없다)

    def _update_candidates(epoch_no, score):
        """epoch_no 의 가중치를 후보로 둘지 정한다. 저장했으면 True.

        그냥 상위 K개를 고르면 33·34·35 처럼 붙은 에포크, 즉 사실상 같은 모델이
        뽑힌다. 그래서 가까운(min_gap 미만) 후보끼리는 더 좋은 하나만 남긴다.
        가장 좋은 에포크는 절대 밀려나지 않으므로 1위는 항상 best 와 같다.
        """
        near = [c for c in cands if abs(c['epoch'] - epoch_no) < min_gap]
        if near:
            if score >= min(c['score'] for c in near):
                return False
            for c in near:
                c['path'].unlink(missing_ok=True)
                cands.remove(c)
        elif len(cands) >= topk:
            worst = max(cands, key=lambda c: c['score'])
            if score >= worst['score']:
                return False
            worst['path'].unlink(missing_ok=True)
            cands.remove(worst)
        path = run_dir / f'work_cand_ep{epoch_no}.pth'
        torch.save(model.state_dict(), path)
        cands.append({'epoch': epoch_no, 'score': score, 'path': path})
        return True

    plot_warned = False

    _cw_t = torch.tensor(_cw, dtype=torch.float32, device=device) if _cw else None

    def _span_weights(sizes):
        """span: 창별 (무게, 주인 샘플 번호). 샘플마다 무게 합 = 1.

        창 1개(비충돌·짧은 충돌)면 1, 여러 개면 첫 창 first_w + 나머지가 1 − first_w 를 나눈다.
        """
        w, owner = [], []
        for i, k in enumerate(sizes.tolist()):
            w += [1.0] if k == 1 else [first_w] + [(1.0 - first_w) / (k - 1)] * (k - 1)
            owner += [i] * k
        return (torch.tensor(w, device=device),
                torch.tensor(owner, device=device, dtype=torch.long))

    def _span_step(inputs, sizes, labels):
        """span 학습 한 스텝 → 배치 손실(샘플 평균, 'pre'의 손실과 같은 척도).

        손실 = Σ 샘플 (클래스 가중치 × Σ 창 무게 × 창 CE) / Σ 클래스 가중치
             — 모든 샘플이 창 1개면 nn.CrossEntropyLoss 와 같다.
        창이 step_cap 을 넘으면 섞어서 고르게 나눠 역전파하고 기울기를 합쳐 한 번에 갱신한다.
        (나눠도 손실·기울기는 같고, BN 통계만 조각 단위로 계산된다)
        """
        w, owner = _span_weights(sizes)
        cls_w = (_cw_t[labels] if _cw_t is not None
                 else torch.ones(labels.size(0), device=device))
        coef = cls_w[owner] * w / cls_w.sum()
        n = inputs.size(0)
        if n <= step_cap:
            chunks = [torch.arange(n, device=device)]
        else:
            # 한 조각에 한 충돌의 창만 몰리지 않게 섞는다(BN 통계가 치우치지 않게)
            chunks = torch.randperm(n, device=device).chunk(math.ceil(n / step_cap))
        optimizer.zero_grad(set_to_none=True)
        total = 0.0
        for idx in chunks:
            x = inputs[idx]
            if channels_last_enabled:
                x = x.contiguous(memory_format=torch.channels_last_3d)
            with torch.autocast(device.type, enabled=amp_enabled):
                out = model(x)
            ce = F.cross_entropy(out.float(), labels[owner[idx]], reduction='none')
            loss = (coef[idx] * ce).sum()
            if amp_enabled:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            total += loss.item()
        if amp_enabled:
            scaler.unscale_(optimizer)  # 클리핑 전 스케일 해제 (AMP)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()
        return total

    def _run_windows(x):
        """검증: 창들을 배치 크기 단위로 나눠 추론한다(fp32, 'pre' 검증과 같은 조건).

        span 은 창이 샘플보다 많아 한 번에 넣으면 메모리가 부족할 수 있다. → 로짓 (N, 2)
        """
        outs = []
        for i in range(0, x.size(0), batch_size):
            chunk = x[i:i + batch_size]
            if channels_last_enabled:
                chunk = chunk.contiguous(memory_format=torch.channels_last_3d)
            outs.append(model(chunk).float())
        return torch.cat(outs)

    def _max_window(logits, sizes):
        """샘플마다 충돌 점수(logit_A − logit_S)가 가장 높은 창의 위치(전체 창 기준)."""
        score = logits[:, 1] - logits[:, 0]
        picks, off = [], 0
        for k in sizes.tolist():
            picks.append(off + int(score[off:off + k].argmax()))
            off += k
        return picks

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        n_seen = 0      # 이번 에포크에 실제로 쓴 클립 수(묶음 구성이 빈 장면을 솎으면 전체보다 적다)
        for batch in train_loader:
            non_blocking = cuda_like
            if span:
                inputs, sizes, labels = batch
                inputs = inputs.to(device, non_blocking=non_blocking)
                labels = labels.to(device, non_blocking=non_blocking)
                train_loss += _span_step(inputs, sizes, labels) * labels.size(0)
                n_seen += labels.size(0)
                continue
            inputs, labels = batch
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
                scaler.unscale_(optimizer)  # 클리핑 전 스케일 해제 (AMP)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), grad_clip_norm)
                optimizer.step()

            train_loss += loss.item() * labels.size(0)
            n_seen += labels.size(0)

        avg_train_loss = train_loss / max(n_seen, 1)

        # rc용/real용 val을 각각 분리해서 성능 측정
        def _evaluate(loader):
            """val 로더 하나 → 지표 dict (로더가 없으면 None).

            loss/acc/rec/prec : argmax(임계값 0.5) 기준 — 기존과 같다.
            lossA / lossS     : 충돌·비충돌 클립 각각의 평균 손실(가중치 없는 CE).
            bal               : (lossA + lossS) / 2 — 클래스 균형 손실.

            ⚠️ val_real 은 51클립 중 40개가 비충돌이라 그냥 손실은 78%가 '비충돌을
               얼마나 확신하느냐'로 정해진다. 목표는 충돌을 놓치지 않는 것이므로
               best 판단에는 두 클래스를 같은 무게로 보는 bal 을 쓴다.
               acc 도 A:S 불균형 때문에 전부 S로 찍어도 높게 나오니 recall 을 본다.
            """
            if loader is None:
                return None
            model.eval()
            vloss, correct = 0.0, 0
            tp = fp = fn = 0                       # A(=1) 기준
            sum_a = sum_s = 0.0
            n_a = n_s = 0
            with torch.inference_mode():
                for batch in loader:
                    if span:
                        inputs, sizes, labels = batch
                    else:
                        inputs, labels = batch
                    inputs = inputs.to(device, non_blocking=cuda_like)
                    labels = labels.to(device, non_blocking=cuda_like)
                    if span:
                        # 손실: 학습과 같은 무게(첫 창 + 나머지 창)로 샘플별 손실을 낸다.
                        # 판정(acc·recall·prec): 서비스처럼 창 하나라도 A면 A —
                        #   충돌 점수가 가장 높은 창으로 판정한다.
                        logits = _run_windows(inputs)
                        w, owner = _span_weights(sizes)
                        ce = torch.zeros(labels.size(0), device=device).index_add_(
                            0, owner, w * F.cross_entropy(logits, labels[owner], reduction='none'))
                        outputs = logits[_max_window(logits, sizes)]
                        cls_w = (_cw_t[labels] if _cw_t is not None
                                 else torch.ones_like(ce))
                        vloss += ((cls_w * ce).sum() / cls_w.sum()).item() * labels.size(0)
                    else:
                        if channels_last_enabled:
                            inputs = inputs.contiguous(
                                memory_format=torch.channels_last_3d)
                        outputs = model(inputs)
                        vloss += criterion(outputs, labels).item() * labels.size(0)
                        ce = F.cross_entropy(outputs.float(), labels, reduction='none')
                    is_a = labels == 1
                    sum_a += ce[is_a].sum().item()
                    sum_s += ce[~is_a].sum().item()
                    n_a += int(is_a.sum().item())
                    n_s += int((~is_a).sum().item())
                    preds = outputs.argmax(dim=1)
                    correct += torch.sum(preds == labels).item()
                    tp += torch.sum((preds == 1) & (labels == 1)).item()
                    fp += torch.sum((preds == 1) & (labels != 1)).item()
                    fn += torch.sum((preds != 1) & (labels == 1)).item()
            n = len(loader.dataset)
            loss_a = sum_a / n_a if n_a else None
            loss_s = sum_s / n_s if n_s else None
            return {
                'loss': vloss / n, 'acc': correct / n,
                'rec': tp / (tp + fn) if (tp + fn) else None,
                'prec': tp / (tp + fp) if (tp + fp) else None,
                'lossA': loss_a, 'lossS': loss_s,
                # 검증셋에 한 클래스가 없으면 균형 손실을 못 구한다
                'bal': (loss_a + loss_s) / 2 if (n_a and n_s) else None,
            }

        rc = _evaluate(val_rc_loader)
        real = _evaluate(val_real_loader)

        def _g(d, k):
            return d[k] if d else None

        # ── 판단 지표 ── 실제 영상(val_real)이 목표라 우선. 없으면(rc만 학습) rc.
        # 한 에포크 값은 흔들리므로(수렴 후 표준편차 0.013) 최근 N에포크 평균을 쓴다.
        ref = real if real is not None else rc
        bal_hist.append(ref['bal'] if ref['bal'] is not None else ref['loss'])
        window = bal_hist[-smooth_n:]
        monitor = sum(window) / len(window)
        scheduler.step(monitor)
        current_lr = optimizer.param_groups[-1]['lr']  # 헤드 LR 표시

        def _fmt(d):
            if d is None:
                return 'N/A'
            rs = f"{d['rec']:.3f}" if d['rec'] is not None else '-'
            return f"{d['loss']:.4f}/{d['acc']:.4f}/A{rs}"
        print(f'Epoch [{epoch+1}/{num_epochs}] Train {avg_train_loss:.4f} | '
              f'val_rc(L/Acc/Arecall) {_fmt(rc)} | '
              f'val_real(L/Acc/Arecall) {_fmt(real)} | '
              f'판단지표 {monitor:.4f} | LR {current_lr:.2e}')

        # best 저장·조기종료 판단 → 후보 저장
        prev_best = early_stopping.best_epoch
        early_stopping(monitor, model, epoch + 1)
        saved_best = (early_stopping.best_epoch == epoch + 1
                      and prev_best != epoch + 1)
        saved_cand = _update_candidates(epoch + 1, monitor)

        # 에포크별 곡선을 CSV로 남긴다 — 학습이 중간에 끊겨도 남도록 매번 append.
        # 과적합 판단(train↓ val↑)과 조합 비교에 이 파일이 근거가 된다.
        with open(log_path, 'a', newline='', encoding='utf-8') as _f:
            csv.writer(_f).writerow([
                epoch + 1, f'{current_lr:.3e}', f'{avg_train_loss:.6f}',
                _csv(_g(rc, 'loss')), _csv(_g(rc, 'acc')),
                _csv(_g(rc, 'rec')), _csv(_g(rc, 'prec')),
                _csv(_g(real, 'loss')), _csv(_g(real, 'acc')),
                _csv(_g(real, 'rec')), _csv(_g(real, 'prec')),
                _csv(_g(rc, 'lossA')), _csv(_g(rc, 'lossS')), _csv(_g(rc, 'bal')),
                _csv(_g(real, 'lossA')), _csv(_g(real, 'lossS')), _csv(_g(real, 'bal')),
                f'{monitor:.6f}', int(saved_best), int(saved_cand),
            ])

        # 학습 곡선 그래프 갱신 — 실패해도 학습은 계속한다
        try:
            from trainplot import plot_trainlog
            plot_trainlog(log_path)
        except Exception as exc:  # noqa: BLE001
            if not plot_warned:
                print(f'[그래프] 생성 실패(학습은 계속): {exc}')
                plot_warned = True

        if early_stopping.early_stop:
            print("조기 종료 조건 충족. 학습을 중단합니다.")
            break

    # ── 최종 가중치 파일명 규칙 적용 ───────────────────────────────
    # hitandrun_[모델명]_[YYMMDD_HHMM]_[에포크수]ep_[earlyY|earlyN]_[ptY|ptN][_span]_[손실율]
    #   · 모델명 : config.MODEL_NAME (s3d / x3d / slowfast) — 폴더별 구분용
    #   · 시각   : 학습 시작 시각 = 회차 폴더 이름의 시각 (예: 2026-07-24 14:05 → 260724_1405)
    #             ⚠️ 2026-10-11 이전 파일은 '학습 종료일'(YYMMDD)만 붙어 있다.
    #   · 에포크 : best 가중치가 저장된 에포크 (예: 50 → 50ep)
    #   · early  : 조기종료로 멈췄으면 earlyY, 아니면 earlyN
    #   · _span  : A 클립 방식이 span 일 때만 붙는다(--a-mode span)
    #   · 손실율 : 저장 당시 '판단 지표'(val_real 균형 손실의 이동평균) 값.
    #             ⚠️ 2026-09-29 이전 파일은 그냥 val_real loss 였으므로 서로 비교 불가.
    date_str = run_stamp
    epoch_str = f'{early_stopping.best_epoch}ep'
    early_str = 'earlyY' if early_stopping.early_stop else 'earlyN'
    loss_str = f'{early_stopping.val_loss_min:.4f}'
    final_name = (f'hitandrun_{config.MODEL_NAME}_{date_str}_'
                  f'{epoch_str}_{early_str}_{run_tag}_{loss_str}.pth')
    # 학습 중엔 회차 폴더의 work_best.pth 로 저장해 두고, 종료 시 규칙 파일명으로 옮긴다
    # (회차 폴더와 weights 가 다른 디스크여도 되게 shutil.move. 같은 이름이 있으면 _2 를 붙인다)
    final_path = _unique_path(weights_dir / final_name)
    shutil.move(str(work_best), str(final_path))
    print(f'\n최종 가중치 저장: {final_path}')

    # ── 후보 정리 ──
    # 1위는 위 최종 파일과 같은 가중치라 작업용 후보 파일을 지운다. 나머지는
    # 순위·에포크·지표값을 붙여 남긴다 → 학습 후, 학습에 안 쓴 영상으로 서비스
    # 경로 평가를 돌려 최종 선택한다(val_real 로 골랐으니 val_real 로 다시 비교하면
    # 그 51클립에 우연히 맞은 쪽이 또 이긴다).
    cand_info = []
    for rank, c in enumerate(sorted(cands, key=lambda c: c['score']), 1):
        if c['epoch'] == early_stopping.best_epoch:
            c['path'].unlink(missing_ok=True)
            dest_name = final_path.name
        else:
            dest = _unique_path(weights_dir / (
                f"hitandrun_{config.MODEL_NAME}_{date_str}_{run_tag}_"
                f"cand{rank}_ep{c['epoch']}_{c['score']:.4f}.pth"))
            shutil.move(str(c['path']), str(dest))
            dest_name = dest.name
            print(f'후보 {rank}위 저장: {dest}')
        cand_info.append({'rank': rank, 'epoch': c['epoch'],
                          'score': round(c['score'], 6), 'file': dest_name})

    run_finished = datetime.now()
    info = dict(run_info)
    info['종료'] = run_finished.strftime('%Y-%m-%d %H:%M:%S')
    info['최종 가중치'] = str(final_path)
    info['작업 중 best 가중치'] = '(최종 가중치로 옮김)'
    info['best 에포크'] = (f"{early_stopping.best_epoch} "
                          f"({'조기 종료' if early_stopping.early_stop else '끝까지 학습'})")
    info['후보 가중치'] = ', '.join(c['file'] for c in cand_info if c['rank'] > 1) or '없음'
    write_run_info(run_dir, list(info.items()))

    # 결과 요약 — 그래프가 best·후보를 표시할 때도 쓴다
    result_path = log_path.with_name(log_path.stem + '.result.json')
    with open(result_path, 'w', encoding='utf-8') as _f:
        json.dump({
            'started': run_started.strftime('%Y-%m-%d %H:%M:%S'),
            'finished': run_finished.strftime('%Y-%m-%d %H:%M:%S'),
            'run_dir': str(run_dir),
            'best_epoch': early_stopping.best_epoch,
            'early_stop': bool(early_stopping.early_stop),
            'final_weights': final_path.name,
            'weights_dir': str(final_path.parent),
            'monitor': f'val_real class-balanced loss, {smooth_n}-epoch avg',
            'a_mode': 'span' if span else 'pre',
            'balance': ({'motion_csv': str(motion_csv) if motion_csv else None,
                         'empty_pct': getattr(config, 'TRAIN_S_EMPTY_MOTION_PCT', None),
                         'empty_keep': getattr(config, 'TRAIN_S_EMPTY_KEEP', None),
                         'spread_a': spread_a} if balance_on else None),
            'candidates': cand_info,
        }, _f, ensure_ascii=False, indent=2)
    try:
        from trainplot import plot_trainlog
        png = plot_trainlog(log_path)
        print(f'[그래프] 학습 곡선: {png}')
    except Exception as exc:  # noqa: BLE001
        print(f'[그래프] 생성 실패: {exc}')

    state_dict = torch.load(final_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    return model
