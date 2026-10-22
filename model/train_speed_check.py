"""학습·검증 한 단계의 GPU 시간을 메모리 배치(channels_last) 켬/끔으로 재서 빠른 쪽을 고른다.

    python -u -m model.train_speed_check              # 묶음 config.TRAIN_BATCH_SIZE, 준비 5 + 측정 20단계씩
    python -u -m model.train_speed_check --steps 30

⚠️ GPU 가 비어 있을 때(학습과 학습 사이) 돌린다 — 학습 중이면 메모리가 모자라고 시간도 틀린다.
데이터는 읽지 않는다(무작위 입력). 학습이 GPU 계산 병목(사용률 100%)이라 계산 시간만 잰다.
AMP·cudnn.benchmark·gradient checkpointing·기울기 자르기(1.0)는 학습(train.py)과 같게 둔다.

두 배치는 같은 계산이라 학습 결과는 부동소수 반올림 차이 외엔 같다(아래 'P(A) 차이'로 확인).
빠른 쪽을 config.USE_CHANNELS_LAST 로 정한다.
"""
import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

import config
import hitandrun_model
from hitandrun_model import HitAndRun3DCNN


def _step_times(model, x, y, channels_last, dev, amp, steps, warmup, train):
    if channels_last:
        x = x.contiguous(memory_format=torch.channels_last_3d)
    crit = nn.CrossEntropyLoss()
    opt = optim.AdamW(model.parameters(), lr=1e-5) if train else None
    scaler = torch.amp.GradScaler('cuda') if (train and amp) else None
    model.train(train)
    times = []
    for i in range(warmup + steps):
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if train:
            opt.zero_grad(set_to_none=True)
            with torch.autocast(dev.type, enabled=amp):
                loss = crit(model(x), y)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
        else:
            with torch.no_grad(), torch.autocast(dev.type, enabled=amp):
                model(x)
        if dev.type == 'cuda':
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    return np.array(times)


def _run(channels_last, state, x, y, dev, amp, steps, warmup):
    """같은 초기 가중치로 학습 단계·검증 단계 시간, 최대 메모리, 검증 P(A)를 잰다."""
    model = HitAndRun3DCNN(num_classes=2, pretrained=False)
    model.load_state_dict(state)
    model = model.to(dev)
    if channels_last:
        model = model.to(memory_format=torch.channels_last_3d)
    # 결과 비교용 P(A) — 학습 전, 같은 가중치·같은 입력
    model.eval()
    xe = x.contiguous(memory_format=torch.channels_last_3d) if channels_last else x
    with torch.no_grad(), torch.autocast(dev.type, enabled=amp):
        p = torch.softmax(model(xe).float(), 1)[:, 1].cpu().numpy()
    if dev.type == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    t_val = _step_times(model, x, y, channels_last, dev, amp, steps, warmup, train=False)
    t_train = _step_times(model, x, y, channels_last, dev, amp, steps, warmup, train=True)
    peak = torch.cuda.max_memory_allocated() / 2 ** 30 if dev.type == 'cuda' else float('nan')
    del model
    if dev.type == 'cuda':
        torch.cuda.empty_cache()
    return t_train, t_val, peak, p


def main():
    ap = argparse.ArgumentParser(description='학습 한 단계 GPU 시간: channels_last 켬/끔 비교')
    ap.add_argument('--batch', type=int, default=config.TRAIN_BATCH_SIZE, help='묶음 크기(기본 학습과 같음)')
    ap.add_argument('--steps', type=int, default=20, help='측정 단계 수(준비 단계 제외)')
    ap.add_argument('--warmup', type=int, default=5, help='준비 단계(cudnn 최적 알고리즘 찾기) — 시간에서 뺌')
    ap.add_argument('--force', action='store_true', help='GPU 메모리가 이미 쓰이고 있어도 진행')
    ap.add_argument('--smoke', action='store_true', help='코드 점검용 — CPU·묶음 1·1단계(시간 의미 없음)')
    args = ap.parse_args()

    if args.smoke:
        dev, amp = torch.device('cpu'), False
        args.batch, args.steps, args.warmup = 1, 1, 0
    else:
        if not torch.cuda.is_available():
            print('GPU 가 잡히지 않습니다 — 학습 서버에서 실행하세요(점검만 하려면 --smoke).')
            return 1
        dev, amp = torch.device('cuda'), True
        torch.backends.cudnn.benchmark = True
        free, total = (v / 2 ** 30 for v in torch.cuda.mem_get_info())
        print(f'[GPU] {torch.cuda.get_device_name(0)} · 빈 메모리 {free:.1f} / {total:.1f} GB')
        if free < 0.85 * total and not args.force:
            print('⚠️ GPU 메모리를 다른 프로세스(학습 등)가 쓰고 있습니다. 학습이 끝난 뒤 실행하세요(--force 로 무시).')
            return 1

    print(f'[조건] 묶음 {args.batch} · {config.CLIP_LENGTH}프레임 · {config.RESIZE[0]}×{config.RESIZE[1]} · '
          f'AMP {"켬" if amp else "끔"} · gradient checkpointing {"켬" if hitandrun_model.GRAD_CHECKPOINT else "끔"} · '
          f'준비 {args.warmup} + 측정 {args.steps}단계', flush=True)
    torch.manual_seed(0)
    state = HitAndRun3DCNN(num_classes=2, pretrained=False).state_dict()
    x = torch.randn(args.batch, 3, config.CLIP_LENGTH, config.RESIZE[1], config.RESIZE[0], device=dev)
    y = torch.randint(0, 2, (args.batch,), device=dev)

    res = {}
    for cl in (True, False):
        name = 'channels_last 켬' if cl else 'channels_last 끔'
        print(f'  {name} 재는 중…', flush=True)
        try:
            res[cl] = _run(cl, state, x, y, dev, amp, args.steps, args.warmup)
        except torch.cuda.OutOfMemoryError:
            print(f'  {name}: GPU 메모리 부족(OOM) — 이 설정은 이 묶음 크기로 학습할 수 없음')
            torch.cuda.empty_cache()
    if not res:
        return 1

    print(f'\n{"설정":<18}{"학습 1단계":>14}{"검증 1단계":>14}{"최대 메모리":>12}')
    for cl, (tt, tv, peak, _) in res.items():
        print(f'{"channels_last " + ("켬" if cl else "끔"):<18}{tt.mean():>9.3f}±{tt.std():.3f}초'
              f'{tv.mean():>9.3f}±{tv.std():.3f}초{peak:>10.2f}GB')
    if len(res) == 2 and not args.smoke:
        (on_t, on_v, _, p_on), (off_t, off_v, _, p_off) = res[True], res[False]
        # 에포크 ≈ 학습 클립 수 × 학습 단계 + 검증 클립 수 × 검증 단계 (분할 8:2, 검증은 학습의 1/4)
        ep_on, ep_off = on_t.mean() + on_v.mean() / 4, off_t.mean() + off_v.mean() / 4
        faster = 'channels_last 끔' if ep_off < ep_on else 'channels_last 켬'
        d_t = (off_t.mean() - on_t.mean()) / on_t.mean() * 100
        d_v = (off_v.mean() - on_v.mean()) / on_v.mean() * 100
        print(f'\n→ 빠른 쪽: {faster} — 에포크 시간 약 {abs(ep_on - ep_off) / max(ep_on, ep_off) * 100:.0f}% 단축')
        print(f'  끔이 켬보다 학습 단계 {abs(d_t):.0f}% {"빠름" if d_t < 0 else "느림"} · '
              f'검증 단계 {abs(d_v):.0f}% {"빠름" if d_v < 0 else "느림"}')
        print(f'  같은 가중치·입력의 P(A) 최대 차이 {np.abs(p_on - p_off).max():.1e} (반올림 수준이면 결과 같음)')
        print(f'  지금 설정: config.USE_CHANNELS_LAST = {config.USE_CHANNELS_LAST}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
