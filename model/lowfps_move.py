"""실제 화면 갱신이 10fps 미만인 영상(config.LOW_FPS_VIDEOS)을 학습·평가에 안 쓰는 폴더로 옮긴다 (2026-11-04).

    python -m model.lowfps_move            # 무엇을 옮길지 보기만
    python -m model.lowfps_move --apply    # 옮기기 (mp4 + txt)
    python -m model.lowfps_move --restore --apply   # 원래 자리로 되돌리기

학습 폴더(config.DATA_DIR) → config.LOW_FPS_EXCLUDED_DIR/train/<같은 상대경로>,
평가 폴더(config.EVAL_FOLDER_PATH) → config.LOW_FPS_EXCLUDED_DIR/eval/<같은 상대경로>.
지우지 않고 옮기기만 한다. 같은 이름 파일이 옮길 곳에 이미 있으면 그 영상은 건너뛴다.
검증은 config.TRAIN_SPLIT_LOCK 으로 고정돼 있어 옮겨도 다른 영상의 분할은 바뀌지 않는다.
"""
import argparse
import shutil
import sys
from pathlib import Path

import config


def _plan(restore):
    names = set(config.LOW_FPS_VIDEOS)
    base = Path(config.LOW_FPS_EXCLUDED_DIR)
    moves = []
    for part, root in (('train', Path(config.DATA_DIR)), ('eval', Path(config.EVAL_FOLDER_PATH))):
        src_root, dst_root = (base / part, root) if restore else (root, base / part)
        if not src_root.exists():
            continue
        for mp4 in sorted(src_root.rglob('*.mp4')):
            if mp4.stem not in names:
                continue
            rel = mp4.relative_to(src_root)
            for src in (mp4, mp4.with_suffix('.txt')):
                if src.exists():
                    moves.append((part, mp4.stem, src, dst_root / rel.with_suffix(src.suffix)))
    return moves


def main():
    ap = argparse.ArgumentParser(description='10fps 미만 영상을 학습·평가 밖 폴더로 옮기기')
    ap.add_argument('--apply', action='store_true', help='실제로 옮긴다(없으면 보기만)')
    ap.add_argument('--restore', action='store_true', help='원래 자리로 되돌린다')
    args = ap.parse_args()

    moves = _plan(args.restore)
    found = {stem for _, stem, _, _ in moves}
    absent = sorted(set(config.LOW_FPS_VIDEOS) - found)
    print(f"[10fps 미만 영상 {'되돌리기' if args.restore else '옮기기'}] 목록 {len(config.LOW_FPS_VIDEOS)}개 · "
          f"옮길 영상 {len(found)}개(파일 {len(moves)}개) · 옮길 곳 "
          f"{config.LOW_FPS_EXCLUDED_DIR if not args.restore else '원래 폴더'}")
    clash = [d for _, _, _, d in moves if d.exists()]
    for part, stem, src, dst in moves:
        print(f"  {part:5s} {src} -> {dst}" + ('  ⚠️ 이미 있음 — 건너뜀' if dst in clash else ''))
    if absent:
        print(f"  (이미 옮겼거나 이 컴퓨터에 없음: {', '.join(absent)})")
    if not args.apply:
        print('보기만 했습니다. 옮기려면 --apply')
        return 0
    done = 0
    for part, stem, src, dst in moves:
        if dst in clash:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        done += 1
    print(f'옮김: 파일 {done}개' + (f' · 건너뜀 {len(clash)}개' if clash else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
