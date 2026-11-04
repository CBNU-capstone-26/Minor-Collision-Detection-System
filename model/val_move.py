"""검증 영상을 학습 폴더에서 검증 폴더로 옮긴다 (2026-11-04).

    python -m model.val_move            # 무엇을 옮길지 보기만
    python -m model.val_move --apply    # 옮기기 (mp4 + txt)
    python -m model.val_move --restore --apply   # 학습 폴더로 되돌리기

옮길 목록 = config.TRAIN_SPLIT_LOCK(model/split_lock.json)의 검증 189개(학습 폴더 기준 상대경로).
config.DATA_DIR/<상대경로> → config.VAL_DATA_DIR/<같은 상대경로>(rcdata/·realdata/ 구조 그대로).
지우지 않고 옮기기만 한다. 옮길 곳에 같은 이름 파일이 이미 있으면 그 영상은 건너뛴다.
옮긴 뒤 학습은 검증 폴더 전부를 검증으로 쓴다(train.py). 검증 클립 저장본은 경로가 바뀌어 첫 검증 때 다시 만든다.
"""
import argparse
import json
import shutil
import sys
from pathlib import Path

import config


def main():
    ap = argparse.ArgumentParser(description='검증 영상을 검증 폴더로 옮기기')
    ap.add_argument('--apply', action='store_true', help='실제로 옮긴다(없으면 보기만)')
    ap.add_argument('--restore', action='store_true', help='학습 폴더로 되돌린다')
    args = ap.parse_args()

    lock = Path(config.TRAIN_SPLIT_LOCK or '')
    if not lock.is_file():
        print(f'에러: 검증 고정 파일이 없습니다 -> {lock}')
        return 1
    rels = [v.replace('\\', '/') for v in json.loads(lock.read_text(encoding='utf-8'))['val']]
    train_root, val_root = Path(config.DATA_DIR), Path(config.VAL_DATA_DIR)
    src_root, dst_root = (val_root, train_root) if args.restore else (train_root, val_root)

    moves, absent = [], []
    for rel in rels:
        mp4 = src_root / rel
        if not mp4.exists():
            absent.append(rel)
            continue
        for src in (mp4, mp4.with_suffix('.txt')):
            if src.exists():
                moves.append((src, dst_root / Path(rel).with_suffix(src.suffix)))
    clash = {d for _, d in moves if d.exists()}
    n_real = sum(1 for s, _ in moves if s.suffix == '.mp4' and 'realdata' in s.parts)
    n_rc = sum(1 for s, _ in moves if s.suffix == '.mp4') - n_real
    print(f"[검증 영상 {'되돌리기' if args.restore else '옮기기'}] {lock.name} 목록 {len(rels)}개 · "
          f"{src_root} → {dst_root} · 옮길 영상 {n_real + n_rc}개(실제 {n_real}, rc {n_rc}) · 파일 {len(moves)}개")
    for src, dst in moves:
        if src.suffix == '.mp4' and ('realdata' in src.parts or len(moves) <= 60):
            print(f"  {src.relative_to(src_root)}" + ('  ⚠️ 옮길 곳에 이미 있음 — 건너뜀' if dst in clash else ''))
    if absent:
        print(f"  (옮길 곳에 이미 있거나 이 컴퓨터에 없음: {len(absent)}개"
              + (f" — 예: {', '.join(Path(a).stem for a in absent[:5])}" if absent else '') + ')')
    if not args.apply:
        print('보기만 했습니다. 옮기려면 --apply')
        return 0
    done = 0
    for src, dst in moves:
        if dst in clash:
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
        done += 1
    print(f'옮김: 파일 {done}개' + (f' · 건너뜀 {len(clash)}개' if clash else ''))
    return 0


if __name__ == '__main__':
    sys.exit(main())
