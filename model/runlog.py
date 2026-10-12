"""학습·평가 1회분 산출물 폴더.

학습이나 평가를 한 번 돌릴 때마다 `<부모>/<YYMMDD_HHMM>_<모델>_<ptY|ptN>` 폴더를
새로 만들어 그 회차의 산출물을 모두 넣는다. 폴더 이름만 보고 언제·어떤 모델로
돌린 결과인지 알 수 있고, 다른 회차의 파일과 섞이지 않는다.
"""
from datetime import datetime
from pathlib import Path


def make_run_dir(parent, model_name, tag, when=None):
    """회차 폴더를 만들어 (폴더 경로, 'YYMMDD_HHMM' 문자열)을 돌려준다.

    같은 분에 두 번 돌리면 `_2`, `_3` 을 붙여 이전 회차를 덮어쓰지 않는다.
    """
    when = when or datetime.now()
    stamp = when.strftime('%y%m%d_%H%M')
    base = Path(parent) / f'{stamp}_{model_name}_{tag}'
    path, n = base, 1
    while path.exists():
        n += 1
        path = base.with_name(f'{base.name}_{n}')
    path.mkdir(parents=True)
    return path, stamp


def write_run_info(run_dir, items):
    """회차 폴더에 run_info.txt 를 쓴다(이미 있으면 새 내용으로 바꾼다).

    items 는 (항목, 값) 순서쌍 목록. 어떤 모델로 언제 돌렸고 어떤 가중치가
    나왔는지(또는 썼는지)를 폴더만 열어 바로 볼 수 있게 한다.
    """
    lines = [f'{k}: {"" if v is None else v}' for k, v in items]
    Path(run_dir, 'run_info.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
