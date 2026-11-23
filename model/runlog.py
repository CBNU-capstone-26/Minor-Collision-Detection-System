"""학습이나 평가 한 번의 결과를 담는 회차 폴더를 다룹니다.

실행할 때마다 <부모>/<YYMMDD_HHMM>_<모델>_<ptY|ptN> 폴더를 새로 만들어 그 회차의 결과를 모두 넣습니다.
폴더 이름만 보고 언제 어떤 모델로 돌린 결과인지 알 수 있고, 다른 회차의 파일과 섞이지 않습니다.
"""
from datetime import datetime
from pathlib import Path


def make_run_dir(parent, model_name, tag, when=None):
    """회차 폴더를 만들고 (폴더 경로, 'YYMMDD_HHMM' 문자열)을 돌려줍니다.
    같은 분에 두 번 돌리면 _2, _3을 붙여 이전 회차를 덮어쓰지 않습니다.
    """
    when = when or datetime.now()
    stamp = when.strftime('%y%m%d_%H%M')
    base = Path(parent) / f'{stamp}_{model_name}_{tag}'
    base.parent.mkdir(parents=True, exist_ok=True)
    path, n = base, 1
    while True:
        # 있는지 먼저 확인하지 않고, 만들어 본 뒤 이미 있으면 다음 번호로 넘어갑니다.
        # 두 학습이 같은 순간에 시작해도 같은 폴더를 함께 쓰지 않습니다.
        try:
            path.mkdir()
            return path, stamp
        except FileExistsError:
            n += 1
            path = base.with_name(f'{base.name}_{n}')


def write_run_info(run_dir, items):
    """회차 폴더에 run_info.txt를 씁니다(이미 있으면 새 내용으로 바꿉니다).
    items는 (항목, 값) 목록입니다. 어떤 조건으로 언제 돌렸고 어떤 가중치가 나왔는지(또는 썼는지) 폴더만 열어 바로 볼 수 있게 합니다.
    """
    lines = [f'{k}: {"" if v is None else v}' for k, v in items]
    Path(run_dir, 'run_info.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
