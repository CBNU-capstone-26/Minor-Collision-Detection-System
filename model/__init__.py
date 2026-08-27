import sys
import os

# model/ 폴더를 sys.path에 추가
# FastAPI 백엔드 등 외부에서 'from model.hit_and_run import HitAndRun3DCNN' 형태로 임포트할 때
# model/ 내부의 절대 임포트(import config 등)가 정상 동작하도록 함
sys.path.insert(0, os.path.dirname(__file__))
