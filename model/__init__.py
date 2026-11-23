import sys
import os

# model/ 폴더를 sys.path에 넣습니다.
# 밖에서 model 패키지로 불러올 때도 안쪽 모듈끼리의 import(import config 등)가 동작하게 하려는 것입니다.
sys.path.insert(0, os.path.dirname(__file__))
