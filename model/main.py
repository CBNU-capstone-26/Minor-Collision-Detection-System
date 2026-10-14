import argparse
from pathlib import Path

import torch

import config
from device_utils import get_device, is_channels_last_3d_supported
from hitandrun_model import HitAndRun3DCNN

'''
학습 시작
python3 -m model.main --mode train                 # A 클립 방식 pre (기본)
python3 -m model.main --mode train --a-mode span   # A 클립 방식 span (config 주석 참고)
python3 -m model.main --mode train --scratch       # 사전학습 없이 무작위 초기화(scratch)
python3 -m model.main --mode train --scratch --lr 1e-3   # 학습률 직접 지정

평가 시작 (--weights 를 안 주면 config.EVAL_WEIGHTS_PATH)
python3 -m model.main --mode eval --weights weights/hitandrun_x3d_..._ptY_span_0.1234.pth

예측 시작
python3 -m model.main --mode predict
'''

def _load_model(weights_path):
    # _load_model은 predict/eval(추론)에서만 사용 → INFER 디바이스 사용
    device = get_device(config.INFER_DEVICE_TYPE)
    model = HitAndRun3DCNN(num_classes=config.MODEL_NUM_CLASSES).to(device)
    if is_channels_last_3d_supported(device) and config.USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last_3d)
    try:
        state_dict = torch.load(
            weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        print(f"가중치 로드 완료: {weights_path}")
    except FileNotFoundError:
        print(f"에러: 가중치 파일을 찾을 수 없습니다 -> {weights_path}")
        raise
    return model, device


def run_train(lr=None):
    from train import train_model
    if not config.DATA_DIR.exists():
        print(f"에러: 데이터 폴더를 찾을 수 없습니다 -> {config.DATA_DIR}")
        return
    print(f"데이터 디렉토리 확인 완료: {config.DATA_DIR}. 학습을 시작합니다!")
    train_model(learning_rate=lr, save_path=config.TRAIN_BEST_MODEL_SAVE_PATH)


def run_predict():
    from inference import predict_hit_and_run_final
    model, _ = _load_model(config.PREDICT_WEIGHTS_PATH)
    out_path, events = predict_hit_and_run_final(model)
    if out_path is None:
        print("추론 실패: 결과 영상이 생성되지 않았습니다.")
        return
    if events:
        print(f"\n[이벤트 요약] 충돌 {len(events)}건 감지")
        for i, ev in enumerate(events, 1):
            print(f"  이벤트 {i}: start_frame={ev['start_frame']}, "
                  f"end_frame={ev['end_frame']}")


def run_eval():
    from evaluate import evaluate_folder_accuracy
    model, _ = _load_model(config.EVAL_WEIGHTS_PATH)
    evaluate_folder_accuracy(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hit-and-Run 3D-CNN 실행 스크립트")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "predict", "eval"],
        help="실행 모드: train(학습) / predict(단일 영상 CAM예측) / eval(정확도 평가)",
    )
    parser.add_argument(
        "--a-mode",
        choices=["pre", "span"],
        help="학습 전용. 충돌(A) 클립 방식 — pre(기본): 충돌 영상당 창 1개 / "
             "span: 접근~충돌 종료를 창 여러 개로 전부 학습. 안 주면 config.TRAIN_A_MODE",
    )
    parser.add_argument(
        "--scratch",
        action="store_true",
        help="학습 전용. Kinetics-400 사전학습 없이 무작위 초기화로 학습(가중치 이름 ptN). "
             "학습률 기본값은 config.TRAIN_SCRATCH_LEARNING_RATE",
    )
    parser.add_argument(
        "--lr",
        type=float,
        help="학습 전용. 학습률(헤드 기준). 안 주면 사전학습 config.TRAIN_LEARNING_RATE, "
             "scratch config.TRAIN_SCRATCH_LEARNING_RATE",
    )
    parser.add_argument(
        "--weights",
        help="eval·predict 전용. 쓸 가중치 파일(안 주면 config 의 EVAL/PREDICT_WEIGHTS_PATH)",
    )
    args = parser.parse_args()
    if args.a_mode and args.mode != "train":
        # 추론(평가·서비스)은 A 방식과 무관하다 — span 으로 학습한 가중치를 --weights 로 고른다
        parser.error("--a-mode 는 학습(--mode train)에만 씁니다. 평가할 가중치는 --weights 로 고르세요.")
    if args.weights and args.mode == "train":
        parser.error("--weights 는 eval·predict 에만 씁니다.")
    if (args.scratch or args.lr is not None) and args.mode != "train":
        parser.error("--scratch·--lr 은 학습(--mode train)에만 씁니다.")
    if args.lr is not None and args.lr <= 0:
        parser.error("--lr 은 0보다 커야 합니다.")
    if args.scratch:
        # 이름표(ptN)도 함께 바꾼다 — config 가 읽힐 때 이미 정해진 값이라서
        config.PRETRAINED = False
        config.PRETRAIN_TAG = "ptN"
    if args.a_mode:
        config.TRAIN_A_MODE = args.a_mode
    if args.weights:
        config.EVAL_WEIGHTS_PATH = config.PREDICT_WEIGHTS_PATH = Path(args.weights)

    if args.mode == "train":
        run_train(args.lr)
    elif args.mode == "predict":
        run_predict()
    elif args.mode == "eval":
        run_eval()
