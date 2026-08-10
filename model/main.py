import argparse
import torch

import config
from device_utils import get_device, is_channels_last_3d_supported
from hitandrun_model import HitAndRun3DCNN


def _load_model(weights_path):
    device = get_device()
    model = HitAndRun3DCNN(num_classes=config.MODEL_NUM_CLASSES).to(device)
    if is_channels_last_3d_supported(device) and config.USE_CHANNELS_LAST:
        model = model.to(memory_format=torch.channels_last_3d)
    try:
        # DirectML은 map_location 직접 지원이 불안정하므로 CPU 경유 로드
        state_dict = torch.load(
            weights_path, map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        print(f"가중치 로드 완료: {weights_path}")
    except FileNotFoundError:
        print(f"에러: 가중치 파일을 찾을 수 없습니다 -> {weights_path}")
        raise
    return model, device


def run_train():
    from train import train_model
    if not config.DATA_DIR.exists():
        print(f"에러: 데이터 폴더를 찾을 수 없습니다 -> {config.DATA_DIR}")
        return
    print(f"데이터 디렉토리 확인 완료: {config.DATA_DIR}. 학습을 시작합니다!")
    train_model()


def run_predict():
    from predict_cam import predict_hit_and_run_final
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
    args = parser.parse_args()

    if args.mode == "train":
        run_train()
    elif args.mode == "predict":
        run_predict()
    elif args.mode == "eval":
        run_eval()
