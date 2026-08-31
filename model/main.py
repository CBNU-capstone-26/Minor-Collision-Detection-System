import argparse

import config

'''
학습 시작
python3 -m model.main --mode train

예측 시작
python3 -m model.main --mode predict
'''

def _load_model(weights_path):
    import torch
    from device_utils import get_device, is_channels_last_3d_supported
    from hitandrun_model import HitAndRun3DCNN

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


def run_train():
    config.DATA_DIR.mkdir(parents=True, exist_ok=True)

    mp4_files = sorted(config.DATA_DIR.glob("*.mp4"))
    if not mp4_files:
        print(f"학습 데이터 폴더를 준비했습니다 -> {config.DATA_DIR}")
        print("학습할 mp4 파일이 아직 없습니다.")
        print("data/train 안에 같은 이름의 영상과 라벨 txt를 넣어주세요. 예: sample01.mp4, sample01.txt")
        return

    missing_txt = [path.name for path in mp4_files if not path.with_suffix(".txt").exists()]
    if missing_txt:
        print(f"에러: mp4와 같은 이름의 bbox/action txt 라벨이 필요합니다 -> {config.DATA_DIR}")
        print("라벨이 없는 영상:")
        for name in missing_txt[:10]:
            print(f"  - {name}")
        if len(missing_txt) > 10:
            print(f"  ... 외 {len(missing_txt) - 10}개")
        return

    print(f"데이터 디렉토리 확인 완료: {config.DATA_DIR}. 학습을 시작합니다!")
    from train import train_model
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
