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

추론 실행 방식 자가 검사 — 서버 설치 직후 한 번 (--weights 를 안 주면 서비스 가중치)
python3 -m model.main --mode check-backend

영상 하나 분석 — 서비스와 같은 방식(사고 의심 구간 + 클립). 결과는 outputs/detect/<시각>_<영상이름>/
python3 -m model.main --mode detect --video data/eval/260828_TA_0001.mp4          # 박스는 같은 이름 .txt 의 0번 차량
python3 -m model.main --mode detect --video 영상.mp4 --bbox 101,45,198,153         # 박스 직접 지정(원본 해상도 x1,y1,x2,y2)

예측 시작 (개발용: config 의 영상 전체에 CAM 합성)
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


def run_detect(video, bbox_arg=None):
    """영상 하나를 서비스와 똑같이 분석한다 — 사고 의심 구간을 찾고 그 구간 클립을 만든다."""
    from datetime import datetime
    from inference import predict_events_and_clips
    video = Path(video)
    if not video.exists():
        print(f"에러: 영상을 찾을 수 없습니다 -> {video}")
        return
    if bbox_arg:
        bbox = [int(float(v)) for v in bbox_arg.split(',')]
        if len(bbox) != 4:
            print("에러: --bbox 는 x1,y1,x2,y2 네 숫자여야 합니다 (예: --bbox 101,45,198,153)")
            return
    else:
        txt = video.with_suffix('.txt')
        boxes = []
        if txt.exists():
            for line in txt.read_text(encoding='utf-8', errors='ignore').splitlines():
                p = [t.strip() for t in line.split(',')]
                if len(p) >= 6 and p[0] == 'car' and p[1] == '0':
                    boxes.append([int(float(v)) for v in p[2:6]])
        if not boxes:
            print(f"에러: 사고 차량 박스가 없습니다. {txt.name} 에 'car,0,x1,y1,x2,y2' 줄을 두거나 "
                  "--bbox x1,y1,x2,y2 로 지정하세요")
            return
        bbox = boxes[0]
    model, _ = _load_model(config.DETECT_WEIGHTS_PATH)
    out_dir = Path(config._ROOT) / 'outputs' / 'detect' / f"{datetime.now():%y%m%d_%H%M%S}_{video.stem}"
    print(f"분석: {video} | 사고 차량 박스 {bbox} | 결과 폴더 {out_dir}")
    events = predict_events_and_clips(model, video_path=str(video), bbox=tuple(bbox), output_dir=out_dir)
    print(f"\n[결과] 사고 의심 구간 {len(events)}건")
    for i, e in enumerate(events, 1):
        print(f"  {i}. {e['start_sec']:.2f}초 ~ {e['end_sec']:.2f}초 (프레임 {e['start_frame']}~{e['end_frame']}), "
              f"충돌 확률 {e['crash_prob']:.3f}\n     클립: {e.get('clip_path')}")


def run_check_backend():
    """서비스 추론 실행 방식을 지금 이 컴퓨터에서 다시 자가 검사하고 결과를 저장한다.

    서비스 워커는 처음 분석 때 같은 검사를 하므로 안 해도 동작은 한다. 미리 해 두면 첫 분석이
    검사 시간만큼 빨라진다. 가중치·패키지·하드웨어를 바꾼 뒤에도 다시 돌리면 된다.
    """
    from infer_runtime import get_runner
    model, device = _load_model(config.CHECK_WEIGHTS_PATH)
    runner = get_runner(model, force_check=True)
    print(f"\n서비스 추론 실행 방식: {runner.name} (장치 {device}) — 저장 위치 {config.INFER_CACHE_DIR}")


def run_eval():
    from evaluate import evaluate_folder_accuracy
    model, _ = _load_model(config.EVAL_WEIGHTS_PATH)
    evaluate_folder_accuracy(model)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hit-and-Run 3D-CNN 실행 스크립트")
    parser.add_argument(
        "--mode",
        required=True,
        choices=["train", "predict", "eval", "check-backend", "detect"],
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
    parser.add_argument("--video", help="detect 전용. 분석할 영상 파일")
    parser.add_argument("--bbox", help="detect 전용. 사고 차량 박스 x1,y1,x2,y2 (안 주면 같은 이름 .txt 의 0번 차량)")
    parser.add_argument(
        "--weights",
        help="eval·predict 전용. 쓸 가중치 파일(안 주면 config 의 EVAL/PREDICT_WEIGHTS_PATH)",
    )
    args = parser.parse_args()
    if args.a_mode and args.mode != "train":
        # 추론(평가·서비스)은 A 방식과 무관하다 — span 으로 학습한 가중치를 --weights 로 고른다
        parser.error("--a-mode 는 학습(--mode train)에만 씁니다. 평가할 가중치는 --weights 로 고르세요.")
    if args.weights and args.mode == "train":
        parser.error("--weights 는 eval·predict·check-backend·detect 에만 씁니다.")
    if args.mode == "detect" and not args.video:
        parser.error("--mode detect 에는 --video 영상경로 가 필요합니다.")
    if (args.video or args.bbox) and args.mode != "detect":
        parser.error("--video·--bbox 는 --mode detect 에만 씁니다.")
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
    config.CHECK_WEIGHTS_PATH = config.DETECT_WEIGHTS_PATH = config.SERVICE_WEIGHTS_PATH
    if args.weights:
        config.DETECT_WEIGHTS_PATH = Path(args.weights)
        config.EVAL_WEIGHTS_PATH = config.PREDICT_WEIGHTS_PATH = Path(args.weights)
        config.CHECK_WEIGHTS_PATH = Path(args.weights)

    if args.mode == "train":
        run_train(args.lr)
    elif args.mode == "predict":
        run_predict()
    elif args.mode == "eval":
        run_eval()
    elif args.mode == "check-backend":
        run_check_backend()
    elif args.mode == "detect":
        run_detect(args.video, args.bbox)
