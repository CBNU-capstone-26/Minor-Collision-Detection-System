# YOLO Segmentation Auto BBox

`auto_bbox_yolo_seg.py`는 YOLO segmentation mask를 이용해 차량의 상/하/좌/우에 최대한 맞는 bbox를 자동 생성합니다.

## 설치

```bash
pip install ultralytics opencv-python
```

## 이미지에서 bbox 생성

```bash
python model/auto_bbox_yolo_seg.py \
  --source input.jpg \
  --output-txt outputs/input_auto_bbox.txt \
  --output-image outputs/input_auto_bbox.jpg
```

## 영상 특정 프레임에서 bbox 생성

```bash
python model/auto_bbox_yolo_seg.py \
  --source data/real/real01.mp4 \
  --frame-index 0 \
  --output-txt data/real/real01.txt \
  --output-image outputs/real01_auto_bbox.jpg
```

## A/S 라벨까지 같이 저장

```bash
python model/auto_bbox_yolo_seg.py \
  --source data/train/sample.mp4 \
  --frame-index 45 \
  --output-txt data/train/sample.txt \
  --output-image outputs/sample_auto_bbox.jpg \
  --label-class A \
  --target-id 0 \
  --start-frame 45
```

저장되는 txt 형식:

```text
car,0,x1,y1,x2,y2
car,1,x1,y1,x2,y2
A,0,45
```

## 알고리즘 흐름

1. `yolov8n-seg.pt`로 차량 후보를 찾습니다.
2. `car`, `truck`, `bus`, `motorcycle` 클래스만 남깁니다.
3. detection bbox 대신 segmentation mask의 실제 외곽 좌표를 계산합니다.
4. mask가 없거나 너무 작으면 YOLO 기본 bbox로 fallback합니다.
5. 프로젝트 학습 포맷인 `car,id,x1,y1,x2,y2`로 저장합니다.
