"""추가 영상 데이터용 A/S 라벨 txt 일괄 생성 도구.

기존 학습 데이터셋은 같은 이름의 mp4/txt 쌍을 읽는다.
예: data/train/sample01.mp4, data/train/sample01.txt

txt 마지막 줄에 다음 형식이 있으면 학습 라벨로 사용된다.
  A,target_id,start_frame  # 사고
  S,target_id,start_frame  # 비사고

이 스크립트는 새로 가져온 영상 약 150개에 대해 위 txt 파일을 자동 생성한다.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".m4v"}
ACCIDENT_KEYWORDS = {"a", "accident", "crash", "hit", "collision", "사고"}
SAFE_KEYWORDS = {"s", "safe", "normal", "no_accident", "non_accident", "비사고"}


@dataclass
class VideoLabel:
    video_path: Path
    label_class: str
    target_id: int = 0
    start_frame: int = 0


def normalize_label(label: str) -> str:
    """다양한 입력 라벨을 학습 포맷 A/S로 정규화한다."""
    value = label.strip().lower()
    if value in ACCIDENT_KEYWORDS:
        return "A"
    if value in SAFE_KEYWORDS:
        return "S"
    raise ValueError(f"라벨은 A 또는 S로 해석 가능해야 합니다: {label}")


def collect_video_paths(input_dir: Path, limit: int) -> list[Path]:
    """입력 폴더 아래 영상 파일을 최대 limit개까지 찾는다."""
    videos = [
        path
        for path in sorted(input_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS
    ]
    return videos[:limit]


def infer_label_from_path(video_path: Path) -> str:
    """상위 폴더명이나 파일명에 들어간 키워드로 A/S를 추정한다."""
    candidates = [video_path.stem, *[parent.name for parent in video_path.parents]]
    for candidate in candidates:
        lowered = candidate.lower()
        tokens = lowered.replace("-", "_").split("_")
        if lowered in ACCIDENT_KEYWORDS or any(token in ACCIDENT_KEYWORDS for token in tokens):
            return "A"
        if lowered in SAFE_KEYWORDS or any(token in SAFE_KEYWORDS for token in tokens):
            return "S"
    raise ValueError(
        f"파일명/폴더명에서 A/S 라벨을 추정하지 못했습니다: {video_path}"
    )


def load_labels_from_csv(csv_path: Path) -> dict[str, tuple[str, int, int]]:
    """CSV에서 파일별 라벨을 읽는다.

    CSV 헤더 예:
      file,label,target_id,start_frame
      accident01.mp4,A,0,35
      safe01.mp4,S,0,0
    """
    labels: dict[str, tuple[str, int, int]] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        required = {"file", "label"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV에는 file,label 헤더가 필요합니다.")

        for row in reader:
            file_name = (row.get("file") or "").strip()
            if not file_name:
                continue
            label_class = normalize_label(row["label"])
            target_id = int(row.get("target_id") or 0)
            start_frame = int(row.get("start_frame") or 0)
            labels[file_name] = (label_class, target_id, start_frame)
            labels[Path(file_name).stem] = (label_class, target_id, start_frame)
    return labels


def build_video_labels(
    input_dir: Path,
    limit: int = 150,
    label_class: str | None = None,
    label_csv: Path | None = None,
    target_id: int = 0,
    start_frame: int = 0,
) -> list[VideoLabel]:
    """추가 영상 목록을 A/S 라벨 정보로 변환한다."""
    videos = collect_video_paths(input_dir, limit)
    csv_labels = load_labels_from_csv(label_csv) if label_csv else {}
    fixed_label = normalize_label(label_class) if label_class else None

    video_labels: list[VideoLabel] = []
    for video_path in videos:
        if video_path.name in csv_labels:
            csv_label, csv_target_id, csv_start_frame = csv_labels[video_path.name]
            video_labels.append(
                VideoLabel(video_path, csv_label, csv_target_id, csv_start_frame)
            )
        elif video_path.stem in csv_labels:
            csv_label, csv_target_id, csv_start_frame = csv_labels[video_path.stem]
            video_labels.append(
                VideoLabel(video_path, csv_label, csv_target_id, csv_start_frame)
            )
        else:
            video_labels.append(
                VideoLabel(
                    video_path=video_path,
                    label_class=fixed_label or infer_label_from_path(video_path),
                    target_id=target_id,
                    start_frame=start_frame,
                )
            )
    return video_labels


def write_label_txt(
    output_txt: Path,
    label_class: str,
    target_id: int = 0,
    start_frame: int = 0,
    overwrite: bool = False,
) -> bool:
    """학습 코드가 읽는 A/S 라벨 txt를 생성한다."""
    if output_txt.exists() and not overwrite:
        return False

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    with output_txt.open("w", encoding="utf-8") as file:
        file.write(f"{label_class},{target_id},{start_frame}\n")
    return True


def create_label_txt_files(
    input_dir: Path,
    output_dir: Path | None = None,
    limit: int = 150,
    label_class: str | None = None,
    label_csv: Path | None = None,
    target_id: int = 0,
    start_frame: int = 0,
    copy_videos: bool = False,
    overwrite: bool = False,
) -> tuple[int, int]:
    """추가 영상 데이터의 txt 라벨을 일괄 생성한다.

    Returns:
        (생성한 txt 개수, 건너뛴 txt 개수)
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve() if output_dir else input_dir
    video_labels = build_video_labels(
        input_dir=input_dir,
        limit=limit,
        label_class=label_class,
        label_csv=label_csv,
        target_id=target_id,
        start_frame=start_frame,
    )

    created = 0
    skipped = 0
    for item in video_labels:
        target_video = output_dir / item.video_path.name
        target_txt = output_dir / f"{item.video_path.stem}.txt"

        if copy_videos:
            output_dir.mkdir(parents=True, exist_ok=True)
            if overwrite or not target_video.exists():
                shutil.copy2(item.video_path, target_video)

        if write_label_txt(
            target_txt,
            item.label_class,
            item.target_id,
            item.start_frame,
            overwrite=overwrite,
        ):
            created += 1
        else:
            skipped += 1

    return created, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="추가 영상 A/S 라벨 txt 일괄 생성")
    parser.add_argument("--input-dir", required=True, help="추가 영상들이 있는 폴더")
    parser.add_argument(
        "--output-dir",
        help="txt를 저장할 폴더. 생략하면 input-dir 옆에 저장",
    )
    parser.add_argument("--limit", type=int, default=150, help="처리할 최대 영상 수")
    parser.add_argument(
        "--label-class",
        choices=("A", "S", "a", "s"),
        help="전체 영상을 같은 라벨로 저장할 때 사용",
    )
    parser.add_argument(
        "--label-csv",
        help="파일별 라벨 CSV 경로. 헤더: file,label,target_id,start_frame",
    )
    parser.add_argument("--target-id", type=int, default=0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument(
        "--copy-videos",
        action="store_true",
        help="영상도 output-dir로 함께 복사",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="기존 txt가 있어도 덮어쓰기",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    created, skipped = create_label_txt_files(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        limit=args.limit,
        label_class=args.label_class,
        label_csv=Path(args.label_csv) if args.label_csv else None,
        target_id=args.target_id,
        start_frame=args.start_frame,
        copy_videos=args.copy_videos,
        overwrite=args.overwrite,
    )
    print(f"created txt files: {created}")
    print(f"skipped txt files: {skipped}")


if __name__ == "__main__":
    main()
