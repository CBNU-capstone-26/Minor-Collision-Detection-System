"""라벨 확인·수정용 로컬 웹 도구 (서비스와 완전 분리).

목적: data/ 아래 영상의 '실제 충돌 프레임'을 눈으로 확인하고 A 라벨을 다듬는다.
HTML5 <video>의 currentTime seek은 코덱에 따라 프레임이 어긋날 수 있어,
**OpenCV로 정확한 인덱스의 프레임을 뽑아 JPEG로 서빙**한다(프레임 단위 정확).

실행:
    venv311/bin/python -m uvicorn labeling.server:app --port 8010
    → http://localhost:8010
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parent.parent
HERE = Path(__file__).resolve().parent

# 라벨 확인 대상 폴더 (프로젝트 루트 기준 상대경로)
VIDEO_DIRS = [
    "data/eval",
    "data/train/realdata",
    "data/train/rcdata",
]

app = FastAPI(title="Hit-and-run 라벨 확인 도구")


# ──────────────────────────── 라벨 입출력 ────────────────────────────
def label_path(video_rel: str) -> Path:
    return ROOT / (os.path.splitext(video_rel)[0] + ".txt")


def parse_label(video_rel: str) -> dict:
    """txt → {'boxes': {id: [x1,y1,x2,y2]}, 'cls': 'A'|'S'|None,
              'target_id': int, 'start_f': int, 'end_f': int}"""
    p = label_path(video_rel)
    out = {"boxes": {}, "cls": None, "target_id": 0,
           "start_f": None, "end_f": None, "exists": p.exists()}
    if not p.exists():
        return out
    for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
        parts = [t.strip() for t in line.strip().split(",")]
        if len(parts) < 2:
            continue
        if parts[0] == "car" and len(parts) >= 6:
            try:
                out["boxes"][int(parts[1])] = [int(float(v)) for v in parts[2:6]]
            except ValueError:
                pass
        elif parts[0] in ("A", "S"):
            out["cls"] = parts[0]
            try:
                out["target_id"] = int(float(parts[1]))
                if len(parts) > 2:
                    out["start_f"] = int(float(parts[2]))
                if len(parts) > 3:
                    out["end_f"] = int(float(parts[3]))
            except ValueError:
                pass
    return out


def write_label(video_rel: str, data: dict) -> None:
    """라벨 저장. 최초 저장 시 원본을 .txt.bak 으로 1회 백업한다."""
    p = label_path(video_rel)
    bak = p.with_suffix(".txt.bak")
    if p.exists() and not bak.exists():
        shutil.copy2(p, bak)          # 원본 보존(최초 1회만)
    lines = []
    for bid in sorted(data["boxes"]):
        x1, y1, x2, y2 = data["boxes"][bid]
        lines.append(f"car,{bid},{x1},{y1},{x2},{y2}")
    # 기존 데이터 규약: 충돌만 'A,...' 한 줄을 붙이고, 비충돌은 아무 줄도 없다.
    # (dataset.py도 A 라인이 없으면 class 'S'로 처리) → S일 땐 라인을 쓰지 않는다.
    if data.get("cls") == "A" and data.get("start_f") is not None:
        end_f = data.get("end_f")
        end_f = data["start_f"] if end_f is None else end_f
        lines.append(f"A,{data['target_id']},{data['start_f']},{end_f}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ──────────────────────────── 프레임 서빙 ────────────────────────────
# VideoCapture를 영상별로 캐시하고, 순차 재생일 때는 seek 없이 read만 해서
# 프레임 정확도와 속도를 동시에 얻는다(코덱 seek 부정확 회피).
_caps: dict[str, tuple[cv2.VideoCapture, int]] = {}


def get_frame(video_rel: str, idx: int):
    abs_path = ROOT / video_rel
    if not abs_path.exists():
        raise HTTPException(404, f"영상 없음: {video_rel}")
    cap, nxt = _caps.get(video_rel, (None, -1))
    if cap is None:
        cap = cv2.VideoCapture(str(abs_path))
        nxt = 0
    # 바로 다음 프레임이거나 가까운 앞쪽이면 순차 read(정확·빠름), 아니면 seek
    if idx == nxt:
        pass
    elif nxt < idx <= nxt + 60:
        for _ in range(idx - nxt):
            cap.read()
    else:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok, frame = cap.read()
    _caps[video_rel] = (cap, idx + 1 if ok else -1)
    if not ok:
        raise HTTPException(404, f"{idx}번 프레임을 읽을 수 없습니다")
    return frame


# ──────────────────────────── API ────────────────────────────
@app.get("/")
def index():
    return FileResponse(HERE / "index.html")


@app.get("/api/videos")
def list_videos():
    """대상 폴더의 영상 목록 + 메타 + 라벨 요약."""
    items = []
    for d in VIDEO_DIRS:
        folder = ROOT / d
        if not folder.exists():
            continue
        for mp4 in sorted(folder.glob("*.mp4")):
            rel = str(mp4.relative_to(ROOT))
            cap = cv2.VideoCapture(str(mp4))
            fps = round(cap.get(cv2.CAP_PROP_FPS) or 0, 2)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            cap.release()
            lab = parse_label(rel)
            items.append({
                "rel": rel, "dir": d, "name": mp4.name,
                "fps": fps, "total": total, "width": w, "height": h,
                "cls": lab["cls"], "start_f": lab["start_f"],
                "end_f": lab["end_f"], "n_boxes": len(lab["boxes"]),
                "span": (None if lab["start_f"] is None or lab["end_f"] is None
                         else lab["end_f"] - lab["start_f"]),
            })
    return {"videos": items}


@app.get("/api/label")
def get_label(video: str):
    return parse_label(video)


class LabelIn(BaseModel):
    video: str
    boxes: dict[str, list[int]]
    cls: str | None = None
    target_id: int = 0
    start_f: int | None = None
    end_f: int | None = None


@app.post("/api/label")
def save_label(body: LabelIn):
    data = {
        "boxes": {int(k): v for k, v in body.boxes.items()},
        "cls": body.cls, "target_id": body.target_id,
        "start_f": body.start_f, "end_f": body.end_f,
    }
    write_label(body.video, data)
    return {"ok": True, "saved": str(label_path(body.video))}


@app.get("/api/frame")
def frame(video: str, idx: int = 0, box: int = 1, w: int = 0):
    """정확한 idx 프레임을 JPEG로. box=1이면 라벨 bbox를 그려서 준다."""
    img = get_frame(video, max(0, idx))
    if box:
        lab = parse_label(video)
        tid = lab["target_id"]
        for bid, (x1, y1, x2, y2) in lab["boxes"].items():
            is_target = (bid == tid)
            color = (0, 0, 255) if is_target else (0, 200, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 3 if is_target else 2)
            tag = f"#{bid}" + (" (기준차량)" if is_target else "")
            cv2.putText(img, tag, (x1, max(y1 - 6, 14)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    if w and w > 0 and img.shape[1] > w:
        scale = w / img.shape[1]
        img = cv2.resize(img, (w, int(img.shape[0] * scale)))
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if not ok:
        raise HTTPException(500, "인코딩 실패")
    return Response(buf.tobytes(), media_type="image/jpeg")
