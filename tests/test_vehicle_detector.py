from types import SimpleNamespace

from backend.app.vehicle_detector import run_hybrid_detection


def det(source, box=(1, 2, 30, 40)):
    return SimpleNamespace(class_name="car", confidence=0.9, bbox=box, source=source)


def test_fallback_runs_when_primary_fails():
    fallback = [det("yolo_fallback")]
    result = run_hybrid_detection(
        primary=lambda: (_ for _ in ()).throw(RuntimeError("down")),
        fallback=lambda: fallback,
        refine=lambda items: items,
    )
    assert result.detections == fallback
    assert result.mode == "yolo_fallback"


def test_refiner_replaces_successful_detections():
    result = run_hybrid_detection(
        primary=lambda: [det("rtdetr")],
        fallback=lambda: [],
        refine=lambda items: [det("rtdetr_seg", (3, 4, 28, 38))],
    )
    assert result.detections[0].source == "rtdetr_seg"
    assert result.mode == "rtdetr_seg"
