from backend.app.api_schemas import VehicleDetectionResponse


def test_vehicle_response_keeps_legacy_fields_and_optional_metadata():
    response = VehicleDetectionResponse.model_validate(
        {
            "video_id": 1,
            "total_detected": 1,
            "detected_vehicles": [{
                "id": 0,
                "class_name": "car",
                "confidence": 0.91,
                "bbox": [1, 2, 30, 40],
                "source": "rtdetr",
            }],
            "detector_mode": "rtdetr_seg",
        }
    )
    assert response.detected_vehicles[0].bbox == [1, 2, 30, 40]
    assert response.detected_vehicles[0].source == "rtdetr"
    assert response.detector_mode == "rtdetr_seg"
