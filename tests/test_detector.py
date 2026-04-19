import numpy as np
from detector import TrafficDetector, DetectionResult

def test_detector_returns_valid_result():
    """Feeds a static black image to the detector to ensure it parses types correctly and doesn't crash."""
    detector = TrafficDetector()
    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    result = detector.detect(dummy_frame)
    
    assert isinstance(result, DetectionResult)
    assert isinstance(result.vehicle_count, int)
    assert isinstance(result.pedestrian_in_roi, int)
    assert isinstance(result.pink_vehicles, list)
    assert isinstance(result.annotated_frame, np.ndarray)
    
    # In a completely black frame, there should be no detections
    assert result.vehicle_count == 0
    assert result.pedestrian_in_roi == 0
    assert len(result.pink_vehicles) == 0
