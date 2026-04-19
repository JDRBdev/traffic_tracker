from dataclasses import dataclass
import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger

from config import settings

@dataclass
class DetectionResult:
    vehicle_count: int
    pedestrian_in_roi: int
    pink_vehicles: list[np.ndarray]
    annotated_frame: np.ndarray

class TrafficDetector:
    def __init__(self):
        logger.info(f"Loading YOLO model: {settings.YOLO_MODEL}")
        self.model = YOLO(settings.YOLO_MODEL)
        
        self.roi_polygon = np.array(settings.ROI_POLYGON, np.int32)
        
        # COCO class indices: person=0, car=2, motorcycle=3, bus=5, truck=7
        self.person_class = 0
        self.vehicle_classes = {2, 3, 5, 7}

    def update_roi(self, new_polygon: list[list[int]]) -> None:
        """Hot-reloads the ROI polygon."""
        self.roi_polygon = np.array(new_polygon, np.int32)
        logger.info(f"ROI updated to {len(self.roi_polygon)} vertices")

    def _is_pink(self, crop: np.ndarray) -> bool:
        """Detects if a given vehicle crop has a significant amount of pink pixels."""
        if crop.size == 0:
            return False
            
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h, w = hsv.shape[:2]
        
        # Dual HSV range to wrap around the hue boundary
        lower_pink1 = np.array([settings.PINK_HUE_LOW, settings.PINK_SAT_LOW, settings.PINK_VAL_LOW])
        upper_pink1 = np.array([settings.PINK_HUE_HIGH, 255, 255])
        
        lower_pink2 = np.array([0, settings.PINK_SAT_LOW, settings.PINK_VAL_LOW])
        upper_pink2 = np.array([10, 255, 255])
        
        mask = (cv2.inRange(hsv, lower_pink1, upper_pink1) | 
                cv2.inRange(hsv, lower_pink2, upper_pink2))
        
        ratio = cv2.countNonZero(mask) / (h * w)
        return ratio >= settings.PINK_PIXEL_THRESHOLD

    def _draw_roi(self, frame: np.ndarray) -> None:
        """Draws the region of interest polygon in green."""
        cv2.polylines(frame, [self.roi_polygon], isClosed=True, color=(0, 255, 0), thickness=2)

    def detect(self, frame: np.ndarray) -> DetectionResult:
        """Runs YOLO detection, checks ROI, and applies pink filter."""
        annotated_frame = frame.copy()
        self._draw_roi(annotated_frame)
        
        # Run inference (disable verbose output for speed)
        results = self.model(frame, verbose=False)
        
        vehicle_count = 0
        pedestrian_in_roi = 0
        pink_vehicles = []
        
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls = int(box.cls[0])
                # conf = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                
                if cls == self.person_class:
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                    # Check if inside ROI
                    if cv2.pointPolygonTest(self.roi_polygon, (cx, cy), False) >= 0:
                        pedestrian_in_roi += 1
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 165, 255), 2)
                        cv2.circle(annotated_frame, (cx, cy), 4, (0, 165, 255), -1)
                        cv2.putText(annotated_frame, "Person (ROI)", (x1, max(0, y1 - 10)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 2)
                        
                elif cls in self.vehicle_classes:
                    vehicle_count += 1
                    crop = frame[max(0, y1):y2, max(0, x1):x2]
                    
                    is_pink = self._is_pink(crop)
                    color = (255, 0, 255) if is_pink else (255, 0, 0)
                    
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    label = "Pink Vehicle" if is_pink else "Vehicle"
                    cv2.putText(annotated_frame, label, (x1, max(0, y1 - 10)), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                    
                    if is_pink:
                        pink_vehicles.append(crop)
                        
        return DetectionResult(
            vehicle_count=vehicle_count,
            pedestrian_in_roi=pedestrian_in_roi,
            pink_vehicles=pink_vehicles,
            annotated_frame=annotated_frame
        )
