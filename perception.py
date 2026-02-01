"""
perception.py - AI Object Detection and Classification Module

This module provides object detection and classification capabilities
using OpenCV and optionally YOLO for more advanced detection.

Supports:
- Color-based detection (HSV filtering)
- Shape detection (contours + geometry analysis)
- YOLO integration (optional, for robust detection)
"""

import numpy as np
import cv2
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from enum import Enum


class ObjectShape(Enum):
    """Detected object shapes."""
    UNKNOWN = "unknown"
    CUBE = "cube"
    SPHERE = "sphere"
    CYLINDER = "cylinder"
    RECTANGLE = "rectangle"


class ObjectColor(Enum):
    """Detectable colors."""
    RED = "red"
    GREEN = "green"
    BLUE = "blue"
    YELLOW = "yellow"
    ORANGE = "orange"
    WHITE = "white"
    BLACK = "black"
    UNKNOWN = "unknown"


@dataclass
class DetectedObject:
    """
    Represents a detected object in the scene.
    """
    id: int
    shape: ObjectShape
    color: ObjectColor
    
    # Bounding box (pixel coordinates)
    bbox: Tuple[int, int, int, int]  # (x, y, w, h)
    
    # Center in pixels
    center_px: Tuple[int, int]
    
    # Center in world coordinates (if transformer available)
    center_world: Optional[Tuple[float, float, float]] = None
    
    # Estimated size
    width_px: int = 0
    height_px: int = 0
    estimated_size_m: float = 0.0
    
    # Detection confidence (0-1)
    confidence: float = 0.0
    
    # Rotation angle (degrees)
    rotation: float = 0.0
    
    def __str__(self):
        return f"{self.color.value} {self.shape.value} @ {self.center_px}"


class ObjectDetector:
    """
    Detects and classifies objects using computer vision.
    """
    
    # HSV color ranges for detection
    COLOR_RANGES: Dict[ObjectColor, Tuple[np.ndarray, np.ndarray]] = {
        ObjectColor.RED: (np.array([0, 100, 100]), np.array([10, 255, 255])),
        ObjectColor.GREEN: (np.array([40, 50, 50]), np.array([80, 255, 255])),
        ObjectColor.BLUE: (np.array([100, 100, 100]), np.array([130, 255, 255])),
        ObjectColor.YELLOW: (np.array([20, 100, 100]), np.array([35, 255, 255])),
        ObjectColor.ORANGE: (np.array([10, 100, 100]), np.array([20, 255, 255])),
    }
    
    def __init__(self, min_area: int = 500, max_area: int = 50000):
        """
        Initialize the detector.
        
        Args:
            min_area: Minimum contour area to consider (pixels²)
            max_area: Maximum contour area to consider (pixels²)
        """
        self.min_area = min_area
        self.max_area = max_area
        self.object_counter = 0
        
        # YOLO model (loaded on demand)
        self.yolo_net = None
        self.yolo_classes = []
        
        # Known object size for distance estimation (meters)
        self.reference_object_size = 0.03  # 3cm default
    
    def detect_objects(self, frame: np.ndarray, 
                       use_yolo: bool = False) -> List[DetectedObject]:
        """
        Detect all objects in the frame.
        
        Args:
            frame: BGR image from camera
            use_yolo: Use YOLO for detection (requires model)
            
        Returns:
            List of DetectedObject instances
        """
        objects = []
        
        if use_yolo and self.yolo_net is not None:
            objects = self._detect_yolo(frame)
        else:
            objects = self._detect_color_shape(frame)
        
        return objects
    
    def _detect_color_shape(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detect objects using color and shape analysis."""
        objects = []
        
        # Convert to HSV
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Detect each color
        for color, (lower, upper) in self.COLOR_RANGES.items():
            # Special handling for red (wraps around)
            if color == ObjectColor.RED:
                mask1 = cv2.inRange(hsv, lower, upper)
                mask2 = cv2.inRange(hsv, np.array([170, 100, 100]), 
                                    np.array([180, 255, 255]))
                mask = mask1 | mask2
            else:
                mask = cv2.inRange(hsv, lower, upper)
            
            # Clean up mask
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, 
                                           cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if self.min_area < area < self.max_area:
                    obj = self._analyze_contour(contour, color, frame)
                    if obj is not None:
                        objects.append(obj)
        
        return objects
    
    def _analyze_contour(self, contour: np.ndarray, 
                         color: ObjectColor,
                         frame: np.ndarray) -> Optional[DetectedObject]:
        """Analyze a contour to determine shape and properties."""
        # Bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Minimum enclosing circle
        (cx, cy), radius = cv2.minEnclosingCircle(contour)
        
        # Contour properties
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # Circularity: 4π × area / perimeter²
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Aspect ratio
        aspect_ratio = float(w) / h if h > 0 else 0
        
        # Rectangularity: area / bounding rect area
        rect_area = w * h
        rectangularity = area / rect_area if rect_area > 0 else 0
        
        # Approximate polygon
        epsilon = 0.04 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        
        # Classify shape
        shape = self._classify_shape(circularity, aspect_ratio, 
                                     rectangularity, vertices)
        
        # Get rotation angle
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)
            rotation = ellipse[2]
        else:
            rotation = 0
        
        self.object_counter += 1
        
        return DetectedObject(
            id=self.object_counter,
            shape=shape,
            color=color,
            bbox=(x, y, w, h),
            center_px=(int(cx), int(cy)),
            width_px=w,
            height_px=h,
            confidence=min(circularity + rectangularity, 1.0),
            rotation=rotation
        )
    
    def _classify_shape(self, circularity: float, aspect_ratio: float,
                        rectangularity: float, vertices: int) -> ObjectShape:
        """Classify shape based on geometric properties."""
        # High circularity = sphere
        if circularity > 0.8:
            return ObjectShape.SPHERE
        
        # Square-ish with high rectangularity = cube (top view)
        if 0.8 < aspect_ratio < 1.2 and rectangularity > 0.85:
            return ObjectShape.CUBE
        
        # Rectangular (elongated)
        if rectangularity > 0.8 and (aspect_ratio < 0.7 or aspect_ratio > 1.3):
            return ObjectShape.RECTANGLE
        
        # High circularity but not quite sphere = cylinder (top view)
        if 0.6 < circularity < 0.85:
            return ObjectShape.CYLINDER
        
        return ObjectShape.UNKNOWN
    
    def classify_object(self, roi: np.ndarray) -> Tuple[ObjectShape, ObjectColor]:
        """
        Classify a region of interest.
        
        Args:
            roi: BGR image of the object area
            
        Returns:
            tuple: (shape, color)
        """
        # Detect in ROI
        objects = self.detect_objects(roi)
        
        if objects:
            return objects[0].shape, objects[0].color
        
        return ObjectShape.UNKNOWN, ObjectColor.UNKNOWN
    
    def get_object_properties(self, obj: DetectedObject, 
                              pixels_per_meter: float = 1000) -> Dict:
        """
        Get detailed properties for grasp planning.
        
        Args:
            obj: Detected object
            pixels_per_meter: Scale factor from calibration
            
        Returns:
            dict: Properties including size, orientation, grasp points
        """
        # Estimate real-world size
        size_m = obj.width_px / pixels_per_meter
        
        # Suggested grasp approach based on shape
        if obj.shape == ObjectShape.CUBE:
            approach = "top-down"
            gripper_rotation = obj.rotation
        elif obj.shape == ObjectShape.SPHERE:
            approach = "top-down"
            gripper_rotation = 0
        elif obj.shape == ObjectShape.CYLINDER:
            approach = "side" if obj.width_px < obj.height_px else "top-down"
            gripper_rotation = obj.rotation + 90
        else:
            approach = "top-down"
            gripper_rotation = 0
        
        return {
            'estimated_size_m': size_m,
            'grasp_approach': approach,
            'gripper_rotation': gripper_rotation % 180,
            'center_offset': (0, 0),  # Can be adjusted for asymmetric objects
            'grip_width': size_m * 1.1,  # Slightly wider than object
        }
    
    def draw_detections(self, frame: np.ndarray, 
                        objects: List[DetectedObject]) -> np.ndarray:
        """
        Draw detection results on frame.
        
        Args:
            frame: Input image
            objects: List of detected objects
            
        Returns:
            np.ndarray: Annotated frame
        """
        annotated = frame.copy()
        
        for obj in objects:
            x, y, w, h = obj.bbox
            cx, cy = obj.center_px
            
            # Color for drawing
            if obj.color == ObjectColor.RED:
                draw_color = (0, 0, 255)
            elif obj.color == ObjectColor.GREEN:
                draw_color = (0, 255, 0)
            elif obj.color == ObjectColor.BLUE:
                draw_color = (255, 0, 0)
            elif obj.color == ObjectColor.YELLOW:
                draw_color = (0, 255, 255)
            else:
                draw_color = (255, 255, 255)
            
            # Bounding box
            cv2.rectangle(annotated, (x, y), (x+w, y+h), draw_color, 2)
            
            # Center point
            cv2.circle(annotated, (cx, cy), 5, draw_color, -1)
            
            # Label
            label = f"{obj.color.value} {obj.shape.value}"
            cv2.putText(annotated, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, draw_color, 2)
        
        return annotated
    
    def load_yolo(self, config_path: str, weights_path: str, names_path: str):
        """
        Load YOLO model for advanced detection.
        
        Args:
            config_path: Path to .cfg file
            weights_path: Path to .weights file
            names_path: Path to .names file (class names)
        """
        try:
            self.yolo_net = cv2.dnn.readNet(weights_path, config_path)
            
            with open(names_path, 'r') as f:
                self.yolo_classes = [line.strip() for line in f.readlines()]
            
            print(f"✅ YOLO loaded with {len(self.yolo_classes)} classes")
        except Exception as e:
            print(f"❌ Failed to load YOLO: {e}")
    
    def _detect_yolo(self, frame: np.ndarray) -> List[DetectedObject]:
        """Detect objects using YOLO."""
        # TODO: Implement YOLO detection
        # This is a placeholder for YOLO integration
        return []


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🧠 Perception Module Test")
    print("=" * 60)
    
    # Create test image with colored shapes
    test_frame = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Draw red cube (square)
    cv2.rectangle(test_frame, (100, 100), (150, 150), (0, 0, 255), -1)
    
    # Draw blue sphere (circle)
    cv2.circle(test_frame, (300, 200), 30, (255, 0, 0), -1)
    
    # Draw green rectangle
    cv2.rectangle(test_frame, (450, 150), (550, 200), (0, 255, 0), -1)
    
    # Draw yellow cylinder (ellipse for top view)
    cv2.ellipse(test_frame, (200, 350), (40, 40), 0, 0, 360, (0, 255, 255), -1)
    
    # Detect
    detector = ObjectDetector()
    objects = detector.detect_objects(test_frame)
    
    print(f"\n🔍 Detected {len(objects)} objects:")
    for obj in objects:
        props = detector.get_object_properties(obj)
        print(f"   - {obj}")
        print(f"     Grasp: {props['grasp_approach']}, rotation: {props['gripper_rotation']:.1f}°")
    
    # Draw and show
    annotated = detector.draw_detections(test_frame, objects)
    
    cv2.imshow("Detection Test", annotated)
    print("\nPress any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
