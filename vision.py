"""
vision.py - Camera to World Coordinate Transformation

This module handles the 2D->3D transformation using homography
for converting webcam pixel coordinates to robot workspace coordinates.

Note: The Z coordinate (height) must be estimated separately or
assumed constant for objects on the table surface.
"""

import numpy as np
import cv2
from typing import Tuple, List, Optional


class HomographyTransformer:
    """
    Transforms 2D pixel coordinates to 2D world coordinates using homography.
    
    For a top-down camera setup, this provides X and Y world coordinates.
    Z is typically assumed to be table height for objects.
    """
    
    def __init__(self):
        """Initialize the transformer."""
        self.H: Optional[np.ndarray] = None  # Homography matrix
        self.H_inv: Optional[np.ndarray] = None  # Inverse for world->pixel
        self.is_calibrated = False
        
        # Default table height (Z) for objects
        self.table_z = 0.0  # meters
    
    def calibrate(self, 
                  pixel_points: List[Tuple[float, float]],
                  world_points: List[Tuple[float, float]]) -> bool:
        """
        Calibrate the homography using corresponding points.
        
        You need at least 4 point correspondences. Place markers at known
        positions in the robot workspace and note their pixel coordinates.
        
        Args:
            pixel_points: List of (px, py) coordinates from camera
            world_points: List of (x, y) coordinates in robot frame (meters)
            
        Returns:
            bool: True if calibration successful
            
        Example:
            >>> # Corners of a 20x20cm square centered at robot
            >>> world_pts = [(0.0, 0.1), (0.2, 0.1), (0.2, -0.1), (0.0, -0.1)]
            >>> pixel_pts = [(100, 50), (540, 50), (540, 430), (100, 430)]
            >>> transformer.calibrate(pixel_pts, world_pts)
        """
        if len(pixel_points) < 4 or len(world_points) < 4:
            print("❌ Need at least 4 point correspondences")
            return False
        
        if len(pixel_points) != len(world_points):
            print("❌ Number of pixel and world points must match")
            return False
        
        # Convert to numpy arrays
        src = np.array(pixel_points, dtype=np.float32)
        dst = np.array(world_points, dtype=np.float32)
        
        # Calculate homography matrix
        self.H, status = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
        
        if self.H is None:
            print("❌ Homography calculation failed")
            return False
        
        # Calculate inverse for world->pixel transformation
        self.H_inv = np.linalg.inv(self.H)
        self.is_calibrated = True
        
        print(f"✅ Homography calibrated with {len(pixel_points)} points")
        return True
    
    def pixels_to_world(self, pixel_x: float, pixel_y: float) -> Tuple[float, float, float]:
        """
        Transform pixel coordinates to world coordinates.
        
        Args:
            pixel_x: X coordinate in image (pixels)
            pixel_y: Y coordinate in image (pixels)
            
        Returns:
            tuple: (world_x, world_y, world_z) in meters
            
        Note: world_z is assumed to be table_z (set during init or via set_table_z)
        """
        if not self.is_calibrated:
            print("⚠️ Transformer not calibrated, returning zeros")
            return (0.0, 0.0, self.table_z)
        
        # Homogeneous coordinates
        pixel_h = np.array([[pixel_x], [pixel_y], [1.0]])
        
        # Apply homography
        world_h = self.H @ pixel_h
        
        # Normalize
        world_x = float(world_h[0, 0] / world_h[2, 0])
        world_y = float(world_h[1, 0] / world_h[2, 0])
        
        return (world_x, world_y, self.table_z)
    
    def world_to_pixels(self, world_x: float, world_y: float) -> Tuple[int, int]:
        """
        Transform world coordinates back to pixel coordinates.
        Useful for visualization overlays.
        
        Args:
            world_x, world_y: World coordinates in meters
            
        Returns:
            tuple: (pixel_x, pixel_y) in image coordinates
        """
        if not self.is_calibrated:
            return (0, 0)
        
        world_h = np.array([[world_x], [world_y], [1.0]])
        pixel_h = self.H_inv @ world_h
        
        pixel_x = int(pixel_h[0, 0] / pixel_h[2, 0])
        pixel_y = int(pixel_h[1, 0] / pixel_h[2, 0])
        
        return (pixel_x, pixel_y)
    
    def set_table_z(self, height: float):
        """Set the assumed Z height for objects on the table."""
        self.table_z = height


def create_calibration_frame(frame: np.ndarray, 
                             pixel_points: List[Tuple[int, int]],
                             world_labels: List[str]) -> np.ndarray:
    """
    Draw calibration points on a frame for visual feedback.
    
    Args:
        frame: Input image
        pixel_points: List of pixel coordinates
        world_labels: List of labels like "(0.0, 0.1)"
        
    Returns:
        np.ndarray: Annotated frame
    """
    annotated = frame.copy()
    
    for (px, py), label in zip(pixel_points, world_labels):
        # Draw crosshair
        cv2.drawMarker(annotated, (px, py), (0, 255, 0), 
                       cv2.MARKER_CROSS, 20, 2)
        # Draw label
        cv2.putText(annotated, label, (px + 10, py - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
    
    return annotated


class CameraCapture:
    """
    Simple webcam capture wrapper with convenience methods.
    """
    
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        """
        Initialize camera capture.
        
        Args:
            camera_id: Camera device ID (0 for default webcam)
            width, height: Capture resolution
        """
        self.camera_id = camera_id
        self.cap = None
        self.width = width
        self.height = height
    
    def open(self) -> bool:
        """Open the camera."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"❌ Cannot open camera {self.camera_id}")
            return False
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        
        print(f"✅ Camera {self.camera_id} opened ({self.width}x{self.height})")
        return True
    
    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a frame from the camera."""
        if self.cap is None:
            return False, None
        return self.cap.read()
    
    def close(self):
        """Release the camera."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


# ============================================================================
# INTERACTIVE CALIBRATION TOOL
# ============================================================================
def run_calibration_wizard(camera_id: int = 0):
    """
    Interactive calibration wizard.
    
    Instructions:
    1. Place 4 markers at known positions in the robot workspace
    2. Click on each marker in the camera view
    3. Enter the world coordinates for each point
    
    Returns:
        HomographyTransformer: Calibrated transformer
    """
    print("=" * 60)
    print("📷 Camera Calibration Wizard")
    print("=" * 60)
    print("\nInstructions:")
    print("1. Place 4+ markers at known positions")
    print("2. Click on each marker in the camera view")
    print("3. Enter world coordinates (meters)")
    print("4. Press 'c' to calibrate, 'q' to quit\n")
    
    # Capture
    cap = CameraCapture(camera_id)
    if not cap.open():
        return None
    
    transformer = HomographyTransformer()
    pixel_points = []
    world_points = []
    
    click_pos = [None]
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            click_pos[0] = (x, y)
    
    cv2.namedWindow("Calibration")
    cv2.setMouseCallback("Calibration", mouse_callback)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Draw existing points
        for i, (px, py) in enumerate(pixel_points):
            cv2.circle(frame, (px, py), 5, (0, 255, 0), -1)
            cv2.putText(frame, str(i+1), (px+10, py), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Status
        cv2.putText(frame, f"Points: {len(pixel_points)}/4+", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, "Click to add, 'c' calibrate, 'q' quit", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        cv2.imshow("Calibration", frame)
        
        key = cv2.waitKey(1) & 0xFF
        
        # Check for click
        if click_pos[0] is not None:
            px, py = click_pos[0]
            click_pos[0] = None
            
            # Get world coordinates from user
            print(f"\n📍 Point {len(pixel_points)+1} at pixel ({px}, {py})")
            try:
                wx = float(input("   Enter world X (meters): "))
                wy = float(input("   Enter world Y (meters): "))
                
                pixel_points.append((px, py))
                world_points.append((wx, wy))
                print(f"   ✅ Added: ({px}, {py}) → ({wx:.3f}, {wy:.3f})")
            except ValueError:
                print("   ❌ Invalid input, point skipped")
        
        if key == ord('c') and len(pixel_points) >= 4:
            if transformer.calibrate(pixel_points, world_points):
                print("\n✅ Calibration successful!")
                break
            
        if key == ord('q'):
            print("\n❌ Calibration cancelled")
            transformer = None
            break
    
    cap.close()
    cv2.destroyAllWindows()
    
    return transformer


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("📷 Vision Module Test")
    print("=" * 60)
    
    # Create transformer with test calibration
    transformer = HomographyTransformer()
    
    # Simulated calibration data (example for 640x480 camera)
    # Assuming camera covers a 30x30cm workspace area
    pixel_pts = [
        (100, 100),   # Top-left
        (540, 100),   # Top-right
        (540, 380),   # Bottom-right
        (100, 380),   # Bottom-left
    ]
    
    world_pts = [
        (0.05, 0.15),   # Robot frame coordinates
        (0.25, 0.15),
        (0.25, -0.15),
        (0.05, -0.15),
    ]
    
    if transformer.calibrate(pixel_pts, world_pts):
        print("\n📍 Testing coordinate transformation:")
        
        test_pixels = [(320, 240), (200, 150), (450, 300)]
        
        for px, py in test_pixels:
            wx, wy, wz = transformer.pixels_to_world(px, py)
            print(f"   Pixel ({px}, {py}) → World ({wx:.3f}, {wy:.3f}, {wz:.3f})m")
        
        # Test reverse
        print("\n🔄 Testing reverse transformation:")
        for wx, wy in [(0.15, 0.0), (0.10, 0.10)]:
            px, py = transformer.world_to_pixels(wx, wy)
            print(f"   World ({wx:.2f}, {wy:.2f}) → Pixel ({px}, {py})")
    
    # Uncomment to run interactive calibration:
    # run_calibration_wizard()
