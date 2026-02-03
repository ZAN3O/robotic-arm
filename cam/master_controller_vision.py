"""
master_controller_vision.py - Robot Control with Real Vision

Combine:
- Real-time camera object detection (YOLOv8)
- Robot arm control (PyBullet)
- Voice commands (optional)

Architecture:
    Camera → YOLO Detection → 3D Position → Robot Pick/Place

Usage:
    python master_controller_vision.py --mode interactive
    python master_controller_vision.py --mode voice
    python master_controller_vision.py --mode calibrate
"""

import pybullet as p
import pybullet_data
import time
import numpy as np
import math
from ikpy.chain import Chain
from ikpy.link import URDFLink, OriginLink
from typing import Optional, Tuple, List

# Import vision modules
try:
    from vision_advanced import (
        AdvancedObjectDetector, SmartCamera, DetectedObject3D,
        DetectionBackend, CameraSource, ObjectDatabase
    )
    VISION_AVAILABLE = True
except ImportError:
    VISION_AVAILABLE = False
    print("⚠️ vision_advanced.py non trouvé")

# Import voice control (optional)
try:
    from voice_control import VoiceController, CommandAction
    VOICE_AVAILABLE = True
except ImportError:
    VOICE_AVAILABLE = False

# Import homography
try:
    from vision import HomographyTransformer
    HOMOGRAPHY_AVAILABLE = True
except ImportError:
    HOMOGRAPHY_AVAILABLE = False


# ==============================================================================
# PHYSICS PARAMETERS
# ==============================================================================

URDF_PATH = "arduino_arm.urdf"
REAL_TIME = True
FPS = 240

BASE_HEIGHT = 0.11
HUMERUS_LENGTH = 0.13
ULNA_LENGTH = 0.13
GRIPPER_LENGTH = 0.12


# ==============================================================================
# KINEMATIC CHAIN
# ==============================================================================

def create_manual_chain():
    """Create kinematic chain."""
    return Chain(name="arduino_arm", links=[
        OriginLink(),
        URDFLink(
            name="base_rotation",
            origin_translation=[0, 0, BASE_HEIGHT],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1],
            bounds=(-2.5, 2.5)
        ),
        URDFLink(
            name="shoulder",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],
            bounds=(-1.57, 1.57)
        ),
        URDFLink(
            name="elbow",
            origin_translation=[0, 0, HUMERUS_LENGTH],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],
            bounds=(-1.57, 1.57)
        ),
        URDFLink(
            name="wrist",
            origin_translation=[0, 0, ULNA_LENGTH],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],
            bounds=(-1.57, 1.57)
        ),
        URDFLink(
            name="tip",
            origin_translation=[0, 0, GRIPPER_LENGTH],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 0],
        ),
    ], active_links_mask=[False, True, True, True, True, False])


# ==============================================================================
# ROBOT CLASS
# ==============================================================================

class RoboticArmWithVision:
    """Robot arm with vision capabilities."""
    
    def __init__(self, urdf_path, offset=[0, 0, 0]):
        """Initialize robot."""
        self.id = p.loadURDF(urdf_path, offset, useFixedBase=1)
        self.chain = create_manual_chain()
        self.active_joints = [0, 1, 2, 3]
        self.gripper_indices = [5, 6]
        
        # High friction
        for i in range(p.getNumJoints(self.id)):
            p.changeDynamics(self.id, i, lateralFriction=2.0,
                           spinningFriction=0.1, jointDamping=0.1)
        
        # Task tracking
        self.current_target: Optional[DetectedObject3D] = None
        self.task_history: List[str] = []
    
    def solve_ik(self, target_xyz):
        """Solve inverse kinematics."""
        ik_sol = self.chain.inverse_kinematics(target_position=target_xyz)
        return ik_sol[1:5]
    
    def move_smooth(self, target_xyz, gripper_open=True, duration=1.5):
        """Smooth movement to target."""
        print(f"   🎯 Moving to: [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]")
        
        target_angles = self.solve_ik(target_xyz)
        
        current_angles = []
        for j in self.active_joints:
            current_angles.append(p.getJointState(self.id, j)[0])
        start_angles = np.array(current_angles)
        
        steps = int(duration * FPS)
        for i in range(steps):
            t = i / steps
            cmd = start_angles + (target_angles - start_angles) * t
            
            # Apply joint commands
            for idx, joint_idx in enumerate(self.active_joints):
                p.setJointMotorControl2(self.id, joint_idx, p.POSITION_CONTROL,
                                      targetPosition=cmd[idx], force=100)
            
            # Gripper
            grip_pos = 0.0 if gripper_open else 0.5
            p.setJointMotorControl2(self.id, self.gripper_indices[0], p.POSITION_CONTROL,
                                  targetPosition=grip_pos, force=50)
            p.setJointMotorControl2(self.id, self.gripper_indices[1], p.POSITION_CONTROL,
                                  targetPosition=-grip_pos, force=50)
            
            p.stepSimulation()
            if REAL_TIME: time.sleep(1./FPS)
        
        # Stabilize
        for _ in range(10):
            p.stepSimulation()
            time.sleep(0.005)
    
    def home(self):
        """Return to home position."""
        print("🏠 Returning home...")
        self.move_smooth([0.15, 0, 0.20], gripper_open=True, duration=1.0)
    
    def pick_detected_object(self, obj: DetectedObject3D):
        """
        Pick an object detected by vision.
        
        Args:
            obj: Detected object with 3D position
        """
        if obj.position_3d is None:
            print("❌ Object has no 3D position (camera not calibrated?)")
            return False
        
        x, y, z = obj.position_3d
        
        print(f"\n🤖 PICK SEQUENCE: {obj.name}")
        print(f"   Position: ({x:.3f}, {y:.3f}, {z:.3f})")
        
        # 1. Approach from above
        print("   ⬆️  Approach...")
        self.move_smooth([x, y, 0.15], gripper_open=True, duration=1.0)
        
        # 2. Descend to object
        print("   ⬇️  Descend...")
        grasp_z = max(z + 0.01, 0.005)  # 1cm above center, min 5mm
        self.move_smooth([x, y, grasp_z], gripper_open=True, duration=1.2)
        
        # 3. Grasp
        print("   🦾 Closing gripper...")
        self.move_smooth([x, y, grasp_z], gripper_open=False, duration=0.5)
        time.sleep(0.3)
        
        # 4. Lift
        print("   🏋️ Lifting...")
        self.move_smooth([x, y, 0.20], gripper_open=False, duration=1.0)
        
        self.current_target = obj
        self.task_history.append(f"picked_{obj.name}")
        
        print(f"   ✅ {obj.name} picked!")
        return True
    
    def place_at(self, position: Tuple[float, float, float]):
        """Place held object at position."""
        if not self.current_target:
            print("⚠️ Not holding any object")
            return
        
        print(f"\n📦 PLACE SEQUENCE")
        
        # Transport
        print("   🚚 Transporting...")
        self.move_smooth(position, gripper_open=False, duration=1.5)
        
        # Release
        print("   🖐️ Releasing...")
        self.move_smooth(position, gripper_open=True, duration=0.5)
        
        self.task_history.append(f"placed_{self.current_target.name}")
        self.current_target = None
        
        print("   ✅ Object placed!")


# ==============================================================================
# INTEGRATED VISION + ROBOT CONTROLLER
# ==============================================================================

class VisionRobotSystem:
    """
    Complete system combining vision and robot control.
    """
    
    def __init__(self, camera_source: CameraSource = CameraSource.WEBCAM):
        """
        Initialize the system.
        
        Args:
            camera_source: Camera source type
        """
        # PyBullet
        self.physics_client = None
        self.robot: Optional[RoboticArmWithVision] = None
        
        # Vision
        self.camera: Optional[SmartCamera] = None
        self.detector: Optional[AdvancedObjectDetector] = None
        self.camera_source = camera_source
        
        # Calibration
        self.is_calibrated = False
        
        print("🔧 VisionRobotSystem initialized")
    
    def setup_simulation(self):
        """Setup PyBullet simulation."""
        print("\n🎮 Setting up simulation...")
        
        self.physics_client = p.connect(p.GUI)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.resetDebugVisualizerCamera(0.7, 0, -40, [0.25, 0, 0])
        
        # Environment
        p.loadURDF("plane.urdf")
        
        # Table
        p.createMultiBody(
            0,
            p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.02]),
            p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.02],
                               rgbaColor=[0.6, 0.4, 0.2, 1]),
            [0.5, 0, -0.02]
        )
        
        # Robot
        self.robot = RoboticArmWithVision(URDF_PATH)
        self.robot.home()
        
        print("✅ Simulation ready")
    
    def setup_vision(self, model_size: str = "yolov8n.pt"):
        """
        Setup vision system.
        
        Args:
            model_size: YOLO model (yolov8n/s/m/l/x)
        """
        if not VISION_AVAILABLE:
            print("❌ Vision modules not available")
            return False
        
        print("\n📷 Setting up vision...")
        
        # Camera
        self.camera = SmartCamera(source=self.camera_source)
        if not self.camera.open():
            return False
        
        # Detector
        self.detector = AdvancedObjectDetector(
            backend=DetectionBackend.YOLO,
            yolo_model=model_size,
            confidence_threshold=0.5
        )
        
        print("✅ Vision ready")
        return True
    
    def calibrate_camera(self):
        """
        Calibrate camera for 3D positioning.
        
        Interactive: places markers and clicks positions.
        """
        if not self.camera or not HOMOGRAPHY_AVAILABLE:
            print("❌ Camera or homography module not available")
            return False
        
        print("\n" + "=" * 60)
        print("📐 CAMERA CALIBRATION")
        print("=" * 60)
        print("Instructions:")
        print("1. Place 4 visible markers at KNOWN positions on table")
        print("2. Measure their X,Y positions in meters (from robot base)")
        print("3. Click on each marker in camera view")
        print("4. Enter their world coordinates")
        print("=" * 60)
        
        from vision import run_calibration_wizard
        
        # Run calibration wizard
        transformer = run_calibration_wizard(camera_id=0)
        
        if transformer:
            self.camera.transformer = transformer
            self.is_calibrated = True
            print("\n✅ Camera calibrated!")
            return True
        else:
            print("\n❌ Calibration failed")
            return False
    
    def detect_objects_live(self) -> List[DetectedObject3D]:
        """
        Detect objects in current camera view.
        
        Returns:
            List of detected objects with 3D positions (if calibrated)
        """
        if not self.camera or not self.detector:
            return []
        
        # Capture frame
        ret, frame = self.camera.read()
        if not ret:
            return []
        
        # Detect
        objects = self.detector.detect_frame(frame)
        
        # Add 3D coordinates if calibrated
        if self.is_calibrated:
            for obj in objects:
                self.camera.add_3d_coordinates(obj)
        
        return objects
    
    def find_and_pick(self, object_name: str) -> bool:
        """
        Find and pick a specific object by name.
        
        Args:
            object_name: Object to find (e.g., "bottle", "cup", "phone")
        """
        if not self.is_calibrated:
            print("⚠️ Camera not calibrated! Run calibration first.")
            return False
        
        print(f"\n🔍 Looking for: {object_name}")
        
        # Detect objects
        objects = self.detect_objects_live()
        
        if not objects:
            print("❌ No objects detected")
            return False
        
        # Find matching object
        target = None
        for obj in objects:
            if object_name.lower() in obj.name.lower():
                target = obj
                break
        
        if not target:
            print(f"❌ '{object_name}' not found")
            print(f"   Available objects: {[o.name for o in objects]}")
            return False
        
        print(f"✅ Found: {target}")
        
        # Pick it
        return self.robot.pick_detected_object(target)
    
    def run_interactive(self):
        """Interactive mode with text commands."""
        if not self.is_calibrated:
            print("⚠️ Run calibration first with --mode calibrate")
            return
        
        print("\n" + "=" * 60)
        print("🎮 INTERACTIVE MODE")
        print("=" * 60)
        print("Commands:")
        print("  pick <object>   - Pick an object (e.g., 'pick bottle')")
        print("  place           - Place held object")
        print("  home            - Return to home position")
        print("  scan            - Show detected objects")
        print("  quit            - Exit")
        print("=" * 60)
        
        while True:
            cmd = input("\n👉 Command: ").strip().lower()
            
            if cmd.startswith("pick "):
                obj_name = cmd.replace("pick ", "").strip()
                self.find_and_pick(obj_name)
            
            elif cmd == "place":
                # Default placement position
                self.robot.place_at([0.0, -0.25, 0.15])
            
            elif cmd == "home":
                self.robot.home()
            
            elif cmd == "scan":
                objects = self.detect_objects_live()
                print(f"\n📦 Detected objects ({len(objects)}):")
                for obj in objects:
                    print(f"   - {obj}")
            
            elif cmd in ["quit", "exit", "q"]:
                print("👋 Goodbye!")
                break
            
            else:
                print("❓ Unknown command")
    
    def run_with_voice(self):
        """Voice control mode."""
        if not VOICE_AVAILABLE:
            print("❌ Voice control not available")
            return
        
        if not self.is_calibrated:
            print("⚠️ Run calibration first")
            return
        
        print("\n🎤 VOICE CONTROL MODE")
        print("Say: 'Bonjour bras' to activate")
        print("Then: 'Prends le <objet>' / 'Pick the <object>'")
        
        vc = VoiceController(language="fr-FR")
        
        def process_command(intent):
            """Process voice command."""
            print(f"\n🗣️ Command: {intent.raw_text}")
            
            if intent.action == CommandAction.PICK:
                # Extract object name from target_color or raw text
                if intent.target_color:
                    # Try to find colored object
                    objects = self.detect_objects_live()
                    for obj in objects:
                        if (obj.dominant_color and 
                            intent.target_color in obj.dominant_color):
                            self.robot.pick_detected_object(obj)
                            return
                
                # Otherwise search by name in raw text
                words = intent.raw_text.lower().split()
                for word in words:
                    if len(word) > 3:  # Reasonable object name
                        if self.find_and_pick(word):
                            return
                
                print("❌ Could not identify target object")
            
            elif intent.action == CommandAction.PLACE:
                self.robot.place_at([0.0, -0.25, 0.15])
            
            elif intent.action == CommandAction.HOME:
                self.robot.home()
            
            vc.is_active = False  # Sleep after command
        
        try:
            vc.run_continuous(process_command)
        except KeyboardInterrupt:
            print("\n⚠️ Stopped")
    
    def shutdown(self):
        """Shutdown all systems."""
        if self.camera:
            self.camera.close()
        if self.physics_client:
            p.disconnect()
        print("🛑 System shutdown")


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Vision Robot Controller")
    parser.add_argument('--mode', choices=['calibrate', 'interactive', 'voice'],
                       default='interactive',
                       help="Operating mode")
    parser.add_argument('--model', default='yolov8n.pt',
                       choices=['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt'],
                       help="YOLO model size")
    parser.add_argument('--camera', choices=['webcam', 'iphone'],
                       default='webcam',
                       help="Camera source")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🤖 ROBOT VISION CONTROLLER")
    print("=" * 60)
    
    # Camera source
    cam_source = (CameraSource.IPHONE_CONTINUITY if args.camera == 'iphone'
                  else CameraSource.WEBCAM)
    
    # Create system
    system = VisionRobotSystem(camera_source=cam_source)
    
    # Setup
    system.setup_simulation()
    
    if not system.setup_vision(model_size=args.model):
        print("❌ Failed to setup vision")
        return
    
    try:
        if args.mode == 'calibrate':
            # Calibration mode
            system.calibrate_camera()
        
        elif args.mode == 'interactive':
            # Calibrate first if needed
            if not system.is_calibrated:
                print("📐 Calibration required...")
                if not system.calibrate_camera():
                    print("⚠️ Calibration failed but continuing...")
            
            # Interactive mode
            system.run_interactive()
        
        elif args.mode == 'voice':
            # Voice mode
            if not system.is_calibrated:
                print("📐 Calibration required...")
                if not system.calibrate_camera():
                    return
            
            system.run_with_voice()
    
    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")
    finally:
        system.shutdown()


if __name__ == "__main__":
    main()
