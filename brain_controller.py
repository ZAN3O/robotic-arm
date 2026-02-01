#!/usr/bin/env python3
"""
brain_controller.py - Main Brain Controller for HowToMechatronics 4-DOF Arm

This is the main orchestration script that integrates all modules:
- Kinematics (IK)
- PyBullet Simulation (Digital Twin)
- Vision (Camera + Homography)
- Perception (Object Detection + Classification)
- Grasp Planning
- Voice Control (Google Speech Recognition)
- Network (ZeroMQ to Raspberry Pi)

Usage:
    python brain_controller.py              # Run demo mode
    python brain_controller.py --gui        # Interactive PyBullet GUI
    python brain_controller.py --voice      # Voice control mode
    python brain_controller.py --camera     # Live camera mode
"""

import argparse
import time
import sys
from typing import Optional, List, Tuple

# Import all our modules
from kinematics import (
    create_kinematic_chain, 
    get_servo_angles, 
    validate_angles,
    check_reachability,
    BASE_HEIGHT, TOTAL_REACH
)
from simulation import RobotSim
from vision import HomographyTransformer, CameraCapture
from perception import ObjectDetector, DetectedObject, ObjectShape, ObjectColor
from grasp_planner import GraspPlanner, PickAndPlace
from voice_control import VoiceController, CommandAction
from network import NetworkController, format_command_for_display


class BrainController:
    """
    Main robot brain controller coordinating all subsystems.
    """
    
    def __init__(self, 
                 use_simulation: bool = True,
                 use_camera: bool = False,
                 use_voice: bool = False,
                 network_mode: str = "simulated"):
        """
        Initialize the brain controller.
        
        Args:
            use_simulation: Enable PyBullet visualization
            use_camera: Enable camera input
            use_voice: Enable voice control
            network_mode: "simulated", "zmq", or "serial"
        """
        print("=" * 60)
        print("🧠 HowToMechatronics Robot Brain Controller")
        print("=" * 60)
        print(f"\n📊 Configuration:")
        print(f"   Simulation: {'✅' if use_simulation else '❌'}")
        print(f"   Camera: {'✅' if use_camera else '❌'}")
        print(f"   Voice: {'✅' if use_voice else '❌'}")
        print(f"   Network: {network_mode}")
        print()
        
        # Initialize kinematic chain
        print("🦾 Initializing kinematics...")
        self.chain = create_kinematic_chain()
        print(f"   Total reach: {TOTAL_REACH * 100:.1f}cm")
        
        # Initialize simulation
        self.sim: Optional[RobotSim] = None
        if use_simulation:
            print("🎮 Initializing PyBullet simulation...")
            self.sim = RobotSim(urdf_path="arduino_arm.urdf", headless=False)
        
        # Initialize vision
        self.camera: Optional[CameraCapture] = None
        self.transformer = HomographyTransformer()
        if use_camera:
            print("📷 Initializing camera...")
            self.camera = CameraCapture()
            # Note: Calibration needed before use
        
        # Initialize perception
        print("👁️ Initializing object detector...")
        self.detector = ObjectDetector()
        
        # Initialize grasp planner
        print("🤚 Initializing grasp planner...")
        self.grasp_planner = GraspPlanner(table_height=0.0)
        self.pick_place = PickAndPlace(self.grasp_planner)
        
        # Initialize voice control
        self.voice: Optional[VoiceController] = None
        if use_voice:
            print("🎤 Initializing voice control (Google Speech)...")
            self.voice = VoiceController(language="fr-FR")
        
        # Initialize network
        print("📡 Initializing network...")
        self.network = NetworkController(mode=network_mode)
        self.network.connect()
        
        # State
        self.current_angles = [0, 0, 0, 0]
        self.current_gripper = 50  # 50% open
        self.detected_objects: List[DetectedObject] = []
        
        print("\n✅ Brain controller initialized!\n")
    
    def move_to_position(self, x: float, y: float, z: float, 
                         gripper: int = None,
                         speed: int = 50) -> bool:
        """
        Move robot to target position.
        
        Args:
            x, y, z: Target position in meters
            gripper: Gripper position (0-100), None to keep current
            speed: Movement speed (0-100)
            
        Returns:
            bool: True if movement successful
        """
        print(f"📍 Moving to ({x:.3f}, {y:.3f}, {z:.3f})m")
        
        # Check reachability
        if not check_reachability(x, y, z):
            print(f"   ❌ Position unreachable")
            return False
        
        # Calculate IK
        angles, success = get_servo_angles(x, y, z, self.chain)
        
        if not success:
            print(f"   ⚠️ IK solution may be imprecise")
        
        # Validate angles
        angles, valid = validate_angles(angles)
        
        # Update state
        self.current_angles = angles
        if gripper is not None:
            self.current_gripper = gripper
        
        # Display
        print(f"   {format_command_for_display(angles, self.current_gripper)}")
        
        # Simulate if enabled
        if self.sim:
            self.sim.move_to_angles(angles, speed=speed/50.0, animate=True)
        
        # Send to network
        self.network.send_to_network(angles, 
                                      gripper=self.current_gripper, 
                                      speed=speed)
        
        return True
    
    def move_to_angles(self, angles: List[float], 
                       gripper: int = None,
                       speed: int = 50) -> bool:
        """
        Move robot to specific joint angles.
        
        Args:
            angles: List of 4 angles in degrees
            gripper: Gripper position (0-100)
            speed: Movement speed
            
        Returns:
            bool: True if successful
        """
        # Validate
        angles, valid = validate_angles(angles)
        
        if not valid:
            print(f"   ⚠️ Some angles were clamped to limits")
        
        # Update state
        self.current_angles = angles
        if gripper is not None:
            self.current_gripper = gripper
        
        # Display
        print(f"🔧 {format_command_for_display(angles, self.current_gripper)}")
        
        # Simulate
        if self.sim:
            self.sim.move_to_angles(angles, speed=speed/50.0)
        
        # Send to network
        self.network.send_to_network(angles, 
                                      gripper=self.current_gripper,
                                      speed=speed)
        
        return True
    
    def go_home(self):
        """Return to home position."""
        print("🏠 Returning home...")
        self.move_to_angles([0, 0, 0, 0], gripper=50, speed=30)
    
    def detect_and_pick(self, target_color: str = None, 
                        target_shape: str = None) -> bool:
        """
        Find an object in the simulation and pick it up.
        
        Args:
            target_color: Color to look for ("red", "blue", etc.)
            target_shape: Shape to look for ("cube", "sphere", etc.)
            
        Returns:
            bool: True if object found and picked
        """
        # Use simulation objects (no camera needed)
        if not self.sim:
            print("❌ Simulation not initialized")
            return False
        
        # Get objects from simulation scene
        scene_objects = getattr(self.sim, 'scene_objects', [])
        
        if not scene_objects:
            print("❌ No objects in scene")
            return False
        
        # Find matching object
        target = None
        for obj in scene_objects:
            # Parse color from name (e.g., "red_cube" -> "red")
            obj_color = obj['name'].split('_')[0] if '_' in obj['name'] else None
            obj_shape = obj['shape']
            
            color_match = (target_color is None or 
                          target_color.lower() in obj['name'].lower())
            shape_match = (target_shape is None or 
                          target_shape.lower() == obj_shape.lower())
            
            if color_match and shape_match:
                target = obj
                break
        
        if target is None:
            print(f"❌ No matching object found (color={target_color}, shape={target_shape})")
            print(f"   Available: {[o['name'] for o in scene_objects]}")
            return False
        
        # Get object position from simulation
        import pybullet as p
        pos, _ = p.getBasePositionAndOrientation(target['id'])
        wx, wy, wz = pos[0], pos[1], pos[2]
        
        print(f"✅ Found {target['name']} at ({wx:.3f}, {wy:.3f}, {wz:.3f})m")
        
        # Pick sequence
        approach_height = 0.15
        grasp_height = wz + 0.02  # Slightly above object
        
        # 1. Open gripper
        print("   Opening gripper...")
        self.set_gripper(100)
        time.sleep(0.3)
        
        # 2. Move above object
        print("   Moving above object...")
        self.move_to_position(wx, wy, approach_height)
        time.sleep(0.3)
        
        # 3. Lower to object
        print("   Lowering to grasp...")
        self.move_to_position(wx, wy, grasp_height)
        time.sleep(0.3)
        
        # 4. Close gripper
        print("   Grasping...")
        self.set_gripper(0)
        time.sleep(0.3)
        
        # 5. Lift object
        print("   Lifting...")
        self.move_to_position(wx, wy, approach_height)
        time.sleep(0.3)
        
        print(f"🎉 Picked up {target['name']}!")
        return True
    
    def set_gripper(self, position: int):
        """Set gripper position (0=closed, 100=open)."""
        self.current_gripper = max(0, min(100, position))
        print(f"🤏 Gripper: {self.current_gripper}%")
        
        # Control simulation gripper
        if self.sim:
            self.sim.set_gripper(self.current_gripper / 100.0)
        
        # Send update to network
        self.network.send_to_network(self.current_angles, 
                                      gripper=self.current_gripper)
    
    def process_voice_command(self) -> bool:
        """
        Listen for and execute a voice command.
        
        Returns:
            bool: True if command executed successfully
        """
        if not self.voice:
            print("❌ Voice control not initialized")
            return False
        
        # Listen for command
        intent = self.voice.listen_command()
        
        if intent is None:
            return False
        
        # Execute command
        result = self.voice.execute_voice_command(intent)
        print(f"🎤 {result['message']}")
        
        if not result['success']:
            return False
        
        # Handle different actions
        if intent.action == CommandAction.STOP:
            self.network.send_emergency_stop()
            
        elif intent.action == CommandAction.HOME:
            self.go_home()
            
        elif intent.action == CommandAction.PICK:
            self.detect_and_pick(
                target_color=intent.target_color,
                target_shape=result['params'].get('object_type')
            )
            
        elif intent.action == CommandAction.OPEN:
            self.set_gripper(100)
            
        elif intent.action == CommandAction.CLOSE:
            self.set_gripper(0)
        
        return True
    
    def run_demo(self):
        """Run a demonstration sequence."""
        print("\n" + "=" * 60)
        print("🎬 Running Demo Sequence")
        print("=" * 60)
        
        # Demo positions
        positions = [
            (0.15, 0.00, 0.20, "Center high"),
            (0.18, 0.10, 0.10, "Front-left low"),
            (0.18, -0.10, 0.10, "Front-right low"),
            (0.20, 0.00, 0.05, "Far reach low"),
            (0.12, 0.00, 0.25, "Near high"),
        ]
        
        for x, y, z, desc in positions:
            print(f"\n➡️ {desc}")
            self.move_to_position(x, y, z)
            time.sleep(1.5)
        
        # Return home
        print("\n🏠 Demo complete, returning home")
        self.go_home()
    
    def run_interactive(self):
        """Run interactive mode with PyBullet sliders."""
        if not self.sim:
            print("❌ Simulation not initialized")
            return
        
        print("\n🎮 Starting interactive mode...")
        print("   Use sliders to control the robot")
        print("   Press Ctrl+C to exit")
        
        self.sim.run_interactive()
    
    def close(self):
        """Clean up and close all resources."""
        print("\n👋 Shutting down...")
        
        if self.sim:
            self.sim.close()
        
        if self.camera:
            self.camera.close()
        
        self.network.disconnect()
        
        print("✅ Shutdown complete")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="HowToMechatronics Robot Brain Controller"
    )
    
    parser.add_argument('--gui', action='store_true',
                        help='Run interactive GUI mode')
    parser.add_argument('--voice', action='store_true',
                        help='Enable voice control')
    parser.add_argument('--camera', action='store_true',
                        help='Enable camera input')
    parser.add_argument('--no-sim', action='store_true',
                        help='Disable PyBullet simulation')
    parser.add_argument('--network', choices=['simulated', 'zmq', 'serial'],
                        default='simulated',
                        help='Network mode (default: simulated)')
    
    args = parser.parse_args()
    
    # Create controller
    brain = BrainController(
        use_simulation=not args.no_sim,
        use_camera=args.camera,
        use_voice=args.voice,
        network_mode=args.network
    )
    
    try:
        if args.gui:
            # Interactive mode
            brain.run_interactive()
        elif args.voice:
            # Voice control loop with wake word
            print("\n🎤 Mode commande vocale")
            print("   Dis 'Bonjour bras' pour m'activer")
            print("   Je t'écoute jusqu'à 2s de silence")
            print("   Dis 'stop' ou Ctrl+C pour quitter\n")
            
            def on_voice_command(intent):
                """Callback for voice commands."""
                print(f"🎯 Action: {intent.action.value}")
                
                if intent.action == CommandAction.PICK:
                    brain.detect_and_pick(
                        target_color=intent.target_color,
                        target_shape=intent.target_object
                    )
                elif intent.action == CommandAction.PLACE:
                    # Place object at position
                    if intent.target_position == "left":
                        brain.move_to_position(0.20, 0.15, 0.10)
                    elif intent.target_position == "right":
                        brain.move_to_position(0.20, -0.15, 0.10)
                    elif intent.target_position == "center":
                        brain.move_to_position(0.25, 0, 0.10)
                    brain.set_gripper(100)  # Release
                elif intent.action == CommandAction.MOVE:
                    # Move to position
                    if intent.target_position == "left":
                        brain.move_to_position(0.20, 0.15, 0.15)
                    elif intent.target_position == "right":
                        brain.move_to_position(0.20, -0.15, 0.15)
                    elif intent.target_position == "center":
                        brain.move_to_position(0.25, 0, 0.15)
                    elif intent.target_position == "front":
                        brain.move_to_position(0.30, 0, 0.10)
                    else:
                        print("   Position: left, right, center, front")
                elif intent.action == CommandAction.HOME:
                    brain.go_home()
                elif intent.action == CommandAction.OPEN:
                    brain.set_gripper(100)
                elif intent.action == CommandAction.CLOSE:
                    brain.set_gripper(0)
                elif intent.action == CommandAction.STOP:
                    brain.network.send_emergency_stop()
            
            brain.voice.run_continuous(on_voice_command)
        else:
            # Demo mode
            brain.run_demo()
    
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted by user")
    
    finally:
        brain.close()


if __name__ == "__main__":
    main()
