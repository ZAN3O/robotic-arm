"""
simulation.py - PyBullet Digital Twin for HowToMechatronics 4-DOF Arm

This module provides a visual simulation environment to validate
robot movements before sending commands to the physical robot.
"""

import pybullet as p
import pybullet_data
import numpy as np
import time
import os
from typing import Optional, List, Tuple

# Import our kinematics module
from kinematics import get_servo_angles, create_kinematic_chain, validate_angles


class RobotSim:
    """
    PyBullet simulation environment for the robotic arm.
    
    Provides:
    - Visual feedback of robot movements
    - Collision detection
    - Movement validation before sending to real robot
    """
    
    def __init__(self, urdf_path: str = "arduino_arm.urdf", headless: bool = False):
        """
        Initialize the PyBullet simulation.
        
        Args:
            urdf_path: Path to the robot URDF file
            headless: If True, run without GUI (for testing)
        """
        self.urdf_path = urdf_path
        self.headless = headless
        
        # PyBullet IDs
        self.physics_client = None
        self.robot_id = None
        self.table_id = None
        self.plane_id = None
        
        # Joint configuration
        self.num_joints = 0
        self.active_joints = []  # Indices of movable joints
        
        # Kinematic chain for IK
        self.chain = create_kinematic_chain()
        
        # Virtual camera configuration
        self.camera_width = 640
        self.camera_height = 480
        self.camera_fov = 60  # Field of view in degrees
        self.camera_near = 0.01
        self.camera_far = 2.0
        
        # Camera position (top-down view of workspace)
        self.camera_position = [0.25, 0, 0.6]  # Above the desk
        self.camera_target = [0.25, 0, 0]  # Looking at desk center
        self.camera_up = [0, 1, 0]  # Y-axis up in camera frame
        
        # Initialize simulation
        self._setup_simulation()
    
    def _setup_simulation(self):
        """Set up the PyBullet environment with table and robot."""
        # Connect to PyBullet
        if self.headless:
            self.physics_client = p.connect(p.DIRECT)
        else:
            self.physics_client = p.connect(p.GUI)
            # Configure camera view
            p.resetDebugVisualizerCamera(
                cameraDistance=0.6,
                cameraYaw=45,
                cameraPitch=-30,
                cameraTargetPosition=[0.1, 0, 0.15]
            )
        
        # Set physics parameters for realistic simulation
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)  # Manual stepping for control
        
        # Configure physics engine for better collision detection
        p.setPhysicsEngineParameter(
            fixedTimeStep=1./240.,
            numSolverIterations=50,
            numSubSteps=4,
            contactBreakingThreshold=0.02,
            enableConeFriction=True
        )
        
        # Load ground plane with friction
        self.plane_id = p.loadURDF("plane.urdf")
        p.changeDynamics(self.plane_id, -1, lateralFriction=1.0, restitution=0.1)
        
        # Load table (using a scaled box)
        self._create_table()
        
        # Load robot
        self._load_robot()
        
        # Configure synthetic camera for GUI debug windows
        if not self.headless:
            self._setup_debug_camera()
        
        print("✅ PyBullet simulation initialized")
        print(f"   Robot joints: {self.num_joints} ({len(self.active_joints)} active)")
    
    def _setup_debug_camera(self):
        """Configure the debug camera windows in PyBullet GUI."""
        # Enable synthetic camera debug visualizer
        p.configureDebugVisualizer(p.COV_ENABLE_RGB_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_DEPTH_BUFFER_PREVIEW, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 1)
        
        # Disable unnecessary visual clutter
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
    
    def _create_table(self):
        """Create a realistic desk environment with the robot on the side."""
        # =============================================
        # DESK - Large wooden desk
        # =============================================
        desk_width = 0.80   # 80cm wide
        desk_depth = 0.50   # 50cm deep  
        desk_height = 0.02  # 2cm thick top
        desk_leg_height = 0.70  # 70cm tall (but we see from table surface)
        
        # Desk top (wood color)
        desk_visual = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[desk_width/2, desk_depth/2, desk_height/2],
            rgbaColor=[0.55, 0.35, 0.2, 1]  # Dark wood
        )
        desk_collision = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[desk_width/2, desk_depth/2, desk_height/2]
        )
        
        # Position desk - robot at edge (left side)
        desk_center_x = 0.25  # Desk center is in front of robot
        desk_center_y = 0.0
        
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=desk_collision,
            baseVisualShapeIndex=desk_visual,
            basePosition=[desk_center_x, desk_center_y, -desk_height/2]
        )
        
        # =============================================
        # OBJECTS on the desk
        # =============================================
        self.scene_objects = []
        
        # Red cube (easy to grab)
        self._add_scene_object("cube", [0.20, 0.10, 0.015], 0.03, [0.9, 0.1, 0.1, 1], "red_cube")
        
        # Blue sphere (ball)
        self._add_scene_object("sphere", [0.25, -0.08, 0.015], 0.025, [0.1, 0.3, 0.9, 1], "blue_ball")
        
        # Green cylinder (can)
        self._add_scene_object("cylinder", [0.35, 0.05, 0.02], 0.02, [0.2, 0.8, 0.2, 1], "green_can")
        
        # Yellow cube
        self._add_scene_object("cube", [0.30, -0.12, 0.0125], 0.025, [0.95, 0.85, 0.1, 1], "yellow_cube")
        
        # Orange sphere
        self._add_scene_object("sphere", [0.15, -0.05, 0.02], 0.04, [1.0, 0.5, 0.0, 1], "orange_ball")
        
        # White box (eraser-like)
        self._add_scene_object("cube", [0.40, 0.00, 0.01], 0.02, [0.95, 0.95, 0.95, 1], "white_box")
        
        # =============================================
        # DECORATIVE ELEMENTS (static)
        # =============================================
        
        # Pen holder (dark cylinder at back)
        pen_holder_vis = p.createVisualShape(
            p.GEOM_CYLINDER,
            radius=0.025,
            length=0.08,
            rgbaColor=[0.2, 0.2, 0.25, 1]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=pen_holder_vis,
            basePosition=[0.45, 0.15, 0.04]
        )
        
        # Small notepad (flat box)
        notepad_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.04, 0.06, 0.005],
            rgbaColor=[0.95, 0.95, 0.8, 1]  # Cream paper
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=notepad_vis,
            basePosition=[0.42, -0.10, 0.005]
        )
        
        # Keyboard outline (flat gray box at back)
        keyboard_vis = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[0.15, 0.05, 0.008],
            rgbaColor=[0.3, 0.3, 0.35, 1]
        )
        p.createMultiBody(
            baseMass=0,
            baseVisualShapeIndex=keyboard_vis,
            basePosition=[0.50, 0.0, 0.008]
        )
        
        # Add workspace boundary
        self._add_workspace_markers()
        
        print(f"   Desk scene created with {len(self.scene_objects)} grabbable objects")
    
    def _add_scene_object(self, shape: str, position: List[float], 
                          size: float, color: List[float], name: str) -> int:
        """Add a named object to the scene."""
        obj_id = self.add_object(shape, position, size, color)
        self.scene_objects.append({
            'id': obj_id,
            'name': name,
            'shape': shape,
            'position': position,
            'size': size,
            'color': color
        })
        return obj_id
    
    def _add_workspace_markers(self):
        """Add visual markers showing robot workspace."""
        # Draw workspace arc (robot can only reach front area)
        radius = 0.32  # ~32cm reach
        segments = 24
        
        # Draw semi-circle in front of robot
        for i in range(segments):
            angle1 = -np.pi/2 + np.pi * i / segments
            angle2 = -np.pi/2 + np.pi * (i + 1) / segments
            
            p1 = [radius * np.cos(angle1), radius * np.sin(angle1), 0.001]
            p2 = [radius * np.cos(angle2), radius * np.sin(angle2), 0.001]
            
            p.addUserDebugLine(p1, p2, [0, 0.6, 0.3], lineWidth=2, lifeTime=0)
        
        # Draw robot base marker
        p.addUserDebugText("🤖", [0, 0, 0.35], [1, 1, 1], textSize=1.5)
    
    def _load_robot(self):
        """Load the robot URDF into the simulation."""
        # Check if URDF exists
        if not os.path.exists(self.urdf_path):
            raise FileNotFoundError(f"URDF not found: {self.urdf_path}")
        
        # Load robot
        start_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.robot_id = p.loadURDF(
            self.urdf_path,
            basePosition=[0, 0, 0],
            baseOrientation=start_orientation,
            useFixedBase=True
        )
        
        # Get joint info - separate arm joints from gripper joints
        self.num_joints = p.getNumJoints(self.robot_id)
        self.active_joints = []  # Arm joints only
        self.gripper_joints = []  # Gripper finger joints
        
        for i in range(self.num_joints):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            # Only track revolute joints (type 0)
            if joint_type == p.JOINT_REVOLUTE:
                joint_data = {
                    'index': i,
                    'name': joint_name,
                    'lower_limit': joint_info[8],
                    'upper_limit': joint_info[9]
                }
                
                # Separate gripper joints from arm joints
                if 'gripper' in joint_name.lower():
                    self.gripper_joints.append(joint_data)
                else:
                    self.active_joints.append(joint_data)
        
        # Set physics properties for all robot links (for collision/grasping)
        for i in range(-1, self.num_joints):
            p.changeDynamics(
                self.robot_id, i,
                lateralFriction=1.0,
                spinningFriction=0.5,
                rollingFriction=0.1,
                contactDamping=100,
                contactStiffness=10000
            )
        
        # Enable self-collision detection
        p.setCollisionFilterGroupMask(self.robot_id, -1, 1, 1)
        
        # Gripper state (0 = closed, 1 = open)
        self.gripper_state = 0.0
        
        print(f"   Gripper joints: {len(self.gripper_joints)}")
        print(f"   Collision enabled for {self.num_joints + 1} links")
    
    def set_gripper(self, openness: float, animate: bool = True):
        """
        Control gripper opening.
        
        Args:
            openness: 0.0 = fully closed, 1.0 = fully open
            animate: Whether to animate the movement
        """
        openness = max(0.0, min(1.0, openness))
        self.gripper_state = openness
        
        for joint in self.gripper_joints:
            # Calculate target position based on joint limits
            if 'left' in joint['name']:
                # Left finger opens in positive direction
                target = joint['lower_limit'] + openness * (joint['upper_limit'] - joint['lower_limit'])
            else:
                # Right finger opens in negative direction (mirrors left)
                target = joint['upper_limit'] + openness * (joint['lower_limit'] - joint['upper_limit'])
            
            if animate:
                p.setJointMotorControl2(
                    self.robot_id,
                    joint['index'],
                    p.POSITION_CONTROL,
                    targetPosition=target,
                    force=10
                )
            else:
                p.resetJointState(self.robot_id, joint['index'], target)
        
        if animate:
            # Step simulation to see the movement
            for _ in range(50):
                p.stepSimulation()
                if not self.headless:
                    time.sleep(1./240.)
        
        state = "ouvert" if openness > 0.5 else "fermé"
        print(f"🤏 Pince: {openness*100:.0f}% ({state})")
    
    def open_gripper(self):
        """Open the gripper fully."""
        self.set_gripper(1.0)
    
    def close_gripper(self):
        """Close the gripper fully."""
        self.set_gripper(0.0)
    
    def move_to_angles(self, angles_deg: List[float], 
                       speed: float = 1.0,
                       animate: bool = True):
        """
        Move robot joints to specified angles.
        
        Args:
            angles_deg: List of 4 angles in degrees [base, shoulder, elbow, wrist]
            speed: Movement speed multiplier (1.0 = normal)
            animate: If True, animate the movement; if False, snap instantly
        """
        if len(angles_deg) != len(self.active_joints):
            raise ValueError(f"Expected {len(self.active_joints)} angles, got {len(angles_deg)}")
        
        # Convert to radians
        angles_rad = [np.radians(a) for a in angles_deg]
        
        if animate:
            # Smooth animation
            steps = int(50 / speed)
            
            # Get current positions
            current = []
            for joint in self.active_joints:
                state = p.getJointState(self.robot_id, joint['index'])
                current.append(state[0])
            
            # Interpolate
            for step in range(steps + 1):
                t = step / steps
                t = t * t * (3 - 2 * t)  # Smooth interpolation
                
                for i, joint in enumerate(self.active_joints):
                    target = current[i] + t * (angles_rad[i] - current[i])
                    p.setJointMotorControl2(
                        self.robot_id,
                        joint['index'],
                        p.POSITION_CONTROL,
                        targetPosition=target,
                        force=100
                    )
                
                p.stepSimulation()
                if not self.headless:
                    time.sleep(1./240.)
        else:
            # Instant positioning
            for i, joint in enumerate(self.active_joints):
                p.resetJointState(self.robot_id, joint['index'], angles_rad[i])
    
    def move_to_position(self, x: float, y: float, z: float,
                        animate: bool = True) -> Tuple[List[float], bool]:
        """
        Move robot end-effector to target position using IK.
        
        Args:
            x, y, z: Target position in meters
            animate: Whether to animate the movement
            
        Returns:
            tuple: (angles_degrees, success)
        """
        # Calculate IK
        angles, success = get_servo_angles(x, y, z, self.chain)
        
        if success:
            # Validate angles
            validated, valid = validate_angles(angles)
            
            # Move to position
            self.move_to_angles(validated, animate=animate)
            
            # Draw target marker
            if not self.headless:
                self._draw_target_marker(x, y, z, success)
        else:
            print(f"❌ Cannot reach position ({x:.2f}, {y:.2f}, {z:.2f})")
        
        return angles, success
    
    def _draw_target_marker(self, x: float, y: float, z: float, success: bool):
        """Draw a visual marker at target position."""
        color = [0, 1, 0] if success else [1, 0, 0]  # Green or red
        
        # Draw cross marker
        size = 0.02
        p.addUserDebugLine([x-size, y, z], [x+size, y, z], color, lineWidth=2, lifeTime=3)
        p.addUserDebugLine([x, y-size, z], [x, y+size, z], color, lineWidth=2, lifeTime=3)
        p.addUserDebugLine([x, y, z-size], [x, y, z+size], color, lineWidth=2, lifeTime=3)
    
    def validate_movement(self, angles_deg: List[float]) -> Tuple[bool, str]:
        """
        Validate if a movement is safe before sending to real robot.
        
        Args:
            angles_deg: Target angles in degrees
            
        Returns:
            tuple: (is_valid, message)
        """
        issues = []
        
        # Check angle limits
        for i, (angle, joint) in enumerate(zip(angles_deg, self.active_joints)):
            lower = np.degrees(joint['lower_limit'])
            upper = np.degrees(joint['upper_limit'])
            
            if angle < lower:
                issues.append(f"{joint['name']}: {angle:.1f}° < min {lower:.1f}°")
            elif angle > upper:
                issues.append(f"{joint['name']}: {angle:.1f}° > max {upper:.1f}°")
        
        if issues:
            return False, "Angle limits exceeded: " + "; ".join(issues)
        
        return True, "Movement validated ✅"
    
    def get_end_effector_position(self) -> List[float]:
        """Get current end-effector position."""
        # Get state of last link
        state = p.getLinkState(self.robot_id, self.num_joints - 1)
        return list(state[0])
    
    # =========================================================================
    # VIRTUAL CAMERA
    # =========================================================================
    
    def get_camera_image(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Capture image from virtual camera.
        
        Returns:
            rgb: RGB image (H, W, 3) uint8
            depth: Depth image (H, W) float32 in meters
            segmentation: Segmentation mask (H, W) int32 with object IDs
        """
        # Compute view matrix
        view_matrix = p.computeViewMatrix(
            cameraEyePosition=self.camera_position,
            cameraTargetPosition=self.camera_target,
            cameraUpVector=self.camera_up
        )
        
        # Compute projection matrix
        aspect = self.camera_width / self.camera_height
        projection_matrix = p.computeProjectionMatrixFOV(
            fov=self.camera_fov,
            aspect=aspect,
            nearVal=self.camera_near,
            farVal=self.camera_far
        )
        
        # Capture image
        _, _, rgba, depth_buffer, segmentation = p.getCameraImage(
            width=self.camera_width,
            height=self.camera_height,
            viewMatrix=view_matrix,
            projectionMatrix=projection_matrix,
            renderer=p.ER_TINY_RENDERER  # Works in headless mode
        )
        
        # Process RGB - reshape flat array to image
        rgba = np.array(rgba, dtype=np.uint8).reshape(self.camera_height, self.camera_width, 4)
        rgb = rgba[:, :, :3]
        
        # Convert depth buffer to actual depth in meters
        depth_buffer = np.array(depth_buffer).reshape(self.camera_height, self.camera_width)
        depth = self.camera_far * self.camera_near / (
            self.camera_far - (self.camera_far - self.camera_near) * depth_buffer
        )
        depth = depth.astype(np.float32)
        
        # Segmentation mask - reshape
        seg = np.array(segmentation, dtype=np.int32).reshape(self.camera_height, self.camera_width)
        
        return rgb, depth, seg
    
    def set_camera_position(self, position: List[float], target: List[float] = None):
        """
        Set camera position and target.
        
        Args:
            position: [x, y, z] camera position
            target: [x, y, z] look-at point (default: desk center)
        """
        self.camera_position = position
        if target is not None:
            self.camera_target = target
    
    def set_camera_topdown(self, height: float = 0.5):
        """Configure camera for top-down view (useful for AI)."""
        self.camera_position = [0.25, 0, height]
        self.camera_target = [0.25, 0, 0]
        self.camera_up = [0, 1, 0]
    
    def set_camera_side(self, distance: float = 0.6):
        """Configure camera for side view."""
        self.camera_position = [0.25, -distance, 0.3]
        self.camera_target = [0.25, 0, 0.15]
        self.camera_up = [0, 0, 1]
    
    def set_camera_perspective(self):
        """Configure camera for 3/4 perspective view (like default GUI)."""
        self.camera_position = [0.5, -0.3, 0.4]
        self.camera_target = [0.2, 0, 0.1]
        self.camera_up = [0, 0, 1]
    
    def save_camera_image(self, filepath: str = "camera_capture.png"):
        """
        Save current camera view to file.
        
        Args:
            filepath: Output file path (.png)
        """
        import cv2
        rgb, _, _ = self.get_camera_image()
        # Convert RGB to BGR for OpenCV
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(filepath, bgr)
        print(f"📷 Image saved: {filepath}")
    
    def get_contacts(self) -> List[dict]:
        """
        Get all contact points involving the robot.
        Useful for AI training to detect collisions.
        
        Returns:
            List of contact dictionaries with position and force info
        """
        contacts = []
        contact_points = p.getContactPoints(bodyA=self.robot_id)
        
        for contact in contact_points:
            contacts.append({
                'body_b': contact[2],  # Other body ID
                'link_a': contact[3],  # Robot link index
                'link_b': contact[4],  # Other body link
                'position': contact[5],  # Contact position on A
                'normal': contact[7],  # Contact normal
                'distance': contact[8],  # Penetration depth (negative = collision)
                'force': contact[9]  # Normal force
            })
        
        return contacts
    
    def is_gripper_holding(self, object_id: int = None) -> bool:
        """
        Check if gripper is in contact with an object.
        
        Args:
            object_id: Specific object to check, or None for any object
            
        Returns:
            True if gripper fingers are touching object
        """
        left_contacts = False
        right_contacts = False
        
        for joint in self.gripper_joints:
            link_idx = joint['index']
            
            if object_id is not None:
                contact_points = p.getContactPoints(
                    bodyA=self.robot_id, 
                    linkIndexA=link_idx,
                    bodyB=object_id
                )
            else:
                contact_points = p.getContactPoints(
                    bodyA=self.robot_id, 
                    linkIndexA=link_idx
                )
            
            if contact_points:
                if 'left' in joint['name']:
                    left_contacts = True
                else:
                    right_contacts = True
        
        # Object is held if BOTH fingers are touching it
        return left_contacts and right_contacts
    
    def step_simulation(self, steps: int = 1):
        """
        Step the physics simulation forward.
        Call this in your training loop.
        
        Args:
            steps: Number of simulation steps to execute
        """
        for _ in range(steps):
            p.stepSimulation()
    
    def add_object(self, shape: str = "cube", 
                   position: List[float] = [0.15, 0, 0.02],
                   size: float = 0.03,
                   color: List[float] = [1, 0, 0, 1],
                   mass: float = 0.05) -> int:
        """
        Add a physical object to the scene for training.
        
        Args:
            shape: "cube", "sphere", or "cylinder"
            position: [x, y, z] position (z is height above table)
            size: Object size in meters
            color: RGBA color
            mass: Object mass in kg
            
        Returns:
            int: PyBullet object ID
        """
        # Adjust z position to place object ON the table surface
        pos = list(position)
        if shape == "cube":
            pos[2] = size / 2  # Half height above table
            col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size/2]*3)
            vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[size/2]*3, rgbaColor=color)
        elif shape == "sphere":
            pos[2] = size / 2  # Radius above table
            col_shape = p.createCollisionShape(p.GEOM_SPHERE, radius=size/2)
            vis_shape = p.createVisualShape(p.GEOM_SPHERE, radius=size/2, rgbaColor=color)
        elif shape == "cylinder":
            pos[2] = size / 2  # Half height above table
            col_shape = p.createCollisionShape(p.GEOM_CYLINDER, radius=size/2, height=size)
            vis_shape = p.createVisualShape(p.GEOM_CYLINDER, radius=size/2, length=size, rgbaColor=color)
        else:
            raise ValueError(f"Unknown shape: {shape}")
        
        obj_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=pos
        )
        
        # Set physical properties for realistic grasping
        p.changeDynamics(
            obj_id, -1,
            lateralFriction=0.8,
            spinningFriction=0.3,
            rollingFriction=0.1,
            restitution=0.2,
            contactDamping=100,
            contactStiffness=10000
        )
        
        return obj_id
    
    def run_interactive(self):
        """Run interactive mode with joint sliders including gripper."""
        if self.headless:
            print("Cannot run interactive mode in headless mode")
            return
        
        # Create sliders for arm joints
        sliders = []
        for joint in self.active_joints:
            slider = p.addUserDebugParameter(
                joint['name'],
                np.degrees(joint['lower_limit']),
                np.degrees(joint['upper_limit']),
                0
            )
            sliders.append(slider)
        
        # Create gripper slider (0-100%)
        gripper_slider = p.addUserDebugParameter(
            "Gripper (0=fermé, 100=ouvert)",
            0, 100, 0
        )
        
        print("🎮 Interactive mode - Use sliders to control joints")
        print("   Gripper slider: 0=closed, 100=open")
        print("   Press Ctrl+C to exit")
        
        last_gripper_value = 0
        
        try:
            while True:
                # Update arm joints
                for i, (slider, joint) in enumerate(zip(sliders, self.active_joints)):
                    target_deg = p.readUserDebugParameter(slider)
                    target_rad = np.radians(target_deg)
                    p.setJointMotorControl2(
                        self.robot_id,
                        joint['index'],
                        p.POSITION_CONTROL,
                        targetPosition=target_rad,
                        force=100
                    )
                
                # Update gripper
                gripper_value = p.readUserDebugParameter(gripper_slider)
                if abs(gripper_value - last_gripper_value) > 1:
                    self.set_gripper(gripper_value / 100.0, animate=False)
                    last_gripper_value = gripper_value
                
                # Keep gripper position
                for joint in self.gripper_joints:
                    if 'left' in joint['name']:
                        target = joint['lower_limit'] + (gripper_value/100) * (joint['upper_limit'] - joint['lower_limit'])
                    else:
                        target = joint['upper_limit'] + (gripper_value/100) * (joint['lower_limit'] - joint['upper_limit'])
                    
                    p.setJointMotorControl2(
                        self.robot_id,
                        joint['index'],
                        p.POSITION_CONTROL,
                        targetPosition=target,
                        force=10
                    )
                
                p.stepSimulation()
                time.sleep(1./240.)
        except KeyboardInterrupt:
            print("\n👋 Interactive mode ended")
    
    def close(self):
        """Clean up and close the simulation."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
            print("🔌 Simulation closed")


# ============================================================================
# TEST / DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🤖 HowToMechatronics Arm - PyBullet Desk Scene")
    print("=" * 60)
    
    # Create simulation - desk scene is auto-created with objects
    sim = RobotSim()
    
    print(f"\n📦 Objects on desk:")
    for obj in sim.scene_objects:
        print(f"   - {obj['name']} at ({obj['position'][0]:.2f}, {obj['position'][1]:.2f})")
    
    print("\n🎯 Testing movements...")
    
    # Test positions
    test_targets = [
        (0.15, 0.05, 0.10),   # Above red cube
        (0.12, -0.08, 0.08),  # Above blue sphere
        (0.20, 0.00, 0.15),   # Center high
    ]
    
    for i, (x, y, z) in enumerate(test_targets):
        print(f"\n📍 Moving to target {i+1}: ({x:.2f}, {y:.2f}, {z:.2f})m")
        angles, success = sim.move_to_position(x, y, z)
        
        if success:
            print(f"   Angles: {[f'{a:.1f}°' for a in angles]}")
            valid, msg = sim.validate_movement(angles)
            print(f"   {msg}")
        
        time.sleep(1)
    
    # Return to home
    print("\n🏠 Returning to home position...")
    sim.move_to_angles([0, 0, 0, 0])
    
    # Run interactive mode
    print("\n" + "=" * 60)
    sim.run_interactive()
    
    sim.close()
