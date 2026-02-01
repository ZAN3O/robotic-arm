"""
IA.py - Reinforcement Learning AI for Robotic Arm Grasping

This module implements a complete AI training system using:
- Gymnasium environment wrapper for PyBullet simulation
- PPO (Proximal Policy Optimization) algorithm from Stable Baselines3
- Custom reward function for object grasping

The AI learns to:
1. Move the arm to objects
2. Open/close gripper at the right time
3. Lift objects successfully

Usage:
    python IA.py --train           # Train the AI
    python IA.py --train --steps 500000  # Train for more steps
    python IA.py --test            # Test trained model
    python IA.py --demo            # Visual demo with trained model
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time
import os
import argparse
from typing import Tuple, Optional, Dict, Any
from pathlib import Path


class RobotArmEnv(gym.Env):
    """
    Gymnasium environment for robotic arm grasping.
    
    Observation space:
        - Joint angles (4 values)
        - Gripper state (1 value)
        - End effector position (3 values)
        - Target object position (3 values)
        - Distance to target (1 value)
        Total: 12 values
    
    Action space:
        - Delta joint angles (4 values) [-1, 1] scaled to degrees
        - Gripper command (1 value) [-1=close, 1=open]
        Total: 5 continuous values
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode: str = None, max_steps: int = 500):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps
        self.current_step = 0
        
        # PyBullet setup
        self.physics_client = None
        self.robot_id = None
        self.target_object_id = None
        self.plane_id = None
        self.table_id = None
        
        # Robot configuration
        self.num_arm_joints = 4
        self.joint_indices = []
        self.gripper_indices = []
        
        # Joint limits (degrees)
        self.joint_limits_low = np.array([-150, -90, -90, -90])
        self.joint_limits_high = np.array([150, 90, 90, 90])
        
        # Action scaling - REDUCED for smoother movements
        self.action_scale = 4.0  # Slightly increased (was 3.0, orig 5.0)
        self.action_scale_near = 1.5  # Faster approach (was 1.0)
        
        # Observation space: 14 dimensions
        # [joint_angles(4), gripper(1), ee_pos(3), direction(3), height_diff(1), distance(1), close_flag(1)]
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(14,),
            dtype=np.float32
        )
        
        # Action space: 5 dimensions (4 joints + gripper)
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(5,),
            dtype=np.float32
        )
        
        # State tracking
        self.current_joint_angles = np.zeros(4)
        self.previous_joint_angles = np.zeros(4)  # Track previous for jerk detection
        self.previous_action = np.zeros(5)  # Track previous action for smoothness
        self.gripper_state = 1.0  # 0=closed, 1=open (start OPEN)
        self.target_position = np.zeros(3)
        self.object_grasped = False
        self.object_lifted = False
        self.current_ee_velocity = 0.0  # Track end effector speed
        
        # Success tracking
        self.success_count = 0
        self.episode_count = 0
        self.recent_successes = []  # Track last 20 episodes for curriculum
        
        # Curriculum Learning
        self.difficulty_level = 1  # Start at Level 1 (Easy)
        print(f"🎓 Environnement initialisé au Niveau {self.difficulty_level}")
        
        # Initialize simulation
        self._setup_simulation()
    
    def _setup_simulation(self):
        """Initialize PyBullet simulation."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        # Connect to PyBullet
        if self.render_mode == "human":
            self.physics_client = p.connect(p.GUI)
            p.resetDebugVisualizerCamera(
                cameraDistance=0.6, cameraYaw=45, cameraPitch=-30,
                cameraTargetPosition=[0.2, 0, 0.1]
            )
        else:
            self.physics_client = p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setPhysicsEngineParameter(
            fixedTimeStep=1./240.,
            numSolverIterations=50,
            numSubSteps=4
        )
        
        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")
        self._create_table()
        self._load_robot()
    
    def _create_table(self):
        """Create simple table."""
        table_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[0.4, 0.25, 0.005]
        )
        table_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[0.4, 0.25, 0.005],
            rgbaColor=[0.6, 0.4, 0.2, 1]
        )
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_vis,
            basePosition=[0.17, 0, 0]  # Closer to robot base
        )
    
    def _load_robot(self):
        """Load robot URDF."""
        urdf_path = os.path.join(os.path.dirname(__file__), "arduino_arm.urdf")
        
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True
        )
        
        # Find joint indices
        self.joint_indices = []
        self.gripper_indices = []
        
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            if joint_type == p.JOINT_REVOLUTE:
                if 'gripper' in joint_name.lower():
                    self.gripper_indices.append(i)
                elif len(self.joint_indices) < 4:
                    self.joint_indices.append(i)
        
        # Set dynamics for grasping
        for i in range(-1, p.getNumJoints(self.robot_id)):
            p.changeDynamics(
                self.robot_id, i,
                lateralFriction=1.0,
                contactDamping=100,
                contactStiffness=10000
            )

        # Custom joint limits to prevent "Head in butt" (folding backwards)
        # Standard limits are [-1.57, 1.57] which allows weird poses.
        # We restrict them to keep the arm forward-facing.
        self.joint_limits_low = np.array([-2.6, -1.8, -1.8, -1.8])
        self.joint_limits_high = np.array([2.6, 0.5, 1.8, 1.8])
    
    def _create_target_object(self):
        """Create a random target object."""
        if self.target_object_id is not None:
            p.removeBody(self.target_object_id)
        
        # Random position on table WITHIN ARM'S REACH
        # Arm reach is ~0.25-0.28m max.
        # Adjusted to 0.15-0.26m to avoid spawning INSIDE the robot base/gripper
        x = np.random.uniform(0.15, 0.26)
        y = np.random.uniform(-0.12, 0.12)
        z = 0.02
        
        self.target_position = np.array([x, y, z])
        self.initial_target_position = self.target_position.copy()  # Store for displacement penalty
        
        # Random color
        colors = [
            [0.9, 0.1, 0.1, 1],  # Red
            [0.1, 0.3, 0.9, 1],  # Blue
            [0.2, 0.8, 0.2, 1],  # Green
            [0.9, 0.8, 0.1, 1],  # Yellow
        ]
        color = colors[np.random.randint(len(colors))]
        
        # Create cube
        size = 0.025
        col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[size/2]*3)
        vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[size/2]*3, rgbaColor=color)
        
        self.target_object_id = p.createMultiBody(
            baseMass=0.05,
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=self.target_position.tolist()
        )
        
        p.changeDynamics(
            self.target_object_id, -1,
            lateralFriction=0.8,
            restitution=0.1
        )
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation."""
        # Joint angles (normalized to [-1, 1])
        joint_angles_norm = (self.current_joint_angles - self.joint_limits_low) / \
                           (self.joint_limits_high - self.joint_limits_low) * 2 - 1
        
        # Gripper state
        gripper = np.array([self.gripper_state * 2 - 1])  # [-1, 1]
        
        # End effector position
        ee_pos = self._get_end_effector_position()
        
        # Target position
        if self.target_object_id is not None:
            target_pos, _ = p.getBasePositionAndOrientation(self.target_object_id)
            target_pos = np.array(target_pos)
        else:
            target_pos = self.target_position
        
        # Direction vector to target (normalized) - CRITICAL for goal-directed behavior
        direction = target_pos - ee_pos
        distance = np.linalg.norm(direction)
        if distance > 0.001:
            direction_norm = direction / distance
        else:
            direction_norm = np.zeros(3)
        
        # Height difference
        height_diff = np.array([target_pos[2] - ee_pos[2]])
        
        obs = np.concatenate([
            joint_angles_norm,      # 4 - current arm state
            gripper,                # 1 - gripper state
            ee_pos,                 # 3 - where am I
            direction_norm,         # 3 - which way to go (normalized)
            height_diff,            # 1 - height difference
            np.array([distance]),   # 1 - how far
            np.array([distance < 0.05]),  # 1 - am I close?
        ]).astype(np.float32)
        
        return obs
        
    def _get_end_effector_position(self) -> np.ndarray:
        """Get end effector position (FINGERTIPS, not wrist)."""
        # Get the gripper link state
        if len(self.gripper_indices) > 0:
            link_idx = self.gripper_indices[0]
            state = p.getLinkState(self.robot_id, link_idx)
            pos = np.array(state[0])
            orn = np.array(state[1])
            
            # The gripper link is the hinge. The fingers extend ~5-6cm from it.
            # We must add an offset to get the true fingertip centroid.
            # Assuming Z-axis of gripper link points towards fingers.
            offset_local = [0, 0, 0.05]  # 5cm offset
            pos_offset, _ = p.multiplyTransforms(pos, orn, offset_local, [0,0,0,1])
            return np.array(pos_offset)
        else:
            # Fallback
            state = p.getLinkState(self.robot_id, self.joint_indices[-1])
            return np.array(state[0])
    
    def _is_object_grasped(self) -> bool:
        """Check if object is being held - ROBUST detection."""
        if self.target_object_id is None:
            return False
        
        # Method 1: Check gripper is closed
        if self.gripper_state > 0.4:  # Gripper not closed enough
            return False
        
        # Method 2: Check ANY contact between gripper and object
        contact_count = 0
        for idx in self.gripper_indices:
            contacts = p.getContactPoints(
                bodyA=self.robot_id,
                linkIndexA=idx,
                bodyB=self.target_object_id
            )
            if contacts:
                contact_count += 1
        
        # Need at least one gripper finger touching
        if contact_count == 0:
            return False
        
        # Method 3: Check object is close to end effector
        ee_pos = self._get_end_effector_position()
        obj_pos, _ = p.getBasePositionAndOrientation(self.target_object_id)
        distance = np.linalg.norm(np.array(ee_pos) - np.array(obj_pos))
        
        # Object must be very close (within 5cm) to be "grasped"
        return distance < 0.05
    
    def _compute_reward(self) -> Tuple[float, bool, Dict]:
        """
        DEFINITIVE reward function for robotic arm grasping.
        
        Reward structure:
        1. Distance reward (continuous)
        2. Progress reward (getting closer)
        3. Position bonus (being above object)
        4. Gripper incentive (close when near)
        5. Grasp success (sparse bonus)
        6. Lift success (terminal bonus)
        """
        reward = 0.0
        terminated = False
        info = {}
        
        ee_pos = self._get_end_effector_position()
        
        if self.target_object_id is not None:
            obj_pos, _ = p.getBasePositionAndOrientation(self.target_object_id)
            obj_pos = np.array(obj_pos)
        else:
            obj_pos = self.target_position
        
        # Cube is 2.5cm, center at z=0.02, so top is at z≈0.03
        # Gripper should be at top of cube to grasp
        grasp_target = obj_pos.copy()
        grasp_target[2] = obj_pos[2] + 0.01  # Slightly into cube for contact
        
        # ================================================================
        # 1. DISTANCE REWARD - Main learning signal
        # ================================================================
        distance = np.linalg.norm(ee_pos - grasp_target)
        
        # Continuous distance reward: closer = better
        # Range [0, 1], with 0.5 at 10cm
        dist_reward = 1.0 / (1.0 + distance * 10.0)
        reward += dist_reward * 2.0
        
        # ================================================================
        # 2. PROGRESS REWARD - Reward improvement
        # ================================================================
        if self._prev_distance is None:
            self._prev_distance = distance
        
        improvement = self._prev_distance - distance
        reward += improvement * 30.0  # Strong gradient
        self._prev_distance = distance
        
        # ================================================================
        # 3. POSITION BONUS - Encourage top-down approach
        # ================================================================
        xy_dist = np.linalg.norm(ee_pos[:2] - obj_pos[:2])
        z_above = ee_pos[2] - obj_pos[2]  # Height above object
        
        if xy_dist < 0.08 and z_above > 0:
            reward += 1.0  # Good position bonus
        
        # ================================================================
        # 4. GRIPPER CONTROL - The critical part!
        # ================================================================
        # When close, gripper state matters A LOT
        if distance < 0.08:
            # Continuous reward for closing (gripper_state: 1=open, 0=closed)
            close_reward = (1.0 - self.gripper_state) * 10.0
            reward += close_reward
            
            # PENALTY for open gripper when very close
            if distance < 0.05 and self.gripper_state > 0.5:
                reward -= 5.0  # Strong penalty: CLOSE IT!
            
            # BONUS for closed gripper when very close
            if distance < 0.05 and self.gripper_state < 0.3:
                reward += 8.0  # Good job closing!
        
        # ================================================================
        # 5. GRASP SUCCESS - Big sparse reward
        # ================================================================
        is_grasped = self._is_object_grasped()
        
        if is_grasped and not self.object_grasped:
            self.object_grasped = True
            reward += 200.0  # HUGE reward!
            info['grasped'] = True
            print("🎯 GRASP!")
        
        # Holding bonus
        if self.object_grasped:
            reward += 10.0  # Keep holding
        
        # ================================================================
        # 6. LIFT SUCCESS - Ultimate goal
        # ================================================================
        if self.object_grasped and obj_pos[2] > 0.08:  # Lifted 8cm
            if not self.object_lifted:
                self.object_lifted = True
                reward += 1000.0  # MASSIVE SUCCESS!
                info['success'] = True
                print("🚀 LIFT SUCCESS!")
                terminated = True
        
        # ================================================================
        # 7. PENALTIES
        # ================================================================
        # Drop penalty
        if self.object_grasped and not is_grasped:
            self.object_grasped = False
            reward -= 50.0
        
        # Time penalty (small)
        reward -= 0.02
        
        # ================================================================
        # 8. SMOOTHNESS PENALTIES - Prevent jerky/violent movements
        # ================================================================
        
        # Penalty for high velocity when close to object
        if distance < 0.10 and hasattr(self, 'current_ee_velocity'):
            velocity_penalty = self.current_ee_velocity * 50.0  # Penalize speed near object
            reward -= velocity_penalty
        
        # Penalty for jerky movements (large changes in action)
        if hasattr(self, 'previous_action'):
            action_change = np.linalg.norm(self.previous_action[:4] - np.zeros(4))  # Assuming action from step
            jerk_penalty = action_change * 0.1  # Small penalty for erratic movements
            reward -= jerk_penalty
        
        # ================================================================
        # 9. OBJECT DISPLACEMENT PENALTY - Don't push the cube!
        # ================================================================
        # Check if object moved from its original position (pushed it)
        if not self.object_grasped and hasattr(self, 'initial_target_position'):
            obj_displacement = np.linalg.norm(obj_pos[:2] - self.initial_target_position[:2])
            if obj_displacement > 0.03:  # Object moved more than 3cm (was 2cm)
                reward -= obj_displacement * 50.0  # Reduced penalty (was 100.0)
                if obj_displacement > 0.10:
                    reward -= 10.0  # Penalty for really pushing it
        
        if terminated and self.object_lifted:
            self.recent_successes.append(1)
        elif terminated:
            self.recent_successes.append(0)
            
        # Keep only last 20
        if len(self.recent_successes) > 20:
            self.recent_successes.pop(0)

        info['distance'] = distance
        info['gripper'] = self.gripper_state
        info['success'] = self.object_lifted
        info['velocity'] = getattr(self, 'current_ee_velocity', 0.0)
        info['difficulty'] = self.difficulty_level
        
        return reward, terminated, info
    
    def set_difficulty(self, level: int):
        """Set difficulty level manually."""
        self.difficulty_level = np.clip(level, 1, 3)
        print(f"🎓 Difficulté changée au Niveau {self.difficulty_level}")

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment with CURRICULUM."""
        super().reset(seed=seed)
        
        self.episode_count += 1
        self.current_step = 0
        self.object_grasped = False
        self.object_lifted = False
        self._prev_distance = None  # Reset progress tracking
        
        # Reset robot position to a STABLE "Ready" pose
        # Instead of vertical (0,0,0,0), we pre-position it above table
        # Angles: Waist=0, Shoulder=-0.5 (fwd), Elbow=1.0 (bent), Wrist=-0.5 (down)
        self.current_joint_angles = np.array([0.0, -0.5, 1.0, -0.5])
        
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, self.current_joint_angles[i])
        
        # Open gripper
        self.gripper_state = 1.0
        self._set_gripper(1.0)
        
        # Create new target object
        self._create_target_object()
        
        # === CURRICULUM SPAWN LOGIC ===
        if self.difficulty_level == 1:
            # LEVEL 1: PERFECT POSITION (Calibrated by User)
            # Base: 20.5, Shoulder: 41.0, Elbow: 79.0, Wrist: 80.5
            self.current_joint_angles = np.array([20.5, 41.0, 79.0, 80.5])
        
        elif self.difficulty_level == 2:
            # LEVEL 2: Near Perfect (+/- 15 degrees noise)
            # Forces the robot to adjust slightly
            perfect_angles = np.array([20.5, 41.0, 79.0, 80.5])
            noise = np.random.uniform(-15, 15, size=4)
            self.current_joint_angles = perfect_angles + noise
            
        else:
            # LEVEL 3: Standard random spawn (zero position)
            self.current_joint_angles = np.zeros(4)

        # Apply joint angles
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, np.radians(self.current_joint_angles[i]))
            
        # Step simulation to settle
        for _ in range(20):
            p.stepSimulation()
        
        obs = self._get_observation()
        
        # Calculate recent success rate
        rate = 0.0
        if len(self.recent_successes) > 0:
            rate = sum(self.recent_successes) / len(self.recent_successes)
            
        info = {
            'success_rate': rate,
            'difficulty': self.difficulty_level
        }
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action with adaptive speed control."""
        self.current_step += 1
        
        # Store previous end effector position for velocity calculation
        prev_ee_pos = self._get_end_effector_position()
        
        # Get distance to target for adaptive speed
        if self.target_object_id is not None:
            obj_pos, _ = p.getBasePositionAndOrientation(self.target_object_id)
            distance_to_target = np.linalg.norm(prev_ee_pos - np.array(obj_pos))
        else:
            distance_to_target = 1.0
        
        # ADAPTIVE ACTION SCALING - Slower when close!
        if distance_to_target < 0.05:
            current_scale = self.action_scale_near * 0.5  # Very slow when very close
        elif distance_to_target < 0.10:
            current_scale = self.action_scale_near  # Slow when close
        elif distance_to_target < 0.15:
            # Interpolate between near and far scales
            t = (distance_to_target - 0.10) / 0.05
            current_scale = self.action_scale_near + t * (self.action_scale - self.action_scale_near)
        else:
            current_scale = self.action_scale  # Normal speed when far
        
        # Parse action with adaptive scaling
        joint_deltas = action[:4] * current_scale
        gripper_cmd = action[4]
        
        # Store previous for jerk calculation
        self.previous_joint_angles = self.current_joint_angles.copy()
        
        # Update joint angles
        self.current_joint_angles += joint_deltas
        self.current_joint_angles = np.clip(
            self.current_joint_angles,
            self.joint_limits_low,
            self.joint_limits_high
        )
        
        # Apply to robot with lower velocity for smoother motion
        for i, idx in enumerate(self.joint_indices):
            target_rad = np.radians(self.current_joint_angles[i])
            p.setJointMotorControl2(
                self.robot_id, idx,
                p.POSITION_CONTROL,
                targetPosition=target_rad,
                force=150,  # STRONG force to hold weight (was 50)
                maxVelocity=2.0  # Limit max velocity
            )
        
        # Update gripper
        self.gripper_state = (gripper_cmd + 1) / 2  # Convert [-1,1] to [0,1]
        self._set_gripper(self.gripper_state)
        
        # Step simulation (MORE substeps for better physics stability)
        for _ in range(10):  # Increased from 5 to 10
            p.stepSimulation()
        if self.render_mode == "human":
            time.sleep(1./60.)
            
            # --- DEBUG VISUALIZATION ---
            # Line from End Effector to Target
            # Red = Far, Green = Close/Grasped
            line_color = [1, 0, 0] if distance_to_target > 0.05 else [0, 1, 0]
            if self._is_object_grasped():
                line_color = [0, 0, 1]  # Blue if GRASPED
                
            p.addUserDebugLine(prev_ee_pos, obj_pos, line_color, lifeTime=0.1, lineWidth=3)
            
            # Text status above robot
            status_text = f"Dist: {distance_to_target:.3f}m | Grip: {self.gripper_state:.2f}"
            if self._is_object_grasped():
                status_text += " | GRASPED!"
            
            p.addUserDebugText(status_text, [0, 0, 0.3], [0, 0, 0], lifeTime=0.1, textSize=1.5)
            
            # ---------------------------
        
        # Calculate end effector velocity
        current_ee_pos = self._get_end_effector_position()
        self.current_ee_velocity = np.linalg.norm(current_ee_pos - prev_ee_pos)
        
        # Store action for smoothness calculation
        self.previous_action = action.copy()
        
        # Get observation and reward
        obs = self._get_observation()
        reward, terminated, info = self._compute_reward()
        truncated = False
        
        # --- CONSOLE LOGGING (User Request) ---
        if self.render_mode == "human" and self.current_step % 10 == 0:
            print(f"📉 Step {self.current_step}: Dist={distance_to_target:.3f} | Grip={self.gripper_state:.2f} | Rew={reward:.2f}")
        # ---------------------------
        
        return obs, reward, terminated, truncated, info
    
    def _set_gripper(self, openness: float):
        """Set gripper position."""
        for idx in self.gripper_indices:
            joint_info = p.getJointInfo(self.robot_id, idx)
            joint_name = joint_info[1].decode('utf-8')
            lower = joint_info[8]
            upper = joint_info[9]
            
            if 'left' in joint_name.lower():
                target = lower + openness * (upper - lower)
            else:
                target = upper + openness * (lower - upper)
            
            p.setJointMotorControl2(
                self.robot_id, idx,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=50  # Strong grip
            )
    
    def render(self) -> Optional[np.ndarray]:
        """Render the environment."""
        if self.render_mode == "rgb_array":
            view_matrix = p.computeViewMatrix([0.5, -0.3, 0.4], [0.2, 0, 0.1], [0, 0, 1])
            proj_matrix = p.computeProjectionMatrixFOV(60, 640/480, 0.01, 2.0)
            
            _, _, rgba, _, _ = p.getCameraImage(
                640, 480, view_matrix, proj_matrix,
                renderer=p.ER_TINY_RENDERER
            )
            
            return np.array(rgba).reshape(480, 640, 4)[:, :, :3]
        
        return None
    
    def close(self):
        """Clean up."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None


class AutoSaveCallback:
    """
    Callback personnalisé pour sauvegarder le modèle et VecNormalize automatiquement.
    Permet d'arrêter le training à tout moment sans perdre la progression.
    """
    def __init__(self, save_path: str, env, save_freq: int = 2048, verbose: int = 1):
        self.save_path = save_path
        self.env = env
        self.save_freq = save_freq
        self.verbose = verbose
        self.n_calls = 0
        self.last_save_step = 0
        
    def __call__(self, locals_dict, globals_dict):
        """Called at each step."""
        self.n_calls += 1
        
        # Sauvegarder à chaque save_freq steps
        if self.n_calls - self.last_save_step >= self.save_freq:
            self.last_save_step = self.n_calls
            
            # Récupérer le modèle
            model = locals_dict.get('self')
            if model is not None:
                # Sauvegarder le modèle
                model.save(self.save_path)
                
                # Sauvegarder VecNormalize
                if self.env is not None:
                    self.env.save(f"{self.save_path}_vecnorm.pkl")
                
                if self.verbose > 0:
                    print(f"\n💾 Auto-save @ step {self.n_calls:,} -> {self.save_path}.zip")
        
        return True  # Continue training


def train_agent(total_timesteps: int = 100000, save_path: str = "models/robot_arm_ai", resume: bool = True):
    """
    Train the RL agent with automatic checkpointing.
    
    Args:
        total_timesteps: Number of timesteps to train
        save_path: Path to save the model
        resume: If True, resume from existing model if available
    """
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("❌ stable-baselines3 non installé!")
        print("   Run: pip install stable-baselines3[extra]")
        return None

    # ... (skipping lines) ...

    # Create new environment
    # Wrap with Monitor to get 'ep_rew_mean' and 'ep_len_mean' in logs
    env = DummyVecEnv([lambda: Monitor(RobotArmEnv(render_mode=None, max_steps=500))])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # === Custom Callback for frequent saves ===
    class FrequentSaveCallback(BaseCallback):
        """
        Sauvegarde le modèle et VecNormalize après chaque rollout (2048 steps).
        Permet d'arrêter le training à tout moment sans perdre la progression.
        """
        def __init__(self, save_path: str, env, verbose: int = 1):
            super().__init__(verbose)
            self.save_path = save_path
            self.env = env
            self.save_count = 0
            
        def _on_rollout_end(self) -> None:
            """Called after each rollout collection."""
            self.save_count += 1
            
            # Sauvegarder le modèle
            self.model.save(self.save_path)
            
            # Sauvegarder VecNormalize (CRITIQUE pour reprendre le training!)
            if self.env is not None:
                self.env.save(f"{self.save_path}_vecnorm.pkl")
            
            if self.verbose > 0:
                current_steps = self.num_timesteps
                print(f"\n💾 Auto-save #{self.save_count} @ {current_steps:,} steps -> {self.save_path}.zip")
        
        def _on_step(self) -> bool:
            return True

    # === Curriculum Level Manager Callback ===
    class CurriculumCallback(BaseCallback):
        """
        Gère la montée de niveau automatiquement.
        """
        def __init__(self, check_freq: int = 2000, verbose: int = 1):
            super().__init__(verbose)
            self.check_freq = check_freq
            
        def _on_step(self) -> bool:
            # Check every check_freq steps
            if self.n_calls % self.check_freq == 0:
                # Access the environment via training_env
                # Note: This assumes DummyVecEnv -> Monitor -> RobotArmEnv
                # We need to access the underlying RobotArmEnv
                env = self.training_env.envs[0].unwrapped
                
                # Check success rate of last 5 episodes
                if hasattr(env, 'recent_successes') and len(env.recent_successes) >= 5:
                    rate = sum(env.recent_successes) / len(env.recent_successes)
                    
                    print(f"   [Curriculum DEBUG] Rate: {rate:.2f} | Buffer: {list(env.recent_successes)} | Level: {env.difficulty_level}")
                    
                    if rate >= 0.8 and env.difficulty_level < 3:
                        env.difficulty_level += 1
                        print(f"\n🎉 NIVEAU SUPÉRIEUR ! Success Rate: {rate*100:.1f}% -> Passage au niveau {env.difficulty_level}")
                        print(f"   La tâche devient plus difficile !")
                    
                    if self.verbose > 0:
                        print(f"   [Curriculum] Level {env.difficulty_level} | Success Rate (last 5): {rate*100:.1f}%")
            return True

    print("=" * 60)
    print("🧠 ENTRAÎNEMENT IA - Bras Robotique (Curriculum Mode)")
    print("=" * 60)
    print(f"   Algorithme: PPO (Proximal Policy Optimization)")
    print(f"   Steps: {total_timesteps:,}")
    print(f"   Save path: {save_path}")
    print(f"   Auto-save: Toutes les 2048 steps (après chaque rollout)")
    print("=" * 60)
    
    # Create directories
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
    os.makedirs("models/checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Check if we should resume from existing model
    model_exists = os.path.exists(f"{save_path}.zip")
    vecnorm_exists = os.path.exists(f"{save_path}_vecnorm.pkl")
    
    if resume and model_exists and vecnorm_exists:
        print("\n🔄 Reprise du training depuis le checkpoint existant...")
        print(f"   Modèle: {save_path}.zip")
        print(f"   VecNorm: {save_path}_vecnorm.pkl")
        
        # Load existing VecNormalize
        vec_env = DummyVecEnv([lambda: RobotArmEnv(render_mode=None, max_steps=500)])
        env = VecNormalize.load(f"{save_path}_vecnorm.pkl", vec_env)
        env.training = True
        env.norm_reward = True
        
        # Load existing model
        model = PPO.load(save_path, env=env, tensorboard_log="./logs/tensorboard/")
        print(f"   ✅ Modèle chargé avec succès!")
    else:
        if resume and model_exists:
            print("\n⚠️ Modèle trouvé mais pas de VecNormalize. Nouveau training...")
        elif resume:
            print("\n📝 Pas de checkpoint trouvé. Nouveau training...")
        
        # Create new environment
        # Wrap with Monitor to track episode stats (reward, length)
        env = DummyVecEnv([lambda: Monitor(RobotArmEnv(render_mode=None, max_steps=500))])
        env = VecNormalize(env, norm_obs=True, norm_reward=True)
        
        # Create new model
        model = PPO(
            "MlpPolicy",
            env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            verbose=1,
            tensorboard_log="./logs/tensorboard/",
            device="cuda"  # Force GPU usage
        )
    
    # Create evaluation environment - DISABLED
    # eval_env = DummyVecEnv([lambda: RobotArmEnv(render_mode=None, max_steps=200)])
    # eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    
    # === Callbacks ===
    
    # 0. Curriculum Callback
    curriculum_callback = CurriculumCallback(check_freq=1000)

    # 1. Auto-save callback (sauvegarde après CHAQUE rollout = 2048 steps)
    autosave_callback = FrequentSaveCallback(save_path, env, verbose=1)
    
    # 2. Checkpoint callback (sauvegarde numérotée toutes les 10000 steps)
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="robot_arm",
        save_vecnormalize=True  # Aussi sauvegarder VecNormalize
    )
    
    # 3. Evaluation callback - DISABLED
    # eval_callback = EvalCallback(...)
    
    print("\n🚀 Début de l'entraînement...")
    print("   💡 Vous pouvez arrêter à tout moment avec Ctrl+C")
    print("   💾 Le modèle est sauvegardé automatiquement!\n")
    
    try:
        # Train with all callbacks (EvalCallback removed)
        model.learn(
            total_timesteps=total_timesteps,
            callback=[curriculum_callback, autosave_callback, checkpoint_callback],
            progress_bar=True,
            reset_num_timesteps=not (resume and model_exists and vecnorm_exists)
        )
        
        # Save final model
        model.save(save_path)
        env.save(f"{save_path}_vecnorm.pkl")
        
        print(f"\n✅ Entraînement terminé!")
        print(f"   Modèle sauvegardé: {save_path}.zip")
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Entraînement interrompu par l'utilisateur!")
        print("   💾 Sauvegarde finale en cours...")
        
        # Save on interrupt
        model.save(save_path)
        env.save(f"{save_path}_vecnorm.pkl")
        
        print(f"   ✅ Modèle sauvegardé: {save_path}.zip")
        print(f"   📝 Pour reprendre: python IA.py --train")
    
    finally:
        env.close()
        # eval_env.close()
    
    return model


def test_agent(model_path: str = "models/robot_arm_ai", episodes: int = 10, stochastic: bool = False):
    """Test a trained agent with proper normalization."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError:
        print("❌ stable-baselines3 non installé!")
        return
    
    print("=" * 60)
    print(f"🧪 TEST IA - Bras Robotique (Stochastic: {stochastic})")
    print("=" * 60)
    
    # Load model
    if not os.path.exists(f"{model_path}.zip"):
        print(f"❌ Modèle non trouvé: {model_path}.zip")
        print("   Entraînez d'abord avec: python IA.py --train")
        return
    
    model = PPO.load(model_path)
    
    # Load normalization stats
    vecnorm_path = f"{model_path}_vecnorm.pkl"
    
    if os.path.exists(vecnorm_path):
        print(f"   ✅ Normalisation chargée: {vecnorm_path}")
        # Create vectorized env with normalization
        vec_env = DummyVecEnv([lambda: RobotArmEnv(render_mode="human", max_steps=500)])
        env = VecNormalize.load(vecnorm_path, vec_env)
        env.training = False  # Don't update stats during test
        env.norm_reward = False  # Don't normalize rewards
        use_vec_env = True
    else:
        print(f"   ⚠️ Pas de normalisation trouvée, utilisation raw")
        env = RobotArmEnv(render_mode="human", max_steps=500)
        use_vec_env = False
    
    successes = 0
    total_rewards = []
    
    for ep in range(episodes):
        if use_vec_env:
            obs = env.reset()
        else:
            obs, info = env.reset()
        
        done = False
        episode_reward = 0
        steps = 0
        
        print(f"\n📺 Episode {ep + 1}/{episodes}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=not stochastic)
            
            if use_vec_env:
                obs, rewards, dones, infos = env.step(action)
                reward = rewards[0]
                done = dones[0]
                info = infos[0]
            else:
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
            
            episode_reward += reward
            steps += 1
            time.sleep(0.01)
        
        total_rewards.append(episode_reward)
        
        if info.get('success', False):
            successes += 1
            print(f"   ✅ Succès! Reward: {episode_reward:.1f} ({steps} steps)")
        else:
            print(f"   ❌ Échec. Reward: {episode_reward:.1f} (dist: {info.get('distance', 0):.3f}m)")
    
    print(f"\n{'='*60}")
    print(f"📊 Résultats: {successes}/{episodes} succès ({100*successes/episodes:.1f}%)")
    print(f"   Reward moyen: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    print(f"{'='*60}")
    
    if use_vec_env:
        env.close()
    else:
        env.close()


def demo_untrained(episodes: int = 3):
    """Demo with random actions (before training)."""
    print("=" * 60)
    print("🎮 DÉMO - Environnement (avant entraînement)")
    print("=" * 60)
    
    env = RobotArmEnv(render_mode="human", max_steps=100)
    
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        
        print(f"\n📺 Episode {ep + 1}/{episodes} (actions aléatoires)")
        
        while not done:
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            time.sleep(0.02)
        
        print(f"   Distance finale: {info['distance']:.3f}m")
    
    env.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="IA Bras Robotique")
    
    parser.add_argument('--train', action='store_true',
                        help='Entraîner l\'IA')
    parser.add_argument('--test', action='store_true',
                        help='Tester l\'IA entraînée')
    parser.add_argument('--demo', action='store_true',
                        help='Démo avant entraînement')
    parser.add_argument('--steps', type=int, default=100000,
                        help='Nombre de steps d\'entraînement')
    parser.add_argument('--episodes', type=int, default=10,
                        help='Nombre d\'épisodes de test')
    parser.add_argument('--model', type=str, default='models/robot_arm_ai',
                        help='Chemin du modèle')
    
    parser.add_argument('--stochastic', action='store_true',
                        help='Test stochastique (non déterministe)')
    
    parser.add_argument('--resume', action='store_true',
                        help='Reprendre l\'entraînement depuis le checkpoint')
    
    args = parser.parse_args()
    
    if args.train:
        train_agent(total_timesteps=args.steps, save_path=args.model, resume=args.resume)
    elif args.test:
        test_agent(model_path=args.model, episodes=args.episodes, stochastic=args.stochastic)
    elif args.demo:
        demo_untrained(episodes=args.episodes)
    else:
        print("🤖 IA Bras Robotique - Apprentissage par Renforcement")
        print()
        print("Usage:")
        print("  python IA.py --demo           # Voir l'environnement")
        print("  python IA.py --train          # Entraîner l'IA (100k steps)")
        print("  python IA.py --train --steps 500000  # Entraînement long")
        print("  python IA.py --test           # Tester l'IA entraînée")
        print()
        print("Prérequis:")
        print("  pip install stable-baselines3[extra]")


if __name__ == "__main__":
    main()
