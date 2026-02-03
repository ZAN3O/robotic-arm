"""
IA_ULTRA_FIXED.py - Version ULTRA Optimisée pour Apprentissage Lift

🔥 NOUVEAUTÉS CRITIQUES :
1. ✅ DENSE REWARD sur hauteur (reward continu, pas juste +50 au seuil)
2. ✅ CURRICULUM DYNAMIQUE (alterne facile/difficile pour généralisation)
3. ✅ SEUIL PROGRESSIF (commence à 2cm, monte graduellement)
4. ✅ HELPER REWARDS pour mouvement vertical
5. ✅ Max steps augmenté (300 au lieu de 250)

Le robot va ENFIN comprendre : Trouver → Saisir → LEVER !
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
    ULTRA FIXED Gymnasium environment - Focus sur apprentissage LIFT.
    
    Observation space (20 dims): ⚠️ EXTENDED ENCORE
        [0-3]   Joint angles (normalized)
        [4]     Gripper state
        [5-7]   EE position (x, y, z)
        [8-10]  EE velocity (vx, vy, vz)
        [11-13] Relative vector to target
        [14]    Distance to target
        [15]    Contact state
        [16]    Object height (obj_z)
        [17]    Object velocity Z (obj_vz)
        [18]    Gripper change rate
        [19]    Lift progress (0-1) ← NOUVEAU
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(self, render_mode: str = None, max_steps: int = 300):
        super().__init__()
        
        self.render_mode = render_mode
        self.max_steps = max_steps  # Augmenté de 250 à 300
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
        
        # Action scaling - ENCORE PLUS LENT
        self.action_scale = 2.0  # Reduced from 2.5
        self.action_scale_near = 0.8  # Very slow when close
        
        # 🔥 NOUVEAU: Observation space étendu (20D)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(20,),  # Was 19, now 20
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
        self.previous_joint_angles = np.zeros(4)
        self.previous_action = np.zeros(5)
        self.gripper_state = 1.0
        self.previous_gripper_state = 1.0
        self.target_position = np.zeros(3)
        self.object_grasped = False
        self.object_lifted = False
        
        # Velocity tracking
        self.ee_velocity = np.zeros(3)
        self.previous_ee_pos = np.zeros(3)
        self.object_velocity_z = 0.0
        self.previous_object_z = 0.0
        
        # 🔥 NOUVEAU: Tracking pour dense rewards
        self.max_object_height_reached = 0.0
        self.initial_object_z = 0.0
        self.best_lift_this_episode = 0.0
        
        # Stage tracking (simplifié)
        self.current_stage = 0
        self.stage_validated = [False, False, False]
        
        # Success tracking
        self.success_count = 0
        self.episode_count = 0
        self.recent_successes = []
        
        # 🔥 CURRICULUM DYNAMIQUE
        self.difficulty_level = 1
        self.lift_threshold = 0.02  # Commence à 2cm (FACILE)
        self.target_lift_threshold = 0.05  # Objectif final 5cm
        
        # 🔥 Curriculum alternance (pour généralisation)
        self.curriculum_mode = "progressive"  # "progressive" ou "alternating"
        
        print(f"🎓 Environnement ULTRA initialisé")
        print(f"   Lift threshold: {self.lift_threshold*100:.1f}cm → {self.target_lift_threshold*100:.1f}cm")
        
        # Initialize simulation
        self._setup_simulation()
    
    def _setup_simulation(self):
        """Initialize PyBullet simulation."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
        
        if p.isConnected():
            p.disconnect()
            time.sleep(0.1)
            
        # Connect
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
        
        # Better physics
        p.setPhysicsEngineParameter(
            fixedTimeStep=1./240.,
            numSolverIterations=100,
            numSubSteps=4,
            contactBreakingThreshold=0.001
        )
        
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        
        # Load environment
        self.plane_id = p.loadURDF("plane.urdf")
        self._create_table()
        self._load_robot()
    
    def _create_table(self):
        """Create table BELOW robot."""
        len_x = 0.4
        width_y = 0.4
        thick = 0.01
        
        table_col = p.createCollisionShape(
            p.GEOM_BOX, halfExtents=[len_x, width_y, thick]
        )
        table_vis = p.createVisualShape(
            p.GEOM_BOX, halfExtents=[len_x, width_y, thick],
            rgbaColor=[0.6, 0.4, 0.2, 1]
        )
        
        self.table_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=table_col,
            baseVisualShapeIndex=table_vis,
            basePosition=[0.3, 0, -0.05]
        )
        
        p.changeDynamics(
            self.table_id, -1,
            lateralFriction=2.0,
            rollingFriction=0.01,
            spinningFriction=0.01
        )
    
    def _load_robot(self):
        """Load robot with HIGH FRICTION."""
        if hasattr(self, 'robot_id') and self.robot_id is not None:
            try:
                p.removeBody(self.robot_id)
            except:
                pass

        urdf_path = os.path.join(os.path.dirname(__file__), "arduino_arm.urdf")
        
        self.robot_id = p.loadURDF(
            urdf_path,
            basePosition=[0, 0, 0],
            useFixedBase=True
        )
        
        # Find joints
        self.joint_indices = []
        self.gripper_indices = []
        
        for i in range(p.getNumJoints(self.robot_id)):
            joint_info = p.getJointInfo(self.robot_id, i)
            joint_name = joint_info[1].decode('utf-8')
            joint_type = joint_info[2]
            
            if joint_type == p.JOINT_REVOLUTE:
                if 'gripper' in joint_name.lower():
                    self.gripper_indices.append(i)
                else:
                    self.joint_indices.append(i)
        
        # MAX FRICTION on gripper
        for gripper_idx in self.gripper_indices:
            p.changeDynamics(
                self.robot_id, gripper_idx,
                lateralFriction=3.0,
                rollingFriction=0.01,
                spinningFriction=0.01
            )
    
    def _create_target_object(self):
        """
        🔥 CURRICULUM DYNAMIQUE avec alternance.
        """
        # 🔥 Alternance facile/moyen/difficile pour généralisation
        if self.curriculum_mode == "alternating":
            # Cycle entre les niveaux pour éviter overfitting
            cycle_levels = [1, 1, 2, 1, 2, 3, 2, 1]  # Pattern varié
            cycle_idx = self.episode_count % len(cycle_levels)
            level = cycle_levels[cycle_idx]
        else:
            # Mode progressif standard
            level = self.difficulty_level
        
        # Position selon niveau
        if level == 1:
            # TRÈS FACILE: Très proche et centré
            x_range = (0.14, 0.18)  # Plus proche qu'avant
            y_range = (-0.03, 0.03)
        elif level == 2:
            x_range = (0.12, 0.22)
            y_range = (-0.08, 0.08)
        else:
            x_range = (0.10, 0.28)
            y_range = (-0.12, 0.12)
        
        x = np.random.uniform(*x_range)
        y = np.random.uniform(*y_range)
        z = 0.015  # 1.5cm above table
        
        self.target_position = np.array([x, y, z])
        self.initial_object_z = z
        
        # Cube
        size = 0.0125
        
        vis_shape = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[size, size, size],
            rgbaColor=[0.1, 0.9, 0.1, 1]
        )
        col_shape = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[size, size, size]
        )
        
        self.target_object_id = p.createMultiBody(
            baseMass=0.01,  # 10g
            baseCollisionShapeIndex=col_shape,
            baseVisualShapeIndex=vis_shape,
            basePosition=self.target_position
        )
        
        # MAX FRICTION
        p.changeDynamics(
            self.target_object_id, -1,
            lateralFriction=3.0,
            rollingFriction=0.01,
            spinningFriction=0.01,
            restitution=0.1
        )
    
    def _get_observation(self) -> np.ndarray:
        """
        🔥 EXTENDED: 20D observation avec lift progress.
        """
        # Joint states
        joint_states = np.array([
            p.getJointState(self.robot_id, idx)[0]
            for idx in self.joint_indices
        ])
        self.current_joint_angles = np.degrees(joint_states)
        
        # Normalize
        normalized_angles = 2 * (self.current_joint_angles - self.joint_limits_low) / \
                           (self.joint_limits_high - self.joint_limits_low) - 1
        normalized_angles = np.clip(normalized_angles, -1, 1)
        
        # Gripper
        if len(self.gripper_indices) >= 2:
            left_pos = p.getJointState(self.robot_id, self.gripper_indices[0])[0]
            gripper_normalized = (left_pos + 0.8) / 0.8
            self.gripper_state = np.clip(gripper_normalized, 0, 1)
        
        # EE position
        ee_link_state = p.getLinkState(self.robot_id, self.gripper_indices[0] - 1)
        ee_pos = np.array(ee_link_state[0])
        
        # EE velocity
        if hasattr(self, 'previous_ee_pos'):
            self.ee_velocity = (ee_pos - self.previous_ee_pos) * 240
        self.previous_ee_pos = ee_pos.copy()
        
        # Target info
        if self.target_object_id is not None:
            obj_pos, _ = p.getBasePositionAndOrientation(self.target_object_id)
            obj_pos = np.array(obj_pos)
            
            # Object velocity Z
            if hasattr(self, 'previous_object_z'):
                self.object_velocity_z = (obj_pos[2] - self.previous_object_z) * 240
            self.previous_object_z = obj_pos[2]
        else:
            obj_pos = self.target_position
            self.object_velocity_z = 0.0
        
        # Relative vector
        relative_vec = obj_pos - ee_pos
        distance = np.linalg.norm(relative_vec)
        
        # Contact
        contact_points = p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=self.target_object_id
        )
        has_contact = 1.0 if len(contact_points) > 0 else 0.0
        
        # Gripper change
        gripper_change = abs(self.gripper_state - self.previous_gripper_state)
        self.previous_gripper_state = self.gripper_state
        
        # 🔥 NOUVEAU: Lift progress (0-1)
        # Mesure combien le cube a été levé par rapport à l'objectif
        lift_height = obj_pos[2] - self.initial_object_z
        lift_progress = np.clip(lift_height / self.lift_threshold, 0, 1)
        
        # Update max height
        self.max_object_height_reached = max(self.max_object_height_reached, lift_height)
        
        # Construct observation
        obs = np.concatenate([
            normalized_angles,              # [0-3]
            [self.gripper_state],          # [4]
            ee_pos,                         # [5-7]
            self.ee_velocity,               # [8-10]
            relative_vec,                   # [11-13]
            [distance],                     # [14]
            [has_contact],                  # [15]
            [obj_pos[2]],                   # [16]
            [self.object_velocity_z],       # [17]
            [gripper_change],               # [18]
            [lift_progress]                 # [19] ← NOUVEAU
        ]).astype(np.float32)
        
        return obs
    
    def _compute_reward(self) -> Tuple[float, Dict[str, Any]]:
        """
        🔥 DENSE REWARD SHAPING avec focus LIFT.
        
        Philosophie : Récompenser TOUT progrès vers le lift, pas juste le succès final.
        """
        reward = 0.0
        info = {
            'stage': self.current_stage,
            'distance': 0.0,
            'contact': False,
            'lifted': False,
            'lift_height': 0.0,
            'success': False
        }
        
        # Get state
        ee_link_state = p.getLinkState(self.robot_id, self.gripper_indices[0] - 1)
        ee_pos = np.array(ee_link_state[0])
        
        obj_pos, _ = p.getBasePositionAndOrientation(self.target_object_id)
        obj_pos = np.array(obj_pos)
        
        distance = np.linalg.norm(obj_pos - ee_pos)
        info['distance'] = distance
        
        # Contact
        contact_points = p.getContactPoints(
            bodyA=self.robot_id,
            bodyB=self.target_object_id
        )
        has_contact = len(contact_points) > 0
        info['contact'] = has_contact
        
        # 🔥 Hauteur lift (métrique clé)
        lift_height = obj_pos[2] - self.initial_object_z
        info['lift_height'] = lift_height
        
        # Success check (avec seuil dynamique)
        object_lifted = lift_height > self.lift_threshold
        info['lifted'] = object_lifted
        
        # EE velocity
        ee_vel_mag = np.linalg.norm(self.ee_velocity)
        
        # ========================================================================
        # 🔥 REWARD COMPONENT 1: APPROACH (Distance-based)
        # ========================================================================
        if distance > 0.05:
            # Far from object - reward for approaching
            approach_reward = 2.0 * (1.0 - np.tanh(distance * 5))
            reward += approach_reward
        else:
            # Close to object - bonus for being near
            reward += 3.0
        
        # ========================================================================
        # 🔥 REWARD COMPONENT 2: CONTACT (With velocity check)
        # ========================================================================
        if has_contact:
            if ee_vel_mag < 0.5:
                # Good contact (gentle)
                reward += 5.0
                
                # 🔥 Bonus for closed gripper WHILE in contact
                if self.gripper_state < 0.3:
                    reward += 3.0
            else:
                # Bad contact (smash)
                reward -= 10.0
        
        # ========================================================================
        # 🔥 REWARD COMPONENT 3: LIFT (DENSE - Le plus important!)
        # ========================================================================
        if has_contact and self.gripper_state < 0.4:
            # On considère que l'objet est "saisi" si contact + pince fermée
            
            # 🔥 DENSE HEIGHT REWARD (récompense CONTINUE)
            # Plus l'objet monte, plus on gagne de points
            height_reward = 20.0 * np.tanh(lift_height * 30)  # Exponential growth
            reward += height_reward
            
            # 🔥 Bonus pour chaque centimètre
            if lift_height > 0.01:  # 1cm
                reward += 5.0
            if lift_height > 0.02:  # 2cm
                reward += 10.0
            if lift_height > 0.03:  # 3cm
                reward += 15.0
            if lift_height > 0.04:  # 4cm
                reward += 20.0
            
            # 🔥 JACKPOT si seuil atteint
            if object_lifted:
                self.object_lifted = True
                reward += 100.0  # HUGE REWARD (augmenté de 50 à 100)
                info['success'] = True
            
            # 🔥 Bonus pour mouvement vertical POSITIF
            if self.object_velocity_z > 0.1:  # Monte
                reward += 2.0 * self.object_velocity_z
            elif self.object_velocity_z < -0.1:  # Descend (BAD)
                reward -= 5.0
            
            # 🔥 Bonus pour nouveau record de hauteur
            if lift_height > self.best_lift_this_episode:
                improvement = lift_height - self.best_lift_this_episode
                reward += 10.0 * improvement / 0.01  # 10 pts par mm de progrès
                self.best_lift_this_episode = lift_height
        
        # ========================================================================
        # PENALTIES
        # ========================================================================
        
        # Anti-smash (quand proche)
        if distance < 0.05 and ee_vel_mag > 0.4:
            reward -= 8.0 * (ee_vel_mag - 0.4)
        
        # Anti-oscillation gripper
        if gripper_change := abs(self.gripper_state - self.previous_gripper_state):
            if gripper_change > 0.3:
                reward -= 0.5 * gripper_change
        
        # Action smoothness
        action_change = np.linalg.norm(self.previous_action)
        if action_change > 0.5:
            reward -= 0.1 * action_change
        
        # Penalty pour laisser tomber l'objet
        if self.max_object_height_reached > 0.02 and lift_height < 0.01:
            reward -= 20.0  # Grosse pénalité pour drop
            info['dropped'] = True
        
        # Small time penalty
        reward -= 0.01
        
        return reward, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action with smoothing."""
        self.current_step += 1
        
        # Clip and smooth
        action = np.clip(action, -1.0, 1.0)
        smoothing_factor = 0.3
        action = smoothing_factor * action + (1 - smoothing_factor) * self.previous_action
        self.previous_action = action.copy()
        
        # Parse action
        joint_actions = action[:4]
        gripper_action = action[4]
        
        # Adaptive scaling
        ee_link_state = p.getLinkState(self.robot_id, self.gripper_indices[0] - 1)
        ee_pos = np.array(ee_link_state[0])
        obj_pos, _ = p.getBasePositionAndOrientation(self.target_object_id)
        distance = np.linalg.norm(np.array(obj_pos) - ee_pos)
        
        if distance < 0.05:
            scale = self.action_scale_near
        else:
            scale = self.action_scale
        
        # Apply joints
        delta_angles = joint_actions * scale
        target_angles = self.current_joint_angles + delta_angles
        target_angles = np.clip(target_angles, self.joint_limits_low, self.joint_limits_high)
        
        for i, joint_idx in enumerate(self.joint_indices):
            p.setJointMotorControl2(
                self.robot_id,
                joint_idx,
                p.POSITION_CONTROL,
                targetPosition=np.radians(target_angles[i]),
                force=50,
                maxVelocity=1.0
            )
        
        # Gripper with deadzone
        if abs(gripper_action) > 0.2:
            if gripper_action > 0.2:
                target_gripper = 1.0
            else:
                target_gripper = 0.0
            
            self.gripper_state = 0.9 * self.gripper_state + 0.1 * target_gripper
        
        # Apply gripper
        left_target = -0.8 * (1 - self.gripper_state)
        right_target = 0.8 * (1 - self.gripper_state)
        
        for i, gripper_idx in enumerate(self.gripper_indices):
            target = left_target if i == 0 else right_target
            p.setJointMotorControl2(
                self.robot_id,
                gripper_idx,
                p.POSITION_CONTROL,
                targetPosition=target,
                force=20,
                maxVelocity=0.5
            )
        
        # Step simulation
        p.stepSimulation()
        
        # Get observation
        obs = self._get_observation()
        
        # Compute reward
        reward, info = self._compute_reward()
        
        # Check termination
        terminated = False
        truncated = False
        
        if self.object_lifted or info.get('success', False):
            terminated = True
            self.success_count += 1
            self.recent_successes.append(1)
            print(f"   🎉 SUCCESS! Lift height: {info['lift_height']*100:.1f}cm")
        
        if self.current_step >= self.max_steps:
            truncated = True
            if not terminated:
                self.recent_successes.append(0)
                # Log best attempt
                if self.max_object_height_reached > 0.01:
                    print(f"   📏 Max lift: {self.max_object_height_reached*100:.1f}cm (target: {self.lift_threshold*100:.1f}cm)")
        
        # Limit history
        if len(self.recent_successes) > 20:
            self.recent_successes.pop(0)
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)
        
        self.current_step = 0
        self.episode_count += 1
        
        # Reset tracking
        self.max_object_height_reached = 0.0
        self.best_lift_this_episode = 0.0
        self.current_stage = 0
        self.stage_validated = [False, False, False]
        
        # 🔥 CURRICULUM: Adapter le seuil de lift
        if len(self.recent_successes) > 10:
            success_rate = sum(self.recent_successes[-20:]) / min(20, len(self.recent_successes))
            
            # Si taux élevé, augmenter progressivement le seuil
            if success_rate > 0.6 and self.lift_threshold < self.target_lift_threshold:
                old_threshold = self.lift_threshold
                self.lift_threshold = min(self.lift_threshold + 0.005, self.target_lift_threshold)
                if old_threshold != self.lift_threshold:
                    print(f"   🎯 Lift threshold increased: {old_threshold*100:.1f}cm → {self.lift_threshold*100:.1f}cm")
            
            # Log stats periodiquement
            if self.episode_count % 20 == 0:
                avg_max_height = np.mean([self.max_object_height_reached] * min(20, len(self.recent_successes)))
                print(f"   📊 [Env] History: {len(self.recent_successes)} | Rate: {success_rate*100:.0f}% | Level: {self.difficulty_level}")
                print(f"        Avg max height: {avg_max_height*100:.1f}cm | Threshold: {self.lift_threshold*100:.1f}cm")
        
        # Remove old object
        if self.target_object_id is not None:
            p.removeBody(self.target_object_id)
        
        # Create new target
        self._create_target_object()
        
        # Reset robot
        home_angles = np.array([0, 25, 110, 25])
        for i, joint_idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, joint_idx, np.radians(home_angles[i]))
        
        # Reset gripper (open)
        for i, gripper_idx in enumerate(self.gripper_indices):
            p.resetJointState(self.robot_id, gripper_idx, 0)
        
        self.gripper_state = 1.0
        self.previous_gripper_state = 1.0
        self.object_grasped = False
        self.object_lifted = False
        self.current_joint_angles = home_angles
        self.previous_action = np.zeros(5)
        self.ee_velocity = np.zeros(3)
        self.previous_ee_pos = np.zeros(3)
        self.object_velocity_z = 0.0
        self.previous_object_z = 0.0
        
        # Settle
        for _ in range(10):
            p.stepSimulation()
        
        obs = self._get_observation()
        info = {'episode': self.episode_count, 'lift_threshold': self.lift_threshold}
        
        return obs, info
    
    def set_difficulty(self, level: int):
        """Set difficulty level."""
        self.difficulty_level = max(1, min(3, level))
        print(f"🎯 Difficulté: Niveau {self.difficulty_level}")
    
    def set_curriculum_mode(self, mode: str):
        """Set curriculum mode: 'progressive' or 'alternating'."""
        self.curriculum_mode = mode
        print(f"📚 Curriculum mode: {mode}")
    
    def close(self):
        """Clean up."""
        if self.physics_client is not None:
            p.disconnect(self.physics_client)
            self.physics_client = None
    
    def render(self):
        """Handled by PyBullet GUI."""
        pass


# ============================================================================
# TRAINING (Optimized)
# ============================================================================

def train_agent(total_timesteps: int = 200000, save_path: str = "models/robot_arm_ultra", 
                resume: bool = True, render: bool = False):
    """Train with ULTRA optimizations."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
        from stable_baselines3.common.monitor import Monitor
    except ImportError:
        print("❌ stable-baselines3 non installé!")
        return None

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs("logs/tensorboard_ultra", exist_ok=True)
    
    render_mode = "human" if render else None
    
    # Max steps 300 (au lieu de 250)
    env = DummyVecEnv([lambda: Monitor(RobotArmEnv(render_mode=render_mode, max_steps=300))])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    class AutoSaveCallback(BaseCallback):
        def __init__(self, save_path: str, env, verbose: int = 1):
            super().__init__(verbose)
            self.save_path = save_path
            self.env = env
            self.save_count = 0
        
        def _on_rollout_end(self) -> None:
            self.save_count += 1
            self.model.save(self.save_path)
            if self.env is not None: 
                self.env.save(f"{self.save_path}_vecnorm.pkl")
            if self.save_count % 5 == 0 and self.verbose > 0:
                print(f"\n💾 Auto-save #{self.save_count} @ {self.num_timesteps:,} steps")
        
        def _on_step(self) -> bool: 
            return True

    print("=" * 60)
    print("🔥 ENTRAÎNEMENT ULTRA - Focus LIFT")
    print("=" * 60)
    
    model_exists = os.path.exists(f"{save_path}.zip")
    vecnorm_exists = os.path.exists(f"{save_path}_vecnorm.pkl")
    
    if resume and model_exists and vecnorm_exists:
        print("\n🔄 Reprise du training...")
        vec_env = DummyVecEnv([lambda: RobotArmEnv(render_mode=None, max_steps=300)])
        env = VecNormalize.load(f"{save_path}_vecnorm.pkl", vec_env)
        env.training = True
        model = PPO.load(save_path, env=env, tensorboard_log="./logs/tensorboard_ultra/")
    else:
        print("\n📝 Nouveau training...")
        model = PPO(
            "MlpPolicy", env, 
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.995,  # Augmenté de 0.99 pour mieux considérer long-terme
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,  # Un peu d'exploration
            verbose=1, 
            tensorboard_log="./logs/tensorboard_ultra/",
            device="cpu"  # Force CPU (mieux pour PPO non-CNN)
        )
    
    autosave_callback = AutoSaveCallback(save_path, env, verbose=1)
    
    try:
        model.learn(
            total_timesteps=total_timesteps, 
            callback=[autosave_callback], 
            progress_bar=True, 
            reset_num_timesteps=not (resume and model_exists)
        )
        model.save(save_path)
        env.save(f"{save_path}_vecnorm.pkl")
        print(f"\n✅ Entraînement terminé!")
    except KeyboardInterrupt:
        print("\n\n⚠️ Interrompu!")
        model.save(save_path)
        env.save(f"{save_path}_vecnorm.pkl")
        print(f"   ✅ Sauvegardé.")
    finally:
        env.close()
    
    return model


def test_agent(model_path: str = "models/robot_arm_ultra", episodes: int = 10, 
               stochastic: bool = False, level: int = 1):
    """Test agent."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError: 
        return

    print(f"🔄 Chargement: {model_path}...")
    vec_env = DummyVecEnv([lambda: RobotArmEnv(render_mode="human", max_steps=300)])
    
    if os.path.exists(f"{model_path}_vecnorm.pkl"):
        env = VecNormalize.load(f"{model_path}_vecnorm.pkl", vec_env)
        env.training = False
        env.norm_reward = False
    else:
        env = vec_env
        
    env.env_method('set_difficulty', level)
    model = PPO.load(model_path, env=env)
    
    successes = 0
    total_rewards = []
    max_heights = []
    
    for ep in range(episodes):
        obs = env.reset()
        done = False
        episode_reward = 0
        steps = 0
        print(f"\n📺 Episode {ep + 1}/{episodes}")
        
        while not done:
            action, _ = model.predict(obs, deterministic=not stochastic)
            obs, rewards, dones, infos = env.step(action)
            reward = rewards[0]
            done = dones[0]
            info = infos[0]
            episode_reward += reward
            steps += 1
            time.sleep(0.01)
        
        total_rewards.append(episode_reward)
        max_height = env.get_attr('max_object_height_reached')[0]
        max_heights.append(max_height)
        
        if info.get('success', False):
            successes += 1
            print(f"   ✅ Succès! Reward: {episode_reward:.1f}, Max height: {max_height*100:.1f}cm")
        else:
            print(f"   ❌ Échec. Reward: {episode_reward:.1f}, Max height: {max_height*100:.1f}cm")
    
    avg_height = np.mean(max_heights) * 100
    print(f"\n📊 Résultats: {successes}/{episodes} ({100*successes/episodes:.1f}%)")
    print(f"   Hauteur moyenne: {avg_height:.1f}cm")
    env.close()


def demo_untrained(episodes: int = 3):
    """Demo."""
    print("🎮 DÉMO")
    env = RobotArmEnv(render_mode="human", max_steps=200)
    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action = env.action_space.sample()
            _, _, done, _, _ = env.step(action)
            time.sleep(0.05)
    env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--demo', action='store_true')
    parser.add_argument('--steps', type=int, default=200000)
    parser.add_argument('--episodes', type=int, default=10)
    parser.add_argument('--model', type=str, default='models/robot_arm_ultra')
    parser.add_argument('--stochastic', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--level', type=int, default=1)
    args = parser.parse_args()
    
    if args.train:
        train_agent(total_timesteps=args.steps, save_path=args.model, 
                   resume=args.resume, render=args.render)
    elif args.test:
        test_agent(model_path=args.model, episodes=args.episodes, 
                  stochastic=args.stochastic, level=args.level)
    elif args.demo:
        demo_untrained(episodes=args.episodes)
    else:
        print("Usage: python IA_ULTRA_FIXED.py --train | --test | --demo")


if __name__ == "__main__":
    main()
