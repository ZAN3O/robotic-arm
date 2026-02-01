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
    
    def __init__(self, render_mode: str = None, max_steps: int = 200):
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
        
        # Action scaling
        self.action_scale = 5.0  # Degrees per action step
        
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
        self.gripper_state = 0.5  # 0=closed, 1=open
        self.target_position = np.zeros(3)
        self.object_grasped = False
        self.object_lifted = False
        
        # Success tracking
        self.success_count = 0
        self.episode_count = 0
        
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
            basePosition=[0.3, 0, 0]
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
    
    def _create_target_object(self):
        """Create a random target object."""
        if self.target_object_id is not None:
            p.removeBody(self.target_object_id)
        
        # Random position on table
        x = np.random.uniform(0.15, 0.35)
        y = np.random.uniform(-0.15, 0.15)
        z = 0.02  # On table
        
        self.target_position = np.array([x, y, z])
        
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
        """Get end effector position."""
        # Get the last link state
        if len(self.gripper_indices) > 0:
            state = p.getLinkState(self.robot_id, self.gripper_indices[0])
        else:
            state = p.getLinkState(self.robot_id, self.joint_indices[-1])
        return np.array(state[0])
    
    def _is_object_grasped(self) -> bool:
        """Check if object is being held."""
        if self.target_object_id is None:
            return False
        
        # Check contacts with both gripper fingers
        left_contact = False
        right_contact = False
        
        for idx in self.gripper_indices:
            contacts = p.getContactPoints(
                bodyA=self.robot_id,
                linkIndexA=idx,
                bodyB=self.target_object_id
            )
            if contacts:
                if 'left' in p.getJointInfo(self.robot_id, idx)[1].decode().lower():
                    left_contact = True
                else:
                    right_contact = True
        
        return left_contact and right_contact
    
    def _compute_reward(self) -> Tuple[float, bool, Dict]:
        """Compute reward with approach-from-above strategy."""
        reward = 0.0
        terminated = False
        info = {}
        
        ee_pos = self._get_end_effector_position()
        
        if self.target_object_id is not None:
            obj_pos, _ = p.getBasePositionAndOrientation(self.target_object_id)
            obj_pos = np.array(obj_pos)
        else:
            obj_pos = self.target_position
        
        # Track object movement (detect bumping)
        if not hasattr(self, '_initial_obj_pos') or self._initial_obj_pos is None:
            self._initial_obj_pos = obj_pos.copy()
        
        obj_moved = np.linalg.norm(obj_pos[:2] - self._initial_obj_pos[:2])  # XY movement
        
        # Distance calculations
        xy_distance = np.linalg.norm(ee_pos[:2] - obj_pos[:2])  # Horizontal distance
        z_distance = ee_pos[2] - obj_pos[2]  # Height above object
        distance_3d = np.linalg.norm(ee_pos - obj_pos)
        
        # Track previous distance for progress
        if self._prev_distance is None:
            self._prev_distance = xy_distance
        
        # === PHASE 1: Get above object (XY alignment) ===
        xy_improvement = self._prev_distance - xy_distance
        reward += xy_improvement * 15.0  # Strong XY approach signal
        self._prev_distance = xy_distance
        
        # Encourage staying above object level
        if z_distance > 0.03:  # Above object
            reward += 0.5  # Small bonus for being above
        elif z_distance < 0 and xy_distance > 0.04:  # Below object but not aligned
            reward -= 1.0  # Penalty for diving too early
        
        # === PHASE 2: XY aligned, now descend ===
        if xy_distance < 0.04:
            reward += 3.0  # Bonus for XY alignment
            
            # Encourage descending when aligned
            if z_distance > 0.02:  # Still above
                reward -= z_distance * 2.0  # Encourage lowering
            else:
                reward += 5.0  # At grasp height!
                
                # === PHASE 3: Close gripper ===
                if self.gripper_state < 0.3:  # Gripper closing
                    reward += 8.0  # Big bonus for closing gripper at right spot
        
        # === COLLISION PENALTY ===
        if obj_moved > 0.02:  # Object moved significantly
            reward -= 20.0  # Heavy penalty for bumping!
            info['bumped'] = True
        
        # === GRASPING ===
        is_grasped = self._is_object_grasped()
        if is_grasped and not self.object_grasped:
            reward += 150.0  # HUGE bonus for grasp
            self.object_grasped = True
            info['grasped'] = True
            print("🎯 GRASP!")
        
        if self.object_grasped:
            reward += 3.0  # Hold bonus
        
        # === LIFTING ===
        if self.object_grasped and obj_pos[2] > 0.06:
            if not self.object_lifted:
                reward += 300.0  # MASSIVE success
                self.object_lifted = True
                self.success_count += 1
                info['lifted'] = True
                print("🚀 LIFT SUCCESS!")
                terminated = True
        
        # Dropping penalty
        if self.object_grasped and not is_grasped:
            reward -= 30.0
            self.object_grasped = False
        
        # Time penalty
        reward -= 0.05
        
        if self.current_step >= self.max_steps:
            terminated = True
        
        info['distance'] = distance_3d
        info['xy_dist'] = xy_distance
        info['object_height'] = obj_pos[2]
        info['success'] = self.object_lifted
        
        return reward, terminated, info
    
    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        """Reset environment."""
        super().reset(seed=seed)
        
        self.episode_count += 1
        self.current_step = 0
        self.object_grasped = False
        self.object_lifted = False
        self._prev_distance = None
        self._initial_obj_pos = None  # Reset object tracking
        
        # Reset robot position
        self.current_joint_angles = np.zeros(4)
        for i, idx in enumerate(self.joint_indices):
            p.resetJointState(self.robot_id, idx, 0)
        
        # Open gripper
        self.gripper_state = 1.0
        self._set_gripper(1.0)
        
        # Create new target object
        self._create_target_object()
        
        # Step simulation to settle
        for _ in range(50):
            p.stepSimulation()
        
        obs = self._get_observation()
        info = {'success_rate': self.success_count / max(1, self.episode_count)}
        
        return obs, info
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute action."""
        self.current_step += 1
        
        # Parse action
        joint_deltas = action[:4] * self.action_scale  # Scale to degrees
        gripper_cmd = action[4]
        
        # Update joint angles
        self.current_joint_angles += joint_deltas
        self.current_joint_angles = np.clip(
            self.current_joint_angles,
            self.joint_limits_low,
            self.joint_limits_high
        )
        
        # Apply to robot
        for i, idx in enumerate(self.joint_indices):
            target_rad = np.radians(self.current_joint_angles[i])
            p.setJointMotorControl2(
                self.robot_id, idx,
                p.POSITION_CONTROL,
                targetPosition=target_rad,
                force=100
            )
        
        # Update gripper
        self.gripper_state = (gripper_cmd + 1) / 2  # Convert [-1,1] to [0,1]
        self._set_gripper(self.gripper_state)
        
        # Step simulation
        for _ in range(10):
            p.stepSimulation()
            if self.render_mode == "human":
                time.sleep(1./240.)
        
        # Get observation and reward
        obs = self._get_observation()
        reward, terminated, info = self._compute_reward()
        truncated = False
        
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
                force=10
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


def train_agent(total_timesteps: int = 100000, save_path: str = "models/robot_arm_ai"):
    """Train the RL agent."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError:
        print("❌ stable-baselines3 non installé!")
        print("   Run: pip install stable-baselines3[extra]")
        return None
    
    print("=" * 60)
    print("🧠 ENTRAÎNEMENT IA - Bras Robotique")
    print("=" * 60)
    print(f"   Algorithme: PPO (Proximal Policy Optimization)")
    print(f"   Steps: {total_timesteps:,}")
    print(f"   Save path: {save_path}")
    print("=" * 60)
    
    # Create directories
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Create environment
    env = DummyVecEnv([lambda: RobotArmEnv(render_mode=None, max_steps=200)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True)
    
    # Create evaluation environment
    eval_env = DummyVecEnv([lambda: RobotArmEnv(render_mode=None, max_steps=200)])
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False)
    
    # Callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path="./models/checkpoints/",
        name_prefix="robot_arm"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best/",
        log_path="./logs/",
        eval_freq=5000,
        n_eval_episodes=10,
        deterministic=True
    )
    
    # Create model
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
        tensorboard_log="./logs/tensorboard/"
    )
    
    print("\n🚀 Début de l'entraînement...\n")
    
    # Train
    model.learn(
        total_timesteps=total_timesteps,
        callback=[checkpoint_callback, eval_callback],
        progress_bar=True
    )
    
    # Save final model
    model.save(save_path)
    env.save(f"{save_path}_vecnorm.pkl")
    
    print(f"\n✅ Entraînement terminé!")
    print(f"   Modèle sauvegardé: {save_path}.zip")
    
    env.close()
    eval_env.close()
    
    return model


def test_agent(model_path: str = "models/robot_arm_ai", episodes: int = 10):
    """Test a trained agent with proper normalization."""
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError:
        print("❌ stable-baselines3 non installé!")
        return
    
    print("=" * 60)
    print("🧪 TEST IA - Bras Robotique")
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
        vec_env = DummyVecEnv([lambda: RobotArmEnv(render_mode="human", max_steps=200)])
        env = VecNormalize.load(vecnorm_path, vec_env)
        env.training = False  # Don't update stats during test
        env.norm_reward = False  # Don't normalize rewards
        use_vec_env = True
    else:
        print(f"   ⚠️ Pas de normalisation trouvée, utilisation raw")
        env = RobotArmEnv(render_mode="human", max_steps=200)
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
            action, _ = model.predict(obs, deterministic=True)
            
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
    
    args = parser.parse_args()
    
    if args.train:
        train_agent(total_timesteps=args.steps, save_path=args.model)
    elif args.test:
        test_agent(model_path=args.model, episodes=args.episodes)
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
