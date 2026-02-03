import pybullet as p
import pybullet_data
import time
import numpy as np
from ikpy.chain import Chain
from ikpy.link import URDFLink, OriginLink

# ==============================================================================
# 1. PARAMÈTRES PHYSIQUES CORRIGÉS
# ==============================================================================
URDF_PATH = "arduino_arm.urdf"
REAL_TIME = True
FPS = 240

# CORRECTION HAUTEUR: 0.05m (Base) + 0.06m (Joint2) = 0.11m
BASE_HEIGHT = 0.11      
HUMERUS_LENGTH = 0.13   
ULNA_LENGTH = 0.13      
GRIPPER_LENGTH = 0.12   

# ==============================================================================
# 2. CHAÎNE CINÉMATIQUE MANUELLE
# ==============================================================================
def create_manual_chain():
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
# 3. CLASSE ROBOT
# ==============================================================================
class RoboticArm:
    def __init__(self, urdf_path, offset=[0, 0, 0]):
        self.id = p.loadURDF(urdf_path, offset, useFixedBase=1)
        self.chain = create_manual_chain()
        
        # --- CORRECTION INDICES MOTEURS ---
        # 0: Base, 1: Shoulder, 2: Elbow, 3: Wrist
        self.active_joints = [0, 1, 2, 3] 
        
        # CORRECTION CRITIQUE PINCE: 
        # Joint 4 est FIXE (Gripper Base). Les doigts sont 5 et 6.
        self.gripper_indices = [5, 6] 
        
        # SUPER GRIP (Friction élevée pour ne pas glisser)
        for i in range(p.getNumJoints(self.id)):
            p.changeDynamics(self.id, i, lateralFriction=2.0, spinningFriction=0.1, jointDamping=0.1)

    def solve_ik(self, target_xyz):
        # On calcule les angles pour atteindre la cible
        ik_sol = self.chain.inverse_kinematics(target_position=target_xyz)
        return ik_sol[1:5]

    def move_smooth(self, target_xyz, gripper_open=True, duration=1.5):
        print(f"   📍 Vers: [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]")
        
        target_angles = self.solve_ik(target_xyz)
        
        # Lecture état actuel
        current_angles = []
        for j in self.active_joints:
            current_angles.append(p.getJointState(self.id, j)[0])
        start_angles = np.array(current_angles)
        
        steps = int(duration * FPS)
        for i in range(steps):
            t = i / steps
            # Interpolation linéaire
            cmd = start_angles + (target_angles - start_angles) * t
            
            # Application Moteurs Bras (Force élevée)
            for idx, joint_idx in enumerate(self.active_joints):
                p.setJointMotorControl2(self.id, joint_idx, p.POSITION_CONTROL, 
                                      targetPosition=cmd[idx], force=100)
            
            # Gestion Pince CORRIGÉE
            # 0.0 = Ouvert (Parallèle)
            # 0.5 = Fermé (Serré)
            if gripper_open:
                grip_pos = 0.0
            else:
                grip_pos = 0.5 # Force la fermeture
            
            # Application aux VRAIS joints de la pince (5 et 6)
            # Doigt Gauche (5) : Positif pour fermer
            p.setJointMotorControl2(self.id, self.gripper_indices[0], p.POSITION_CONTROL, targetPosition=grip_pos, force=50)
            # Doigt Droit (6) : Négatif pour fermer
            p.setJointMotorControl2(self.id, self.gripper_indices[1], p.POSITION_CONTROL, targetPosition=-grip_pos, force=50)

            p.stepSimulation()
            if REAL_TIME: time.sleep(1./FPS)
            
        # Stabilisation
        for _ in range(10): 
            p.stepSimulation()
            time.sleep(0.005)

# ==============================================================================
# 4. MAIN
# ==============================================================================
def run():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(0.8, 90, -30, [0.2, 0, 0.1])
    
    p.loadURDF("plane.urdf")
    
    # Table (X=0.5)
    p.createMultiBody(0, 
                      p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.02]),
                      p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.02], rgbaColor=[0.6, 0.4, 0.2, 1]), 
                      [0.5, 0, -0.02])

    robot = RoboticArm(URDF_PATH)
    
    print("🚀 DÉMARRAGE MODE DEEP GRASP CORRIGÉ")
    
    while True:
        # 1. Spawn Cube
        cube_pos = [np.random.uniform(0.18, 0.22), np.random.uniform(-0.1, 0.1), 0.02]
        vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015], rgbaColor=[1, 0, 0, 1])
        col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.015, 0.015, 0.015])
        cube_id = p.createMultiBody(0.02, col, vis, cube_pos)
        p.changeDynamics(cube_id, -1, lateralFriction=2.0) # Le cube ne glisse pas

        print(f"\n📦 Cube: {cube_pos}")
        time.sleep(0.5)
        
        # 2. Approche Haute
        robot.move_smooth([cube_pos[0], cube_pos[1], 0.15], gripper_open=True, duration=1.0)
        
        # 3. PLONGÉE PROFONDE (Avec Base corrigée, 0.005mm touche presque le sol)
        print("⏬ Plongée Profonde...")
        robot.move_smooth([cube_pos[0], cube_pos[1], 0.005], gripper_open=True, duration=1.5)
        
        # 4. Saisie
        print("🦞 GRIP !")
        robot.move_smooth([cube_pos[0], cube_pos[1], 0.005], gripper_open=False, duration=0.5)
        time.sleep(0.2)
        
        # 5. Levage
        print("🏋️ Levage...")
        robot.move_smooth([cube_pos[0], cube_pos[1], 0.20], gripper_open=False, duration=1.0)
        
        # 6. Drop
        robot.move_smooth([0.15, 0.20, 0.15], gripper_open=False)
        robot.move_smooth([0.15, 0.20, 0.15], gripper_open=True)
        
        p.removeBody(cube_id)

if __name__ == "__main__":
    run()