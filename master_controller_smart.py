import pybullet as p
import pybullet_data
import time
import numpy as np
import math
from ikpy.chain import Chain
from ikpy.link import URDFLink, OriginLink
from voice_control import VoiceController, CommandAction

# ==============================================================================
# 1. PARAMÈTRES PHYSIQUES
# ==============================================================================
URDF_PATH = "arduino_arm.urdf"
REAL_TIME = True
FPS = 240

BASE_HEIGHT = 0.11      
HUMERUS_LENGTH = 0.13   
ULNA_LENGTH = 0.13      
GRIPPER_LENGTH = 0.12

# ==============================================================================
# 2. CHAÎNE CINÉMATIQUE
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
        self.active_joints = [0, 1, 2, 3] 
        self.gripper_indices = [5, 6] 
        
        for i in range(p.getNumJoints(self.id)):
            p.changeDynamics(self.id, i, lateralFriction=2.0, spinningFriction=0.1, jointDamping=0.1)

    def solve_ik(self, target_xyz):
        ik_sol = self.chain.inverse_kinematics(target_position=target_xyz)
        return ik_sol[1:5]

    def move_smooth(self, target_xyz, gripper_open=True, duration=1.5):
        print(f"   📍 Vers: [{target_xyz[0]:.3f}, {target_xyz[1]:.3f}, {target_xyz[2]:.3f}]")
        target_angles = self.solve_ik(target_xyz)
        
        current_angles = []
        for j in self.active_joints:
            current_angles.append(p.getJointState(self.id, j)[0])
        start_angles = np.array(current_angles)
        
        steps = int(duration * FPS)
        for i in range(steps):
            t = i / steps
            cmd = start_angles + (target_angles - start_angles) * t
            
            for idx, joint_idx in enumerate(self.active_joints):
                p.setJointMotorControl2(self.id, joint_idx, p.POSITION_CONTROL, 
                                      targetPosition=cmd[idx], force=100)
            
            if gripper_open:
                grip_pos = 0.0
            else:
                grip_pos = 0.5 
            
            p.setJointMotorControl2(self.id, self.gripper_indices[0], p.POSITION_CONTROL, targetPosition=grip_pos, force=50)
            p.setJointMotorControl2(self.id, self.gripper_indices[1], p.POSITION_CONTROL, targetPosition=-grip_pos, force=50)

            p.stepSimulation()
            if REAL_TIME: time.sleep(1./FPS)
            
        for _ in range(10): 
            p.stepSimulation()
            time.sleep(0.005)

    def home(self):
        self.move_smooth([0.15, 0, 0.20], gripper_open=True, duration=1.0)

# ==============================================================================
# 4. ORACLE & LOGIQUE SMART
# ==============================================================================

def simulated_perception(target_color_name, memory):
    target_rgba = None
    # 🔧 FIX: Accept English keys from VoiceController
    if target_color_name == "red": target_rgba = [1, 0, 0, 1]
    elif target_color_name == "green": target_rgba = [0, 1, 0, 1]
    elif target_color_name == "blue": target_rgba = [0, 0, 1, 1]
    
    if target_rgba is None: return None
    
    print(f"👁️ Oracle recherche : {target_color_name}...")
    for obj_id, props in memory.items():
        if props['color'] == target_rgba:
            pos, _ = p.getBasePositionAndOrientation(obj_id)
            return pos
    return None

def run():
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    p.resetDebugVisualizerCamera(0.7, 0, -40, [0.25, 0, 0])
    
    p.loadURDF("plane.urdf")
    
    # Table
    p.createMultiBody(0, 
                      p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.02]),
                      p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.4, 0.02], rgbaColor=[0.6, 0.4, 0.2, 1]), 
                      [0.5, 0, -0.02])

    robot = RoboticArm(URDF_PATH)
    robot.home()
    
    # --- ETAPE 1 : ENVIRONNEMENT MULTI-OBJETS SMART ---
    print("📦 Génération de la scène...")
    objects_memory = {}
    # 🔧 FIX: Use English keys to match VoiceController intent
    colors = {
        "red": [1, 0, 0, 1],
        "green": [0, 1, 0, 1],
        "blue": [0, 0, 1, 1]
    }
    
    existing_positions = []
    
    # SMART SPAWNER
    for name, rgba in colors.items():
        for _ in range(50):
            x = np.random.uniform(0.15, 0.25)
            y = np.random.uniform(-0.15, 0.15)
            
            collision = False
            for ex, ey in existing_positions:
                if np.linalg.norm([x-ex, y-ey]) < 0.05: 
                    collision = True
                    break
            
            if not collision:
                existing_positions.append((x, y))
                
                # 🧠 LEAD TECH TRICK: Orientation Radiale
                # Le robot n'a pas de poignet rotatif (Wrist Roll).
                # Pour attraper un cube sur le côté, le cube DOIT faire face au robot.
                # Angle optimal = arctan2(y, x)
                optimal_yaw = math.atan2(y, x)
                orientation = p.getQuaternionFromEuler([0, 0, optimal_yaw])
                
                cube_pos = [x, y, 0.015]
                vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.012, 0.012, 0.012], rgbaColor=rgba)
                col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.012, 0.012, 0.012])
                obj_id = p.createMultiBody(0.02, col, vis, cube_pos, orientation)
                p.changeDynamics(obj_id, -1, lateralFriction=2.0)
                
                objects_memory[obj_id] = {'color': rgba, 'name': name}
                print(f"   + Cube {name.upper()} placé en {x:.2f}, {y:.2f} (Yaw: {math.degrees(optimal_yaw):.1f}°)")
                break
    
    print("\n✅ SYSTÈME 'SMART PICK' PRÊT.")
    
    # --- CALLBACK VOCAL ---
    def process_voice_command(intent):
        """Callback exécuté quand une commande vocale est comprise"""
        print(f"\n🗣️ ORDRE REÇU : {intent.raw_text}")
        
        # 🔧 FIX: Auto-Sleep after command to prevent "Always On" annoyance
        # The user must say "Bonjour bras" again for the next command.
        vc.is_active = False 

        if intent.action == CommandAction.STOP:
            print("🛑 ARRÊT D'URGENCE INTERNE")
            return
            
        # 🔧 FIX: Check for English keys from VoiceController (red, green, blue)
        if intent.target_color in ["red", "green", "blue"]:
            print(f"🤖 Intention détectée : CIBLE = {intent.target_color.upper()}")
            
            target_pos = simulated_perception(intent.target_color, objects_memory)
            
            if target_pos:
                print(f"🎯 Cible verrouillée : x={target_pos[0]:.3f} y={target_pos[1]:.3f}")
                
                # Séquence Pick & Place Améliorée
                robot.move_smooth([target_pos[0], target_pos[1], 0.15], gripper_open=True)
                
                print("⏬ Descente...")
                robot.move_smooth([target_pos[0], target_pos[1], 0.005], gripper_open=True, duration=1.0)
                
                print("🦞 Saisie...")
                robot.move_smooth([target_pos[0], target_pos[1], 0.005], gripper_open=False, duration=0.5)
                time.sleep(0.2)
                
                print("🏋️ Levage...")
                robot.move_smooth([target_pos[0], target_pos[1], 0.20], gripper_open=False, duration=1.0)
                
                print("📦 Livraison...")
                robot.move_smooth([0.0, -0.25, 0.15], gripper_open=False)
                robot.move_smooth([0.0, -0.25, 0.15], gripper_open=True)
                
                robot.home()
            else:
                print(f"❌ Erreur : Je ne vois pas de cube {intent.target_color}.")
        else:
            print("❓ Je n'ai pas compris quelle couleur prendre (rouge, vert, bleu ?)")

    # --- MAIN LOOP AVEC VOICE CONTROL ---
    try:
        # Initialisation du contrôleur vocal
        vc = VoiceController(language="fr-FR")
        
        if not vc.recognizer:
            print("⚠️ Erreur micro/reconnaissance. Passage en mode TEXTE.")
            while True:
                cmd = input("⌨️ Commande (ex: 'rouge') : ")
                # Mock intent for text fallback
                class MockIntent: pass
                intent = MockIntent()
                intent.raw_text = cmd
                intent.action = CommandAction.PICK
                intent.target_color = cmd.lower()
                process_voice_command(intent)
        else:
            # Lancement boucle vocale infinie
            vc.run_continuous(process_voice_command)
            
    except KeyboardInterrupt:
        print("Fin du programme.")

if __name__ == "__main__":
    run()
