import pybullet as p
import pybullet_data
import time
import numpy as np
import os
import sys

def calibrate():
    # Connect to Physics Server
    try:
        pid = p.connect(p.GUI)
    except Exception as e:
        print(f"Failed to connect: {e}")
        return

    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    p.resetSimulation() # Clear scene
    p.setGravity(0, 0, -9.81)
    
    # Load Plane and Table (FIXED POSITION)
    p.loadURDF("plane.urdf")
    
    # Same Table as IA_FIXED.py (X=0.3, Z=-0.05)
    len_x = 0.4
    width_y = 0.4
    thick = 0.01
    
    table_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[len_x, width_y, thick])
    table_vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[len_x, width_y, thick], rgbaColor=[0.6, 0.4, 0.2, 1])
    
    # FIX 1: Table Position (Matches IA_FIXED.py)
    table_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=table_col,
        baseVisualShapeIndex=table_vis,
        basePosition=[0.3, 0, -0.05]
    )
    
    # Load Robot
    urdf_path = "arduino_arm.urdf" 
    if not os.path.exists(urdf_path):
        urdf_path = os.path.join(os.path.dirname(__file__), "arduino_arm.urdf")
        
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
    
    # Identify joints
    joint_indices = []
    gripper_indices = []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        jname = info[1].decode('utf-8')
        if info[2] == p.JOINT_REVOLUTE:
            if 'gripper' in jname.lower():
                gripper_indices.append(i)
            else:
                joint_indices.append(i)
            
    # Add target object (FIXED Level 1 Position)
    # [0.18, 0.0, 0.02] (Closer = Easier)
    # Adjusted to match new table height? Table is at -0.05.
    # If Level 1 is X=0.18...
    # Let's put a visual marker for the "Target Zone" of Level 1
    
    target_pos = [0.18, 0.0, -0.035] # 1.5cm above table (-0.05 + 0.015)
    
    vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.0125]*3, rgbaColor=[0.1, 0.9, 0.1, 1])
    col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.0125]*3)
    
    target_id = p.createMultiBody(
        baseMass=0.01,
        baseCollisionShapeIndex=col_shape,
        baseVisualShapeIndex=vis_shape,
        basePosition=target_pos
    )
    
    # Sliders for joints
    # START AT POSE (EZ Mode): [0.0, 25.3, 112.1, 25.3]
    sliders = []
    defaults = [0.0, 25.3, 112.1, 25.3] 
    
    names = ["Base", "Shoulder", "Elbow", "Wrist"]
    for i, idx in enumerate(joint_indices):
        sid = p.addUserDebugParameter(names[i], -150, 150, defaults[i])
        sliders.append(sid)
        
    # Slider for Gripper
    gripper_slider = p.addUserDebugParameter("Gripper (0=Closed, 1=Open)", 0, 1, 1) 
        
    print("="*60)
    print("🛠️ MODE CALIBRAGE (TELEPORT/GHOST)")
    print(f"🎯 CUBE FIXE : {target_pos}")
    print(f"🤖 POSE DEPART : {defaults}")
    print("="*60)
    print("Utilise les sliders pour trouver la bonne position.")
    print("Si la fenêtre se ferme, le script s'arrête.")
    
    try:
        while p.isConnected():
            # Read Joint sliders
            for i, sid in enumerate(sliders):
                angle_deg = p.readUserDebugParameter(sid)
                p.resetJointState(robot_id, joint_indices[i], np.radians(angle_deg))
                
            # Read Gripper slider
            grip_val = p.readUserDebugParameter(gripper_slider)
            
            # FIX 2: Correct Gripper Logic (Matches IA.py Fix)
            # URDF LIMITS: Left (0 to 0.5), Right (-0.5 to 0)
            # 0=Closed (val=0), 1=Open (val=1)
            
            # Closed (val=0) -> Left=0.5, Right=-0.5
            # Open (val=1)   -> Left=0.0, Right=0.0
            
            # Logic: Target = (1 - val) * Limit
            left_target = (1 - grip_val) * 0.5   # 0 -> 0.5
            right_target = (1 - grip_val) * -0.5 # 0 -> -0.5
            
            p.resetJointState(robot_id, gripper_indices[0], left_target) # Left
            p.resetJointState(robot_id, gripper_indices[1], right_target)# Right
                
            p.stepSimulation()
            time.sleep(0.01)
            
    except KeyboardInterrupt:
        print("Arrêt par utilisateur.")
    except Exception as e:
        print(f"Simulation arrêtée: {e}")
    finally:
        if p.isConnected():
            p.disconnect()

if __name__ == "__main__":
    calibrate()
