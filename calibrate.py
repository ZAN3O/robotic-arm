import pybullet as p
import pybullet_data
import time
import numpy as np
import os

def calibrate():
    # Connect to Physics Server
    p.connect(p.GUI)
    p.setAdditionalSearchPath(pybullet_data.getDataPath())
    p.setGravity(0, 0, -9.81)
    
    # Load Plane and Table
    p.loadURDF("plane.urdf")
    table_id = p.createMultiBody(
        baseMass=0,
        baseCollisionShapeIndex=p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.4, 0.25, 0.005]),
        baseVisualShapeIndex=p.createVisualShape(p.GEOM_BOX, halfExtents=[0.4, 0.25, 0.005], rgbaColor=[0.6, 0.4, 0.2, 1]),
        basePosition=[0.17, 0, 0]
    )
    
    # Load Robot
    urdf_path = "arduino_arm.urdf" # Ensure this is in the same dir or provide full path
    if not os.path.exists(urdf_path):
        urdf_path = os.path.join(os.path.dirname(__file__), "arduino_arm.urdf")
        
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0], useFixedBase=True)
    
    # Identify joints
    joint_indices = []
    for i in range(p.getNumJoints(robot_id)):
        info = p.getJointInfo(robot_id, i)
        if info[2] == p.JOINT_REVOLUTE and 'gripper' not in info[1].decode('utf-8').lower():
            joint_indices.append(i)
            
    # Add target object (Green Cube)
    target_pos = [0.15, 0.05, 0.02]
    vis_shape = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.0125]*3, rgbaColor=[0.1, 0.9, 0.1, 1])
    target_id = p.createMultiBody(baseVisualShapeIndex=vis_shape, basePosition=target_pos)
    
    # Sliders for joints
    sliders = []
    defaults = [0, 20, -30, -30] # My current guess for Level 1
    
    names = ["Base", "Shoulder", "Elbow", "Wrist"]
    for i, idx in enumerate(joint_indices):
        sid = p.addUserDebugParameter(names[i], -150, 150, defaults[i])
        sliders.append(sid)
        
    print("Adjust sliders to position the gripper DIRECTLY above the green cube.")
    print("Press Ctrl+C to stop.")
    
    while True:
        # Read sliders
        angles = []
        for i, sid in enumerate(sliders):
            angle = p.readUserDebugParameter(sid)
            angles.append(angle)
            
            # Apply to robot
            p.resetJointState(robot_id, joint_indices[i], np.radians(angle))
            
        p.stepSimulation()
        time.sleep(0.01)

if __name__ == "__main__":
    calibrate()
