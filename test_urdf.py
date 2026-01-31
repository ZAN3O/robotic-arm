import pybullet as p
import pybullet_data
import time

# 1. Démarrer la simulation
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)

# 2. Charger le sol
planeId = p.loadURDF("plane.urdf")

# 3. Charger TON robot
# startPos = [0, 0, 0]
startOrientation = p.getQuaternionFromEuler([0, 0, 0])
robotId = p.loadURDF("arduino_arm.urdf", [0, 0, 0], startOrientation, useFixedBase=1)

# 4. Ajout de curseurs pour bouger les axes manuellement
sliders = []
num_joints = p.getNumJoints(robotId)
for i in range(num_joints):
    # On ignore le joint "tip" qui est fixe
    info = p.getJointInfo(robotId, i)
    joint_name = info[1].decode("utf-8")
    if "tip" not in joint_name:
        sid = p.addUserDebugParameter(joint_name, -1.57, 1.57, 0)
        sliders.append(sid)

# 5. Boucle principale
while True:
    # Lire les curseurs et appliquer aux moteurs
    for i, slider in enumerate(sliders):
        target_pos = p.readUserDebugParameter(slider)
        p.setJointMotorControl2(robotId, i, p.POSITION_CONTROL, targetPosition=target_pos)
    
    p.stepSimulation()
    time.sleep(1./240.)