"""
requete.py - Quick ZMQ test to send angles to Pi → Arduino

Usage: python requete.py
"""

import zmq

HOST = "10.0.0.80"
PORT = 5555

# 5 servo angles: base(pin8), shoulder(pin5), elbow(pin7), wrist_roll(pin6), wrist_yaw(pin4)
target_angles = [90, 45, 120, 90, 90]

# Gripper (0=fermé, 180=ouvert, 70=défaut)
gripper = 70

# Speed (1=lent, 100=rapide)
speed = 40

ctx = zmq.Context()
s = ctx.socket(zmq.REQ)
s.connect(f"tcp://{HOST}:{PORT}")

cmd = {
    "target_angles": target_angles,
    "gripper": gripper,
    "speed": speed
}

print(f"📡 Envoi → {HOST}:{PORT}")
print(f"   Angles: {target_angles}")
print(f"   Gripper: {gripper}, Speed: {speed}")

s.send_json(cmd)
response = s.recv_json()

print(f"📥 Réponse: {response}")
