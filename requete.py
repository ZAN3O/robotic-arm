"""
requete.py - Envoi de commandes au bras robotique via ZMQ (Pi → Arduino)

Usage:
    python requete.py              → envoie la position IDLE par défaut
    python requete.py idle         → position repos
    python requete.py custom       → position custom (modifie ci-dessous)
"""

import zmq
import sys

HOST = "10.0.0.80"
PORT = 5555

# ============================================================================
# POSITIONS PRÉDÉFINIES
# ============================================================================
# Format: [base(pin8), shoulder(pin5), elbow(pin7), wrist_roll(pin6), wrist_yaw(pin4)]
#         + gripper(pin9), speed

POSITIONS = {
    "idle": {
        "angles": [90, 90, 90, 90, 90],
        "gripper": 70,
        "speed": 30,
        "description": "🏠 Position repos (idle)"
    },
    "up": {
        "angles": [90, 45, 45, 90, 90],
        "gripper": 70,
        "speed": 30,
        "description": "⬆️ Bras levé"
    },
    "open": {
        "angles": [90, 90, 90, 90, 90],
        "gripper": 30,
        "speed": 40,
        "description": "✋ Pince ouverte"
    },
    "close": {
        "angles": [90, 90, 90, 90, 90],
        "gripper": 150,
        "speed": 40,
        "description": "🤏 Pince fermée"
    },
    "custom": {
        "angles": [90, 45, 120, 90, 90],
        "gripper": 70,
        "speed": 40,
        "description": "🔧 Position custom"
    },
}

# ============================================================================
# ENVOI
# ============================================================================
# Choix de la position : argument ou "idle" par défaut
mode = sys.argv[1] if len(sys.argv) > 1 else "idle"

if mode not in POSITIONS:
    print(f"❌ Position '{mode}' inconnue. Disponibles: {', '.join(POSITIONS.keys())}")
    sys.exit(1)

pos = POSITIONS[mode]
target_angles = pos["angles"]
gripper = pos["gripper"]
speed = pos["speed"]

print(f"{pos['description']}")
print(f"   Angles: {target_angles}")
print(f"   Gripper: {gripper}, Speed: {speed}")

ctx = zmq.Context()
s = ctx.socket(zmq.REQ)
s.connect(f"tcp://{HOST}:{PORT}")

s.send_json({
    "target_angles": target_angles,
    "gripper": gripper,
    "speed": speed
})

print(f"📡 Envoyé → {HOST}:{PORT}")
response = s.recv_json()
print(f"📥 Réponse: {response}")
