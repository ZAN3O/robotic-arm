import zmq

HOST = "10.0.0.80"
PORT = 5555

# Angles de base (ex: ton exemple)
base_angles = [90, 45, 120, 90]

# --- OPTION A: même offset pour tous les servos ---
offset = +100
  # mets -10 pour baisser, +5 pour augmenter, etc.
target_angles = [a + offset for a in base_angles]

# --- OPTION B: offsets différents par servo (décommente pour utiliser) ---
# offsets = [0, +5, -10, +2]
# assert len(offsets) == len(base_angles)
# target_angles = [a + da for a, da in zip(base_angles, offsets)]

# (Optionnel) clamp 0..180 si tes servos sont dans cette plage
target_angles = [max(0, min(180, a)) for a in target_angles]

ctx = zmq.Context()
s = ctx.socket(zmq.REQ)
s.connect(f"tcp://{HOST}:{PORT}")

s.send_json({
    "target_angles": target_angles,
    "gripper": 70,
    "speed": 40
})

print("Sent:", target_angles)
print("Recv:", s.recv_json())
