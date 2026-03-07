"""
real_arm_controller.py - Main PC Orchestrator for Real Robotic Arm

Combines:
  - Camera (top-down fixed view)
  - OWL-ViT object detection
  - Homography (pixel → world coordinates)
  - Inverse Kinematics (world → servo angles)
  - Voice control ("Bonjour bras, attrape le ciseau")
  - ZMQ communication to Raspberry Pi

Usage:
    1. First run:  python calibrate_real.py   (one-time calibration)
    2. On Pi run:   python3 pi_relay.py
    3. Then run:    python real_arm_controller.py
    
    python real_arm_controller.py --pi-host 10.0.0.80 --camera-id 0
"""

import cv2
import torch
import zmq
import json
import time
import threading
import numpy as np
import argparse
import os
from PIL import Image
from transformers import OwlViTProcessor, OwlViTForObjectDetection

from vision import CameraCapture, HomographyTransformer
from kinematics import create_kinematic_chain, get_servo_angles, check_reachability
from voice_control import VoiceController, CommandAction


# ============================================================================
# CONFIGURATION
# ============================================================================
PI_HOST = "10.0.0.80"
PI_PORT = 5555

CAMERA_ID = 0
CAMERA_W = 1280
CAMERA_H = 720

OWL_MODEL = "google/owlvit-base-patch32"
DETECTION_LABELS = ["pen", "notebook", "ruler", "eraser", "glue stick", "scissors", "glasses"]
SCORE_THRESHOLD = 0.15

# Table/object height assumptions (meters)
TABLE_Z = 0.0      # Table surface height in robot frame
OBJECT_Z = 0.015   # Approx height of objects on table (half-height for grasp)
APPROACH_Z = 0.08   # Height to approach from above
LIFT_Z = 0.12       # Height to lift to after grasp

# Home position for servos (degrees, servo space 0-180)
HOME_ANGLES = [90, 90, 90, 90, 90]  # base, shoulder, elbow, wrist_pitch, wrist_roll
HOME_GRIPPER = 70
DEFAULT_SPEED = 40

# Gripper angles
GRIPPER_OPEN = 30
GRIPPER_CLOSED = 120


# ============================================================================
# IK ANGLE → SERVO ANGLE MAPPING
# ============================================================================
def ik_to_servo_angles(ik_angles: list) -> list:
    """
    Convert IK angles (centered on 0, range roughly ±90°) 
    to servo angles (0-180°, centered on 90°).
    
    IK gives: base_rotation, shoulder, elbow, wrist
    We need:  base, shoulder, elbow, wrist_pitch, wrist_roll (5 servos)
    
    The 5th servo (wrist_roll) stays at 90° (neutral) unless specified.
    """
    servo_angles = []
    for angle in ik_angles[:4]:
        # Map: IK 0° → Servo 90°, IK -90° → Servo 0°, IK +90° → Servo 180°
        servo = angle + 90.0
        servo = max(0, min(180, servo))
        servo_angles.append(int(round(servo)))
    
    # Add wrist_roll at neutral
    servo_angles.append(90)
    
    return servo_angles


# ============================================================================
# ZMQ CLIENT (PC → Pi)
# ============================================================================
class PiConnection:
    """ZMQ client to send commands to Raspberry Pi."""
    
    def __init__(self, host: str, port: int):
        self.host = host
        self.port = port
        self.ctx = None
        self.sock = None
        self.connected = False
    
    def connect(self) -> bool:
        try:
            self.ctx = zmq.Context()
            self.sock = self.ctx.socket(zmq.REQ)
            self.sock.setsockopt(zmq.RCVTIMEO, 15000)  # 15s timeout
            self.sock.setsockopt(zmq.SNDTIMEO, 5000)
            self.sock.connect(f"tcp://{self.host}:{self.port}")
            self.connected = True
            print(f"✅ Connecté au Pi: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Connexion Pi échouée: {e}")
            return False
    
    def send_move(self, angles: list, gripper: int = 70, speed: int = 40) -> dict:
        """Send movement command to Pi → Arduino."""
        if not self.connected:
            return {"status": "error", "response": "not connected"}
        
        cmd = {
            "target_angles": angles,
            "gripper": gripper,
            "speed": speed
        }
        
        print(f"   📡 → Pi: angles={angles} gripper={gripper} speed={speed}")
        
        try:
            self.sock.send_json(cmd)
            response = self.sock.recv_json()
            print(f"   📡 ← Pi: {response}")
            return response
        except zmq.error.Again:
            print("   ⏱️ Timeout Pi")
            return {"status": "error", "response": "timeout"}
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return {"status": "error", "response": str(e)}
    
    def send_home(self) -> dict:
        return self.send_move(HOME_ANGLES, HOME_GRIPPER, 30)
    
    def send_stop(self) -> dict:
        if not self.connected:
            return {"status": "error"}
        try:
            self.sock.send_json({"type": "EMERGENCY_STOP"})
            return self.sock.recv_json()
        except:
            return {"status": "error"}
    
    def close(self):
        if self.sock:
            self.sock.close()
        if self.ctx:
            self.ctx.term()
        self.connected = False


# ============================================================================
# DETECTION
# ============================================================================
class ObjectDetector:
    """OWL-ViT object detector."""
    
    def __init__(self, model_name: str, labels: list, threshold: float):
        self.labels = labels
        self.threshold = threshold
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"🧠 Device: {self.device}")
        
        print(f"🔄 Chargement modèle {model_name}...")
        self.processor = OwlViTProcessor.from_pretrained(model_name)
        self.model = OwlViTForObjectDetection.from_pretrained(model_name).to(self.device)
        self.model.eval()
        
        # Pre-tokenize text queries
        inputs_text = self.processor(text=labels, return_tensors="pt").to(self.device)
        self.input_ids = inputs_text["input_ids"]
        self.attention_mask = inputs_text["attention_mask"]
        
        print(f"✅ Modèle chargé. Labels: {labels}")
    
    def detect(self, frame: np.ndarray) -> list:
        """
        Run detection on a frame.
        
        Returns:
            List of dicts: [{label, score, bbox: (x1,y1,x2,y2), center: (cx,cy)}]
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        inputs_img = self.processor(images=pil_img, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                pixel_values=inputs_img["pixel_values"],
                input_ids=self.input_ids,
                attention_mask=self.attention_mask
            )
        
        target_sizes = torch.Tensor([pil_img.size[::-1]]).to(self.device)
        results = self.processor.image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=self.threshold
        )[0]
        
        detections = []
        for score, label_idx, box in zip(results["scores"], results["labels"], results["boxes"]):
            box_list = [round(v, 1) for v in box.tolist()]
            x1, y1, x2, y2 = box_list
            cx = (x1 + x2) / 2
            cy = (y1 + y2) / 2
            
            detections.append({
                "label": self.labels[label_idx],
                "score": round(score.item(), 3),
                "bbox": (int(x1), int(y1), int(x2), int(y2)),
                "center": (int(cx), int(cy))
            })
        
        # Sort by score descending
        detections.sort(key=lambda d: d["score"], reverse=True)
        return detections


# ============================================================================
# MAIN CONTROLLER
# ============================================================================
class RealArmController:
    """Orchestrates detection + voice + IK + ZMQ for real arm control."""
    
    def __init__(self, args):
        self.args = args
        self.running = True
        self.detections = []
        self.last_command = None
        self.pending_intent = None
        
        # IK chain
        self.chain = create_kinematic_chain()
        
        # Homography
        self.transformer = HomographyTransformer()
        self._load_calibration()
        
        # Pi connection
        self.pi = PiConnection(args.pi_host, args.pi_port)
        
        # Detector
        self.detector = ObjectDetector(OWL_MODEL, DETECTION_LABELS, SCORE_THRESHOLD)
        
        # Voice controller
        self.voice = VoiceController(language="fr-FR")
        
        # Camera
        self.cap = CameraCapture(args.camera_id, CAMERA_W, CAMERA_H)
    
    def _load_calibration(self):
        """Load homography calibration from file."""
        calib_path = os.path.join(os.path.dirname(__file__), "calibration.json")
        
        if not os.path.exists(calib_path):
            print("⚠️ Pas de calibration.json trouvé!")
            print("   Lance d'abord: python calibrate_real.py")
            print("   Mode DEMO: coordonnées simulées\n")
            return
        
        with open(calib_path, 'r') as f:
            data = json.load(f)
        
        H = np.array(data["homography_matrix"])
        self.transformer.H = H
        self.transformer.H_inv = np.linalg.inv(H)
        self.transformer.is_calibrated = True
        self.transformer.table_z = data.get("table_z", 0.0)
        
        print(f"✅ Calibration chargée ({len(data.get('pixel_points', []))} points)")
    
    def _find_object(self, target_label: str) -> dict:
        """Find the best detection matching target label."""
        for det in self.detections:
            if target_label.lower() in det["label"].lower():
                return det
        return None
    
    def _find_any_matching(self, intent) -> dict:
        """Find detection matching voice intent (label fuzzy matching)."""
        # Map French object names to English detection labels
        object_mapping = {
            "ciseau": "scissors",
            "ciseaux": "scissors",
            "stylo": "pen",
            "crayon": "pen",
            "cahier": "notebook",
            "carnet": "notebook",
            "règle": "ruler",
            "regle": "ruler",
            "gomme": "eraser",
            "colle": "glue stick",
            "lunettes": "glasses",
        }
        
        # Get target from intent
        raw = intent.raw_text.lower()
        
        # Try direct label match
        for label in DETECTION_LABELS:
            if label in raw:
                det = self._find_object(label)
                if det:
                    return det
        
        # Try French mapping
        for fr_name, en_label in object_mapping.items():
            if fr_name in raw:
                det = self._find_object(en_label)
                if det:
                    return det
        
        # If intent has target_object, try that
        if intent.target_object:
            for label in DETECTION_LABELS:
                if intent.target_object in label or label in intent.target_object:
                    det = self._find_object(label)
                    if det:
                        return det
        
        return None
    
    def _execute_pick(self, detection: dict):
        """
        Execute a full pick sequence:
        1. Move above object
        2. Descend
        3. Close gripper
        4. Lift
        """
        label = detection["label"]
        cx, cy = detection["center"]
        
        # Pixel → World
        if self.transformer.is_calibrated:
            wx, wy, wz = self.transformer.pixels_to_world(cx, cy)
        else:
            # Demo mode - rough estimate
            wx = 0.15 + (cx - CAMERA_W/2) * 0.0003
            wy = (CAMERA_H/2 - cy) * 0.0003
            wz = TABLE_Z
        
        print(f"\n{'='*60}")
        print(f"🎯 SAISIE: {label}")
        print(f"   Pixel: ({cx}, {cy})")
        print(f"   Monde: ({wx:.3f}, {wy:.3f})m")
        print(f"{'='*60}")
        
        # Check reachability
        if not check_reachability(wx, wy, APPROACH_Z):
            print(f"❌ Position hors de portée!")
            return False
        
        # Step 1: Approach from above (open gripper)
        print("\n1️⃣ Approche haute...")
        angles_approach, ok1 = get_servo_angles(wx, wy, APPROACH_Z, self.chain)
        if ok1:
            servo_approach = ik_to_servo_angles(angles_approach)
            self.pi.send_move(servo_approach, GRIPPER_OPEN, DEFAULT_SPEED)
            time.sleep(1.5)
        else:
            print(f"⚠️ IK approche imprécis, on tente quand même")
            servo_approach = ik_to_servo_angles(angles_approach)
            self.pi.send_move(servo_approach, GRIPPER_OPEN, DEFAULT_SPEED)
            time.sleep(1.5)
        
        # Step 2: Descend to object
        print("2️⃣ Descente vers l'objet...")
        grasp_z = TABLE_Z + OBJECT_Z
        angles_grasp, ok2 = get_servo_angles(wx, wy, grasp_z, self.chain)
        servo_grasp = ik_to_servo_angles(angles_grasp)
        self.pi.send_move(servo_grasp, GRIPPER_OPEN, int(DEFAULT_SPEED * 0.7))
        time.sleep(2.0)
        
        # Step 3: Close gripper
        print("3️⃣ 🦞 Fermeture pince!")
        self.pi.send_move(servo_grasp, GRIPPER_CLOSED, DEFAULT_SPEED)
        time.sleep(1.0)
        
        # Step 4: Lift
        print("4️⃣ 🏋️ Levage!")
        angles_lift, ok3 = get_servo_angles(wx, wy, LIFT_Z, self.chain)
        servo_lift = ik_to_servo_angles(angles_lift)
        self.pi.send_move(servo_lift, GRIPPER_CLOSED, int(DEFAULT_SPEED * 0.7))
        time.sleep(1.5)
        
        print(f"\n✅ {label} attrapé et soulevé!")
        
        # Step 5: Return to home (still holding)
        time.sleep(2.0)
        print("5️⃣ 🏠 Retour position home...")
        self.pi.send_move(HOME_ANGLES, GRIPPER_CLOSED, 30)
        time.sleep(2.0)
        
        # Step 6: Release
        print("6️⃣ ✋ Lâché!")
        self.pi.send_move(HOME_ANGLES, GRIPPER_OPEN, DEFAULT_SPEED)
        time.sleep(1.0)
        
        return True
    
    def _voice_thread(self):
        """Background thread for voice recognition."""
        print("\n🎤 Thread vocal démarré")
        print("   Dis 'Bonjour bras' pour activer\n")
        
        while self.running:
            try:
                if not self.voice.is_active:
                    # Wait for wake word
                    text, conf = self.voice.listen(timeout=5.0)
                    if text and self.voice.check_wake_word(text):
                        print("🟢 ACTIVÉ!")
                        self.voice.is_active = True
                        
                        # Check if command in same phrase
                        intent = self.voice.parse_intent(text)
                        if intent.action != CommandAction.UNKNOWN:
                            self.pending_intent = intent
                            result = self.voice.execute_voice_command(intent)
                            print(f"   {result['message']}")
                else:
                    # Listen for command
                    intent = self.voice.listen_command()
                    if intent and intent.action != CommandAction.UNKNOWN:
                        self.pending_intent = intent
                        result = self.voice.execute_voice_command(intent)
                        print(f"   {result['message']}")
                        
                        if intent.action == CommandAction.STOP:
                            self.pi.send_stop()
                            
            except Exception as e:
                print(f"⚠️ Voice error: {e}")
                time.sleep(1)
    
    def run(self):
        """Main loop: detection + voice + display."""
        # Connect to Pi
        if not self.pi.connect():
            print("⚠️ Pi non connecté. Mode affichage seul.")
        
        # Open camera
        if not self.cap.open():
            print("❌ Caméra introuvable")
            return
        
        # Start voice thread
        voice_thread = threading.Thread(target=self._voice_thread, daemon=True)
        voice_thread.start()
        
        # Move to home position
        if self.pi.connected:
            print("🏠 Position home...")
            self.pi.send_home()
            time.sleep(1)
        
        print("\n" + "=" * 60)
        print("🚀 REAL ARM CONTROLLER ACTIF")
        print("=" * 60)
        print("   Caméra: Live detection OWL-ViT")
        print("   Voix:   'Bonjour bras, attrape le [objet]'")
        print("   Clavier: 'h'=home  's'=stop  'q'=quitter")
        print("=" * 60 + "\n")
        
        cv2.namedWindow("Real Arm Controller", cv2.WINDOW_NORMAL)
        
        t0 = time.perf_counter()
        frame_count = 0
        fps = 0.0
        
        try:
            while self.running:
                ok, frame = self.cap.read()
                if not ok:
                    break
                
                # Run detection
                self.detections = self.detector.detect(frame)
                
                # Check for pending voice command
                if self.pending_intent is not None:
                    intent = self.pending_intent
                    self.pending_intent = None
                    
                    if intent.action == CommandAction.PICK:
                        det = self._find_any_matching(intent)
                        if det:
                            self._execute_pick(det)
                        else:
                            print(f"❌ Objet non trouvé dans la scène: '{intent.raw_text}'")
                    
                    elif intent.action == CommandAction.HOME:
                        self.pi.send_home()
                    
                    elif intent.action == CommandAction.STOP:
                        self.pi.send_stop()
                    
                    elif intent.action == CommandAction.OPEN:
                        self.pi.send_move(HOME_ANGLES, GRIPPER_OPEN, DEFAULT_SPEED)
                    
                    elif intent.action == CommandAction.CLOSE:
                        self.pi.send_move(HOME_ANGLES, GRIPPER_CLOSED, DEFAULT_SPEED)
                
                # Draw detections
                display = frame.copy()
                for det in self.detections:
                    x1, y1, x2, y2 = det["bbox"]
                    label = f"{det['label']}: {det['score']:.2f}"
                    
                    cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(display, label, (x1, y1 - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    
                    # Draw center
                    cx, cy = det["center"]
                    cv2.circle(display, (cx, cy), 5, (0, 0, 255), -1)
                    
                    # Show world coordinates if calibrated
                    if self.transformer.is_calibrated:
                        wx, wy, wz = self.transformer.pixels_to_world(cx, cy)
                        world_text = f"({wx:.2f}, {wy:.2f})m"
                        cv2.putText(display, world_text, (x1, y2 + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                
                # FPS
                frame_count += 1
                if frame_count % 5 == 0:
                    dt = time.perf_counter() - t0
                    fps = 5 / dt if dt > 0 else 0
                    t0 = time.perf_counter()
                
                # HUD
                h, w = display.shape[:2]
                cv2.rectangle(display, (0, 0), (w, 40), (0, 0, 0), -1)
                
                status_parts = [
                    f"FPS: {fps:.1f}",
                    f"Objets: {len(self.detections)}",
                    f"Pi: {'✅' if self.pi.connected else '❌'}",
                    f"Voix: {'🟢' if self.voice.is_active else '💤'}",
                    f"Calib: {'✅' if self.transformer.is_calibrated else '❌'}"
                ]
                status_text = " | ".join(status_parts)
                cv2.putText(display, status_text, (10, 28),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                cv2.imshow("Real Arm Controller", display)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('h'):
                    print("🏠 Home!")
                    self.pi.send_home()
                elif key == ord('s'):
                    print("🛑 STOP!")
                    self.pi.send_stop()
                elif key == ord('p'):
                    # Quick pick: grab the highest-score detection
                    if self.detections:
                        self._execute_pick(self.detections[0])
                    else:
                        print("❌ Aucun objet détecté")
        
        except KeyboardInterrupt:
            print("\n👋 Arrêt...")
        finally:
            self.running = False
            self.cap.close()
            self.pi.close()
            cv2.destroyAllWindows()


# ============================================================================
# MAIN
# ============================================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Real Arm Controller")
    parser.add_argument("--pi-host", type=str, default=PI_HOST)
    parser.add_argument("--pi-port", type=int, default=PI_PORT)
    parser.add_argument("--camera-id", type=int, default=CAMERA_ID)
    return parser.parse_args()


def main():
    args = parse_args()
    controller = RealArmController(args)
    controller.run()


if __name__ == "__main__":
    main()
