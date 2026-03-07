"""
pi_relay.py - Raspberry Pi ZMQ-to-Serial Relay

Runs on the Raspberry Pi.
Receives JSON commands from the PC via ZMQ (WiFi),
forwards them as serial commands to the Arduino Mega.

Usage:
    python3 pi_relay.py
    python3 pi_relay.py --port 5555 --serial /dev/ttyACM0 --baud 115200
"""

import zmq
import serial
import time
import json
import argparse
import sys


# ============================================================================
# POSITION IDLE (repos) — envoyée au démarrage
# ============================================================================
# base(pin8), shoulder(pin5), elbow(pin7), wrist_roll(pin6), wrist_yaw(pin4)
IDLE_ANGLES = [90, 90, 90, 90, 90]
IDLE_GRIPPER = 70
IDLE_SPEED = 30  # Lent pour le démarrage


def parse_args():
    parser = argparse.ArgumentParser(description="Pi Relay: ZMQ → Serial → Arduino")
    parser.add_argument("--port", type=int, default=5555, help="ZMQ port to listen on")
    parser.add_argument("--serial", type=str, default="/dev/ttyACM0", 
                        help="Serial port for Arduino (try /dev/ttyUSB0 if ACM0 fails)")
    parser.add_argument("--baud", type=int, default=115200, help="Serial baud rate")
    return parser.parse_args()


class ArduinoSerial:
    """Manages serial communication with Arduino Mega."""
    
    def __init__(self, port: str, baud: int):
        self.port = port
        self.baud = baud
        self.ser = None
    
    def connect(self) -> bool:
        """Open serial connection to Arduino."""
        try:
            self.ser = serial.Serial(self.port, self.baud, timeout=5)
            time.sleep(2)  # Wait for Arduino reset
            
            # Flush startup messages
            while self.ser.in_waiting:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                print(f"   Arduino: {line}")
            
            print(f"✅ Arduino connecté sur {self.port} @ {self.baud}")
            return True
            
        except serial.SerialException as e:
            print(f"❌ Erreur serial: {e}")
            print(f"   Essayez: --serial /dev/ttyUSB0 ou /dev/ttyACM1")
            return False
    
    def send_command(self, angles: list, gripper: int, speed: int) -> str:
        """
        Send angle command to Arduino.
        
        Args:
            angles: List of servo angles [base, shoulder, elbow, wrist_roll, wrist_yaw]
            gripper: Gripper angle (0-180)
            speed: Movement speed (1-100)
        
        Returns:
            Response string from Arduino
        """
        if self.ser is None:
            return "ERR:not_connected"
        
        # Build command: <base,shoulder,elbow,wrist_pitch,wrist_roll,gripper,speed>
        all_values = list(angles) + [gripper, speed]
        cmd = "<" + ",".join(str(int(v)) for v in all_values) + ">"
        
        print(f"   → Arduino: {cmd}")
        
        self.ser.write(cmd.encode('utf-8'))
        self.ser.flush()
        
        # Wait for response
        response = self._read_response(timeout=10)
        print(f"   ← Arduino: {response}")
        
        return response
    
    def send_raw(self, raw_cmd: str) -> str:
        """Send a raw command string (e.g., <HOME>, <STOP>)."""
        if self.ser is None:
            return "ERR:not_connected"
        
        cmd = f"<{raw_cmd}>"
        print(f"   → Arduino: {cmd}")
        self.ser.write(cmd.encode('utf-8'))
        self.ser.flush()
        
        return self._read_response(timeout=10)
    
    def _read_response(self, timeout: float = 5) -> str:
        """Read response line from Arduino with timeout."""
        start = time.time()
        while (time.time() - start) < timeout:
            if self.ser.in_waiting > 0:
                line = self.ser.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    return line
            time.sleep(0.01)
        return "ERR:timeout"
    
    def close(self):
        """Close serial connection."""
        if self.ser:
            self.ser.close()
            print("🔌 Serial fermé")


def run_relay(zmq_port: int, arduino: ArduinoSerial):
    """Main relay loop: ZMQ → Serial."""
    
    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.bind(f"tcp://*:{zmq_port}")
    
    print(f"\n{'='*60}")
    print(f"🍓 PI RELAY ACTIF")
    print(f"   ZMQ:    tcp://*:{zmq_port}")
    print(f"   Serial: {arduino.port}")
    print(f"{'='*60}")
    print("En attente de commandes du PC...\n")
    
    try:
        while True:
            # Wait for command from PC
            message = sock.recv_json()
            print(f"\n📥 Reçu: {json.dumps(message, indent=2)}")
            
            # Handle different command types
            cmd_type = message.get("type", "move")
            
            if cmd_type == "EMERGENCY_STOP":
                response = arduino.send_raw("STOP")
                sock.send_json({"status": "ok", "response": response})
                
            elif cmd_type == "HOME":
                response = arduino.send_raw("HOME")
                sock.send_json({"status": "ok", "response": response})
                
            elif cmd_type == "STATUS":
                response = arduino.send_raw("STATUS")
                sock.send_json({"status": "ok", "response": response})
                
            else:
                # Standard move command
                angles = message.get("target_angles", [90, 90, 90, 90, 90])
                gripper = message.get("gripper", 70)
                speed = message.get("speed", 40)
                
                response = arduino.send_command(angles, gripper, speed)
                
                success = response.startswith("OK")
                sock.send_json({
                    "status": "ok" if success else "error", 
                    "response": response
                })
            
            print(f"📤 Répondu au PC\n")
    
    except KeyboardInterrupt:
        print("\n\n👋 Arrêt du relay")
    finally:
        sock.close()
        ctx.term()
        arduino.close()


def main():
    args = parse_args()
    
    print("=" * 60)
    print("🍓 Raspberry Pi - Relay ZMQ → Serial")
    print("=" * 60)
    
    # Connect to Arduino
    arduino = ArduinoSerial(args.serial, args.baud)
    if not arduino.connect():
        sys.exit(1)
    
    # Envoyer la position IDLE au démarrage
    print(f"\n🏠 Position IDLE au démarrage: {IDLE_ANGLES} gripper={IDLE_GRIPPER}")
    response = arduino.send_command(IDLE_ANGLES, IDLE_GRIPPER, IDLE_SPEED)
    print(f"   Réponse: {response}\n")
    
    # Run relay
    run_relay(args.port, arduino)


if __name__ == "__main__":
    main()
