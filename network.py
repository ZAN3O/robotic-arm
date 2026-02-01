"""
network.py - ZeroMQ Network Communication Module

This module handles communication between the Brain (Mac M4)
and the Motor Controller (Raspberry Pi) via ZeroMQ.

Protocol: JSON messages over ZeroMQ PUB/SUB or REQ/REP
"""

import json
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Any
from datetime import datetime


@dataclass
class MotorCommand:
    """
    Command to send to the motor controller.
    """
    # Servo angles in degrees
    base_angle: float      # Joint 1: Base rotation (-90 to +90)
    shoulder_angle: float  # Joint 2: Shoulder (-90 to +90)
    elbow_angle: float     # Joint 3: Elbow (-90 to +90)
    wrist_angle: float     # Joint 4: Wrist (-90 to +90)
    
    # Gripper (0-100, where 0=closed, 100=fully open)
    gripper: int = 50
    
    # Speed (0-100)
    speed: int = 50
    
    # Timestamp
    timestamp: str = ""
    
    # Command ID for tracking
    cmd_id: int = 0
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(asdict(self))
    
    @classmethod
    def from_angles(cls, angles: List[float], gripper: int = 50, 
                    speed: int = 50) -> 'MotorCommand':
        """Create command from angle list."""
        if len(angles) != 4:
            raise ValueError("Expected 4 angles")
        return cls(
            base_angle=angles[0],
            shoulder_angle=angles[1],
            elbow_angle=angles[2],
            wrist_angle=angles[3],
            gripper=gripper,
            speed=speed
        )


@dataclass  
class RobotStatus:
    """
    Status received from the motor controller.
    """
    # Current joint angles
    angles: List[float]
    
    # Gripper state
    gripper: int
    
    # Is robot moving?
    is_moving: bool
    
    # Any error codes
    error_code: int = 0
    error_message: str = ""
    
    # Timestamp
    timestamp: str = ""
    
    @classmethod
    def from_json(cls, json_str: str) -> 'RobotStatus':
        """Parse status from JSON."""
        data = json.loads(json_str)
        return cls(**data)


class NetworkController:
    """
    ZeroMQ network interface for robot communication.
    
    Supports:
    - PUB/SUB: For one-way command streaming
    - REQ/REP: For command-response pairs
    """
    
    def __init__(self, mode: str = "simulated"):
        """
        Initialize network controller.
        
        Args:
            mode: "simulated" (print only), "zmq" (real ZeroMQ), or "serial" (USB)
        """
        self.mode = mode
        self.context = None
        self.socket = None
        self.connected = False
        self.command_counter = 0
        
        # Default connection settings
        self.host = "192.168.1.100"  # Raspberry Pi IP
        self.port = 5555
    
    def connect(self, host: str = None, port: int = None) -> bool:
        """
        Connect to the motor controller.
        
        Args:
            host: IP address of Raspberry Pi
            port: ZeroMQ port
            
        Returns:
            bool: True if connected successfully
        """
        if host:
            self.host = host
        if port:
            self.port = port
        
        if self.mode == "simulated":
            print(f"🔌 [SIMULATED] Would connect to {self.host}:{self.port}")
            self.connected = True
            return True
        
        elif self.mode == "zmq":
            try:
                import zmq
                self.context = zmq.Context()
                self.socket = self.context.socket(zmq.REQ)
                self.socket.connect(f"tcp://{self.host}:{self.port}")
                self.socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1s timeout
                
                print(f"✅ Connected to {self.host}:{self.port}")
                self.connected = True
                return True
                
            except ImportError:
                print("❌ ZeroMQ not installed. Run: pip install pyzmq")
                return False
            except Exception as e:
                print(f"❌ Connection failed: {e}")
                return False
        
        return False
    
    def disconnect(self):
        """Disconnect from motor controller."""
        if self.socket:
            self.socket.close()
        if self.context:
            self.context.term()
        self.connected = False
        print("🔌 Disconnected")
    
    def send_to_network(self, angles: List[float], 
                        gripper: int = 50,
                        speed: int = 50) -> Optional[Dict]:
        """
        Send motor command to the network.
        
        Args:
            angles: List of 4 servo angles in degrees
            gripper: Gripper position (0-100)
            speed: Movement speed (0-100)
            
        Returns:
            dict: Response from controller (or simulated response)
        """
        self.command_counter += 1
        
        # Create command
        cmd = MotorCommand.from_angles(angles, gripper=gripper, speed=speed)
        cmd.cmd_id = self.command_counter
        
        json_data = cmd.to_json()
        
        if self.mode == "simulated":
            return self._send_simulated(cmd, json_data)
        elif self.mode == "zmq":
            return self._send_zmq(json_data)
        
        return None
    
    def _send_simulated(self, cmd: MotorCommand, json_data: str) -> Dict:
        """Simulated send (print only)."""
        print("\n" + "=" * 50)
        print("📡 SIMULATED NETWORK OUTPUT")
        print("=" * 50)
        print(f"Command ID: {cmd.cmd_id}")
        print(f"Angles: Base={cmd.base_angle:.1f}°, Shoulder={cmd.shoulder_angle:.1f}°, "
              f"Elbow={cmd.elbow_angle:.1f}°, Wrist={cmd.wrist_angle:.1f}°")
        print(f"Gripper: {cmd.gripper}%  Speed: {cmd.speed}%")
        print("-" * 50)
        print(f"JSON: {json_data}")
        print("=" * 50)
        
        return {
            'success': True,
            'cmd_id': cmd.cmd_id,
            'message': 'Simulated send successful'
        }
    
    def _send_zmq(self, json_data: str) -> Optional[Dict]:
        """Send via ZeroMQ."""
        if not self.connected or not self.socket:
            print("❌ Not connected")
            return None
        
        try:
            # Send command
            self.socket.send_string(json_data)
            
            # Wait for response
            response = self.socket.recv_string()
            
            return json.loads(response)
            
        except Exception as e:
            print(f"❌ Send failed: {e}")
            return None
    
    def send_emergency_stop(self) -> bool:
        """Send emergency stop command."""
        stop_cmd = {
            'type': 'EMERGENCY_STOP',
            'timestamp': datetime.now().isoformat()
        }
        
        if self.mode == "simulated":
            print("🛑 [SIMULATED] EMERGENCY STOP SENT")
            return True
        
        elif self.mode == "zmq" and self.socket:
            try:
                self.socket.send_string(json.dumps(stop_cmd))
                return True
            except:
                return False
        
        return False
    
    def send_home(self) -> Optional[Dict]:
        """Send command to return to home position."""
        return self.send_to_network([0, 0, 0, 0], gripper=50, speed=30)
    
    def get_status(self) -> Optional[RobotStatus]:
        """Request current robot status."""
        if self.mode == "simulated":
            # Return fake status
            return RobotStatus(
                angles=[0, 0, 0, 0],
                gripper=50,
                is_moving=False,
                timestamp=datetime.now().isoformat()
            )
        
        elif self.mode == "zmq" and self.socket:
            try:
                status_request = {'type': 'STATUS_REQUEST'}
                self.socket.send_string(json.dumps(status_request))
                response = self.socket.recv_string()
                return RobotStatus.from_json(response)
            except:
                return None
        
        return None


def format_command_for_display(angles: List[float], 
                               gripper: int = 50) -> str:
    """
    Format angles for nice display.
    
    Args:
        angles: List of 4 angles
        gripper: Gripper position
        
    Returns:
        str: Formatted string
    """
    names = ["Base", "Shoulder", "Elbow", "Wrist"]
    parts = [f"{name}: {angle:+6.1f}°" for name, angle in zip(names, angles)]
    return " | ".join(parts) + f" | Gripper: {gripper}%"


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("📡 Network Module Test")
    print("=" * 60)
    
    # Create controller in simulated mode
    net = NetworkController(mode="simulated")
    
    # Connect
    net.connect()
    
    # Test commands
    test_angles = [
        [0, 0, 0, 0],
        [30, 45, -20, 10],
        [-15, 60, -45, 30],
    ]
    
    print("\n📤 Sending test commands:\n")
    
    for angles in test_angles:
        result = net.send_to_network(angles, gripper=50, speed=50)
        print(f"Result: {result}\n")
    
    # Test status
    print("\n📊 Getting robot status:")
    status = net.get_status()
    if status:
        print(f"   Angles: {status.angles}")
        print(f"   Gripper: {status.gripper}%")
        print(f"   Moving: {status.is_moving}")
    
    # Test emergency stop
    print("\n🛑 Testing emergency stop:")
    net.send_emergency_stop()
    
    # Disconnect
    net.disconnect()
