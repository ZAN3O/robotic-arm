"""
kinematics.py - Kinematic Chain Configuration for HowToMechatronics 4-DOF Arm

This module configures the ikpy kinematic chain with the exact physical
dimensions of the robot and provides inverse kinematics calculations.

Dimensions (in meters):
- Base Height: 0.06m (6cm) - Ground to shoulder axis
- Humerus: 0.13m (13cm) - Shoulder to elbow axis
- Ulna: 0.13m (13cm) - Elbow to wrist axis  
- Gripper: 0.11m (11cm) - Wrist to gripper tip
"""

import numpy as np
from ikpy.chain import Chain
from ikpy.link import URDFLink, OriginLink

# ============================================================================
# PHYSICAL DIMENSIONS (meters)
# ============================================================================
BASE_HEIGHT = 0.06      # 6 cm - Sol à axe épaule
SHOULDER_OFFSET = 0.00  # Offset latéral épaule (0 pour ce design)
HUMERUS_LENGTH = 0.13   # 13 cm - Épaule à coude
ULNA_LENGTH = 0.13      # 13 cm - Coude à poignet
GRIPPER_LENGTH = 0.11   # 11 cm - Poignet à bout pince

# Total reach (approximate): 0.13 + 0.13 + 0.11 = 0.37m horizontal max
TOTAL_REACH = HUMERUS_LENGTH + ULNA_LENGTH + GRIPPER_LENGTH


def create_kinematic_chain() -> Chain:
    """
    Create the ikpy kinematic chain for the HowToMechatronics 4-DOF arm.
    
    The chain follows the DH convention with:
    - Joint 1: Base rotation (Z-axis)
    - Joint 2: Shoulder pitch (Y-axis)
    - Joint 3: Elbow pitch (Y-axis)
    - Joint 4: Wrist pitch (Y-axis)
    
    Returns:
        Chain: Configured ikpy chain ready for IK calculations
    """
    chain = Chain(name="howtomechatronics_arm", links=[
        # Base fixe (origine)
        OriginLink(),
        
        # Joint 1: Rotation base (autour de Z)
        URDFLink(
            name="base_rotation",
            origin_translation=[0, 0, BASE_HEIGHT],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1],  # Rotation autour de Z
            bounds=(-np.pi/2, np.pi/2)  # -90° à +90°
        ),
        
        # Joint 2: Épaule (pitch autour de Y)
        URDFLink(
            name="shoulder",
            origin_translation=[0, 0, 0],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],  # Rotation autour de Y
            bounds=(-np.pi/2, np.pi/2)
        ),
        
        # Link: Humerus (bras)
        URDFLink(
            name="humerus",
            origin_translation=[0, 0, HUMERUS_LENGTH],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],  # Joint 3: Coude
            bounds=(-np.pi/2, np.pi/2)
        ),
        
        # Link: Ulna (avant-bras)
        URDFLink(
            name="ulna",
            origin_translation=[0, 0, ULNA_LENGTH],
            origin_orientation=[0, 0, 0],
            rotation=[0, 1, 0],  # Joint 4: Poignet
            bounds=(-np.pi/2, np.pi/2)
        ),
        
        # End effector (gripper tip) - marked as inactive
        URDFLink(
            name="gripper_tip",
            origin_translation=[0, 0, GRIPPER_LENGTH],
            origin_orientation=[0, 0, 0],
            rotation=[0, 0, 1],  # Dummy rotation axis (link is inactive)
        ),
    ], active_links_mask=[False, True, True, True, True, False])
    
    return chain


def get_servo_angles(x: float, y: float, z: float, 
                     chain: Chain = None,
                     max_iterations: int = 100) -> tuple[list[float], bool]:
    """
    Calculate servo angles (in degrees) to reach target position.
    
    Args:
        x: Target X position in meters (forward/backward)
        y: Target Y position in meters (left/right)
        z: Target Z position in meters (up/down)
        chain: Optional pre-created chain (created if None)
        max_iterations: Max IK solver iterations
        
    Returns:
        tuple: (angles_degrees, success)
            - angles_degrees: List of 4 servo angles [base, shoulder, elbow, wrist]
            - success: True if IK solution found within tolerance
    """
    if chain is None:
        chain = create_kinematic_chain()
    
    # Target position
    target_position = [x, y, z]
    
    # Solve IK (using L-BFGS-B optimizer)
    # Initial position: all joints at 0
    initial_position = [0] * len(chain.links)
    
    try:
        ik_solution = chain.inverse_kinematics(
            target_position=target_position,
            initial_position=initial_position,
            max_iter=max_iterations
        )
        
        # Verify solution by forward kinematics
        fk_result = chain.forward_kinematics(ik_solution)
        achieved_position = fk_result[:3, 3]
        
        # Calculate error
        error = np.linalg.norm(np.array(target_position) - achieved_position)
        success = error < 0.01  # 1cm tolerance
        
        # Extract active joint angles (skip origin link and end effector)
        # Indices: 1=base, 2=shoulder, 3=elbow, 4=wrist
        active_angles_rad = [
            ik_solution[1],  # Base rotation
            ik_solution[2],  # Shoulder
            ik_solution[3],  # Elbow
            ik_solution[4],  # Wrist
        ]
        
        # Convert to degrees
        angles_degrees = [np.degrees(angle) for angle in active_angles_rad]
        
        if not success:
            print(f"⚠️ IK Warning: Target {target_position} achieved with error {error:.4f}m")
        
        return angles_degrees, success
        
    except Exception as e:
        print(f"❌ IK Error: {e}")
        return [0, 0, 0, 0], False


def validate_angles(angles: list[float], 
                   min_angle: float = -90, 
                   max_angle: float = 90) -> tuple[list[float], bool]:
    """
    Validate and clamp servo angles to safe limits.
    
    Args:
        angles: List of angles in degrees
        min_angle: Minimum allowed angle (default -90°)
        max_angle: Maximum allowed angle (default +90°)
        
    Returns:
        tuple: (clamped_angles, all_valid)
    """
    clamped = []
    all_valid = True
    
    for i, angle in enumerate(angles):
        if angle < min_angle or angle > max_angle:
            all_valid = False
            clamped_angle = max(min_angle, min(max_angle, angle))
            print(f"⚠️ Joint {i}: {angle:.1f}° clamped to {clamped_angle:.1f}°")
            clamped.append(clamped_angle)
        else:
            clamped.append(angle)
    
    return clamped, all_valid


def check_reachability(x: float, y: float, z: float) -> bool:
    """
    Quick check if a position is potentially reachable.
    
    Args:
        x, y, z: Target position in meters
        
    Returns:
        bool: True if position is within theoretical reach
    """
    # Distance from base axis
    horizontal_dist = np.sqrt(x**2 + y**2)
    
    # Height above base
    height_above_base = z - BASE_HEIGHT
    
    # 3D distance from shoulder
    dist_from_shoulder = np.sqrt(horizontal_dist**2 + height_above_base**2)
    
    # Check against max reach
    max_reach = HUMERUS_LENGTH + ULNA_LENGTH + GRIPPER_LENGTH
    min_reach = abs(HUMERUS_LENGTH - ULNA_LENGTH - GRIPPER_LENGTH)
    
    return min_reach < dist_from_shoulder < max_reach


# ============================================================================
# TEST / DEMO
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🦾 HowToMechatronics Arm - Kinematic Chain Test")
    print("=" * 60)
    
    # Create chain
    chain = create_kinematic_chain()
    print(f"\n✅ Chain created with {len(chain.links)} links")
    print(f"   Total theoretical reach: {TOTAL_REACH * 100:.1f} cm")
    
    # Test positions
    test_positions = [
        (0.15, 0.00, 0.15),   # Front, medium height
        (0.10, 0.10, 0.20),   # Front-left, higher
        (0.20, -0.05, 0.10),  # Front-right, lower
        (0.25, 0.00, 0.06),   # Far reach, base height
    ]
    
    print("\n📍 Testing IK for various positions:")
    print("-" * 60)
    
    for x, y, z in test_positions:
        reachable = check_reachability(x, y, z)
        print(f"\nTarget: ({x:.2f}, {y:.2f}, {z:.2f})m - Reachable: {reachable}")
        
        if reachable:
            angles, success = get_servo_angles(x, y, z, chain)
            status = "✅" if success else "⚠️"
            print(f"  {status} Angles: Base={angles[0]:.1f}°, Shoulder={angles[1]:.1f}°, "
                  f"Elbow={angles[2]:.1f}°, Wrist={angles[3]:.1f}°")
            
            # Validate
            validated, valid = validate_angles(angles)
            if not valid:
                print(f"  📌 Validated angles: {validated}")
