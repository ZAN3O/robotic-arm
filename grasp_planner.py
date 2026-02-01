"""
grasp_planner.py - Grasp Strategy Planning Module

This module computes optimal grasp poses and approach trajectories
based on detected object properties.
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, List, Optional
from enum import Enum

# Import our perception module types
from perception import DetectedObject, ObjectShape


class GraspApproach(Enum):
    """Grasp approach strategies."""
    TOP_DOWN = "top_down"       # Approach from above
    SIDE = "side"               # Approach from side
    ANGLED = "angled"           # 45° approach


@dataclass
class GraspPose:
    """
    Represents a planned grasp pose.
    """
    # Target position for end-effector (meters)
    position: Tuple[float, float, float]
    
    # Gripper rotation (degrees around Z)
    rotation: float
    
    # Gripper aperture (how wide to open, meters)
    aperture: float
    
    # Approach strategy
    approach: GraspApproach
    
    # Pre-grasp offset (approach from this offset before final grasp)
    pre_grasp_offset: Tuple[float, float, float]
    
    # Lift height after grasp (meters)
    lift_height: float = 0.05
    
    # Confidence in this grasp (0-1)
    confidence: float = 1.0


class GraspPlanner:
    """
    Plans grasp poses and approach trajectories for pick operations.
    """
    
    # Default heights for different operations (meters)
    DEFAULT_APPROACH_HEIGHT = 0.08  # Height to approach from
    DEFAULT_GRASP_HEIGHT = 0.02     # Height for grasping
    DEFAULT_LIFT_HEIGHT = 0.10      # Height to lift after grasp
    
    # Gripper limits (meters)
    GRIPPER_MIN_APERTURE = 0.01   # 1cm min opening
    GRIPPER_MAX_APERTURE = 0.06   # 6cm max opening
    
    # Safety margins
    OBJECT_CLEARANCE = 0.01  # 1cm clearance around objects
    
    def __init__(self, table_height: float = 0.0):
        """
        Initialize the grasp planner.
        
        Args:
            table_height: Height of table surface in robot frame (meters)
        """
        self.table_height = table_height
    
    def compute_grasp_pose(self, obj: DetectedObject,
                           object_height: float = 0.03) -> GraspPose:
        """
        Compute optimal grasp pose for a detected object.
        
        Args:
            obj: Detected object with world coordinates
            object_height: Estimated object height (meters)
            
        Returns:
            GraspPose: Planned grasp configuration
        """
        if obj.center_world is None:
            raise ValueError("Object must have world coordinates")
        
        x, y, _ = obj.center_world
        
        # Select approach strategy based on shape
        approach = self._select_approach(obj.shape, object_height)
        
        # Calculate grasp height
        if approach == GraspApproach.TOP_DOWN:
            # Grasp from top, position at half height
            grasp_z = self.table_height + object_height / 2
        elif approach == GraspApproach.SIDE:
            # Side grasp, position at half height
            grasp_z = self.table_height + object_height / 2
        else:
            # Angled approach
            grasp_z = self.table_height + object_height * 0.6
        
        # Gripper rotation
        rotation = self._compute_rotation(obj)
        
        # Gripper aperture (slightly wider than object)
        estimated_width = obj.estimated_size_m if obj.estimated_size_m > 0 else 0.03
        aperture = min(
            max(estimated_width * 1.2, self.GRIPPER_MIN_APERTURE),
            self.GRIPPER_MAX_APERTURE
        )
        
        # Pre-grasp offset (approach from above)
        pre_grasp_offset = (0, 0, self.DEFAULT_APPROACH_HEIGHT - grasp_z)
        
        return GraspPose(
            position=(x, y, grasp_z),
            rotation=rotation,
            aperture=aperture,
            approach=approach,
            pre_grasp_offset=pre_grasp_offset,
            lift_height=self.DEFAULT_LIFT_HEIGHT,
            confidence=obj.confidence
        )
    
    def _select_approach(self, shape: ObjectShape, 
                         height: float) -> GraspApproach:
        """Select best approach strategy for object shape."""
        if shape == ObjectShape.SPHERE:
            return GraspApproach.TOP_DOWN
        
        elif shape == ObjectShape.CUBE:
            return GraspApproach.TOP_DOWN
        
        elif shape == ObjectShape.CYLINDER:
            # Side grasp for tall cylinders
            if height > 0.05:
                return GraspApproach.SIDE
            return GraspApproach.TOP_DOWN
        
        elif shape == ObjectShape.RECTANGLE:
            return GraspApproach.TOP_DOWN
        
        # Default
        return GraspApproach.TOP_DOWN
    
    def _compute_rotation(self, obj: DetectedObject) -> float:
        """Compute optimal gripper rotation for object."""
        if obj.shape == ObjectShape.SPHERE:
            return 0.0  # Doesn't matter for spheres
        
        elif obj.shape == ObjectShape.CUBE:
            # Align with cube edges (nearest 45°)
            return round(obj.rotation / 45) * 45
        
        elif obj.shape == ObjectShape.CYLINDER:
            # Perpendicular to major axis
            return (obj.rotation + 90) % 180
        
        elif obj.shape == ObjectShape.RECTANGLE:
            # Align with short axis
            return obj.rotation
        
        return 0.0
    
    def get_gripper_aperture(self, object_width: float) -> float:
        """
        Calculate gripper aperture for object width.
        
        Args:
            object_width: Object width in meters
            
        Returns:
            float: Gripper aperture in meters
        """
        # Add margin for safe grasp
        aperture = object_width + self.OBJECT_CLEARANCE * 2
        
        # Clamp to limits
        return max(
            min(aperture, self.GRIPPER_MAX_APERTURE),
            self.GRIPPER_MIN_APERTURE
        )
    
    def plan_approach(self, 
                      current_pos: Tuple[float, float, float],
                      grasp_pose: GraspPose,
                      obstacles: Optional[List[DetectedObject]] = None
                      ) -> List[Tuple[float, float, float]]:
        """
        Plan approach trajectory from current position to grasp pose.
        
        Args:
            current_pos: Current end-effector position (x, y, z)
            grasp_pose: Target grasp pose
            obstacles: List of obstacles to avoid
            
        Returns:
            List of waypoints [(x, y, z), ...]
        """
        waypoints = []
        
        target_x, target_y, target_z = grasp_pose.position
        offset_x, offset_y, offset_z = grasp_pose.pre_grasp_offset
        
        # Pre-grasp position (above target)
        pre_grasp = (
            target_x + offset_x,
            target_y + offset_y,
            target_z + offset_z
        )
        
        # Step 1: Move up if below safe height
        safe_height = self.DEFAULT_APPROACH_HEIGHT
        if current_pos[2] < safe_height:
            waypoints.append((current_pos[0], current_pos[1], safe_height))
        
        # Step 2: Move to above target (XY movement at safe height)
        waypoints.append((pre_grasp[0], pre_grasp[1], safe_height))
        
        # Step 3: Pre-grasp position
        waypoints.append(pre_grasp)
        
        # Step 4: Final grasp position
        waypoints.append(grasp_pose.position)
        
        return waypoints
    
    def plan_place(self,
                   current_pos: Tuple[float, float, float],
                   place_pos: Tuple[float, float, float],
                   ) -> List[Tuple[float, float, float]]:
        """
        Plan trajectory to place an object.
        
        Args:
            current_pos: Current position (holding object)
            place_pos: Target placement position
            
        Returns:
            List of waypoints
        """
        waypoints = []
        
        safe_height = max(current_pos[2], self.DEFAULT_APPROACH_HEIGHT)
        
        # Lift if needed
        if current_pos[2] < safe_height:
            waypoints.append((current_pos[0], current_pos[1], safe_height))
        
        # Move above target
        waypoints.append((place_pos[0], place_pos[1], safe_height))
        
        # Pre-place (slightly above)
        waypoints.append((place_pos[0], place_pos[1], place_pos[2] + 0.02))
        
        # Final place
        waypoints.append(place_pos)
        
        return waypoints


class PickAndPlace:
    """
    High-level pick and place operations.
    """
    
    def __init__(self, planner: GraspPlanner):
        """
        Initialize pick and place controller.
        
        Args:
            planner: GraspPlanner instance
        """
        self.planner = planner
        self.holding_object: Optional[DetectedObject] = None
    
    def pick(self, obj: DetectedObject) -> List[dict]:
        """
        Generate pick sequence for an object.
        
        Args:
            obj: Object to pick
            
        Returns:
            List of action dictionaries
        """
        # Compute grasp
        grasp = self.planner.compute_grasp_pose(obj)
        
        # Current position (would come from robot state in real use)
        current = (0.15, 0.0, 0.15)
        
        # Plan approach
        waypoints = self.planner.plan_approach(current, grasp)
        
        actions = []
        
        # Open gripper
        actions.append({
            'action': 'gripper',
            'aperture': grasp.aperture,
            'description': 'Open gripper'
        })
        
        # Move through waypoints
        for i, wp in enumerate(waypoints):
            actions.append({
                'action': 'move',
                'position': wp,
                'description': f'Move to waypoint {i+1}'
            })
        
        # Close gripper
        actions.append({
            'action': 'gripper',
            'aperture': 0,  # Close fully
            'description': 'Close gripper to grasp'
        })
        
        # Lift
        lift_pos = (
            grasp.position[0],
            grasp.position[1],
            grasp.position[2] + grasp.lift_height
        )
        actions.append({
            'action': 'move',
            'position': lift_pos,
            'description': 'Lift object'
        })
        
        self.holding_object = obj
        return actions
    
    def place(self, position: Tuple[float, float, float]) -> List[dict]:
        """
        Generate place sequence.
        
        Args:
            position: Target placement position
            
        Returns:
            List of action dictionaries
        """
        if self.holding_object is None:
            return [{'action': 'error', 'description': 'Not holding any object'}]
        
        actions = []
        
        # Current position (assume lifted)
        current = (position[0], position[1], 0.15)
        
        # Plan place trajectory
        waypoints = self.planner.plan_place(current, position)
        
        # Move through waypoints
        for i, wp in enumerate(waypoints):
            actions.append({
                'action': 'move',
                'position': wp,
                'description': f'Move to place waypoint {i+1}'
            })
        
        # Open gripper to release
        actions.append({
            'action': 'gripper',
            'aperture': 0.04,
            'description': 'Open gripper to release'
        })
        
        # Retract upward
        retract_pos = (position[0], position[1], position[2] + 0.05)
        actions.append({
            'action': 'move',
            'position': retract_pos,
            'description': 'Retract after placing'
        })
        
        self.holding_object = None
        return actions


# ============================================================================
# TEST
# ============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🤚 Grasp Planner Test")
    print("=" * 60)
    
    from perception import ObjectColor
    
    # Create test object
    test_obj = DetectedObject(
        id=1,
        shape=ObjectShape.CUBE,
        color=ObjectColor.RED,
        bbox=(100, 100, 50, 50),
        center_px=(125, 125),
        center_world=(0.15, 0.05, 0.0),
        estimated_size_m=0.03,
        confidence=0.95,
        rotation=15.0
    )
    
    # Create planner
    planner = GraspPlanner(table_height=0.0)
    
    # Compute grasp
    grasp = planner.compute_grasp_pose(test_obj, object_height=0.03)
    
    print(f"\n📦 Object: {test_obj}")
    print(f"\n🎯 Grasp Pose:")
    print(f"   Position: ({grasp.position[0]:.3f}, {grasp.position[1]:.3f}, {grasp.position[2]:.3f})m")
    print(f"   Rotation: {grasp.rotation:.1f}°")
    print(f"   Aperture: {grasp.aperture*100:.1f}cm")
    print(f"   Approach: {grasp.approach.value}")
    
    # Plan approach
    current = (0.1, 0.0, 0.15)
    waypoints = planner.plan_approach(current, grasp)
    
    print(f"\n📍 Approach trajectory ({len(waypoints)} waypoints):")
    for i, wp in enumerate(waypoints):
        print(f"   {i+1}. ({wp[0]:.3f}, {wp[1]:.3f}, {wp[2]:.3f})m")
    
    # Full pick sequence
    pick_place = PickAndPlace(planner)
    actions = pick_place.pick(test_obj)
    
    print(f"\n🔄 Pick sequence ({len(actions)} actions):")
    for action in actions:
        print(f"   - {action['description']}")
