"""
TrajectoryAgent: Tracks the path of each person over time.
Monitors motion features: speed, direction, distance traveled.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import defaultdict
import math


class TrajectoryAgent:
    """Agent responsible for tracking person trajectories and motion analysis."""
    
    def __init__(self):
        """Initialize TrajectoryAgent."""
        self.trajectories = defaultdict(list)  # person_id -> list of (x, y, timestamp)
        self.speeds = defaultdict(list)  # person_id -> list of speeds
        self.total_distances = defaultdict(float)  # person_id -> total distance traveled
        
    def update_trajectory(self, person_id: int, centroid: Tuple[int, int], 
                         timestamp: int, fps: float = 30.0) -> Dict:
        """
        Update trajectory for a person and compute motion features.
        
        Args:
            person_id: Unique identifier for the person
            centroid: (x, y) coordinates of person's centroid
            timestamp: Frame number or timestamp
            fps: Frames per second (for speed calculation)
            
        Returns:
            Dictionary with trajectory information:
            {
                'path': List of (x, y, timestamp),
                'current_speed': float (m/s),
                'distance_traveled': float (meters),
                'is_moving': bool,
                'direction': float (degrees)
            }
        """
        # Add new point to trajectory
        self.trajectories[person_id].append((centroid[0], centroid[1], timestamp))
        
        # Keep only recent trajectory points (last 100 frames)
        if len(self.trajectories[person_id]) > 100:
            self.trajectories[person_id] = self.trajectories[person_id][-100:]
        
        # Calculate motion features
        trajectory = self.trajectories[person_id]
        
        if len(trajectory) < 2:
            return {
                'path': trajectory,
                'current_speed': 0.0,
                'distance_traveled': 0.0,
                'is_moving': False,
                'direction': 0.0
            }
        
        # Calculate total distance traveled (assuming pixel to meter conversion)
        # For simplicity, we'll use pixels. In real scenario, calibrate camera.
        total_distance_pixels = 0.0
        for i in range(1, len(trajectory)):
            x1, y1, _ = trajectory[i-1]
            x2, y2, _ = trajectory[i]
            distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_distance_pixels += distance
        
        # Convert pixels to meters (rough estimate: 1 meter ≈ 50 pixels)
        # This should be calibrated based on camera setup
        PIXELS_PER_METER = 50.0
        total_distance_meters = total_distance_pixels / PIXELS_PER_METER
        self.total_distances[person_id] = total_distance_meters
        
        # Calculate current speed (using last 5 frames)
        speed_pixels_per_frame = 0.0
        if len(trajectory) >= 5:
            recent_points = trajectory[-5:]
            for i in range(1, len(recent_points)):
                x1, y1, _ = recent_points[i-1]
                x2, y2, _ = recent_points[i]
                distance = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                speed_pixels_per_frame += distance
            speed_pixels_per_frame /= (len(recent_points) - 1)
        
        # Convert to m/s
        speed_m_per_sec = (speed_pixels_per_frame / PIXELS_PER_METER) * fps
        
        # Calculate direction (angle in degrees)
        if len(trajectory) >= 2:
            x1, y1, _ = trajectory[-2]
            x2, y2, _ = trajectory[-1]
            dx = x2 - x1
            dy = y2 - y1
            direction = math.degrees(math.atan2(dy, dx))
        else:
            direction = 0.0
        
        # Determine if person is moving (speed > threshold)
        is_moving = speed_m_per_sec > 0.1  # 0.1 m/s threshold
        
        return {
            'path': trajectory,
            'current_speed': speed_m_per_sec,
            'distance_traveled': total_distance_meters,
            'is_moving': is_moving,
            'direction': direction
        }
    
    def get_trajectory(self, person_id: int) -> List[Tuple[int, int, int]]:
        """Get full trajectory for a person."""
        return self.trajectories.get(person_id, [])
    
    def get_distance_traveled(self, person_id: int) -> float:
        """Get total distance traveled by a person in meters."""
        return self.total_distances.get(person_id, 0.0)
    
    def clear_trajectory(self, person_id: int):
        """Clear trajectory data for a person."""
        if person_id in self.trajectories:
            del self.trajectories[person_id]
        if person_id in self.speeds:
            del self.speeds[person_id]
        if person_id in self.total_distances:
            del self.total_distances[person_id]
    
    def clear_all(self):
        """Clear all trajectory data."""
        self.trajectories.clear()
        self.speeds.clear()
        self.total_distances.clear()

