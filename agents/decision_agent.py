"""
DecisionAgent: Applies rule-based logic for loitering & restricted area intrusion detection.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from shapely.geometry import Point, Polygon


class DecisionAgent:
    """Agent responsible for rule-based decision making."""
    
    def __init__(self, loitering_duration_threshold: float = 120.0,
                 min_movement_meters: float = 5.0,
                 enable_loitering_detection: bool = True,
                 restricted_zones: List[Dict] = None):
        """
        Initialize DecisionAgent.
        
        Args:
            loitering_duration_threshold: Duration in seconds to flag as loitering
            min_movement_meters: Minimum movement in meters to avoid loitering flag
            enable_loitering_detection: Toggle loitering rule on/off
            restricted_zones: List of restricted zone definitions
        """
        self.loitering_duration_threshold = loitering_duration_threshold
        self.min_movement_meters = min_movement_meters
        self.enable_loitering_detection = enable_loitering_detection
        self.restricted_zones = restricted_zones or []
        self.zone_polygons = {}
        
        # Convert zone polygons to Shapely polygons for efficient checking
        for zone in self.restricted_zones:
            name = zone.get('name', 'Unknown')
            polygon_coords = zone.get('polygon', [])
            if len(polygon_coords) >= 3:
                self.zone_polygons[name] = {
                    'polygon': Polygon(polygon_coords),
                    'threat_level': zone.get('threat_level', 'medium')
                }
    
    def check_restricted_zone(self, centroid: Tuple[int, int]) -> Optional[Dict]:
        """
        Check if a person's centroid is inside a restricted zone.
        
        Args:
            centroid: (x, y) coordinates of person's centroid
            
        Returns:
            Dictionary with zone information if inside, None otherwise:
            {
                'zone_name': str,
                'threat_level': str,
                'is_inside': bool
            }
        """
        point = Point(centroid[0], centroid[1])
        
        for zone_name, zone_data in self.zone_polygons.items():
            if zone_data['polygon'].contains(point):
                return {
                    'zone_name': zone_name,
                    'threat_level': zone_data['threat_level'],
                    'is_inside': True
                }
        
        return None
    
    def evaluate_person(self, person_id: int, 
                       dwell_duration: float,
                       distance_traveled: float,
                       centroid: Tuple[int, int],
                       is_moving: bool = False) -> Dict:
        """
        Evaluate a person's behavior and determine status.
        
        Args:
            person_id: Unique identifier for the person
            dwell_duration: Time spent in area (seconds)
            distance_traveled: Total distance traveled (meters)
            centroid: Current position (x, y)
            is_moving: Whether person is currently moving
            
        Returns:
            Dictionary with evaluation results:
            {
                'person_id': int,
                'status': str ('Normal', 'Loitering', 'Intrusion'),
                'threat_level': str ('low', 'medium', 'high'),
                'reason': str,
                'zone_info': dict or None
            }
        """
        # Check restricted zone intrusion
        zone_info = self.check_restricted_zone(centroid)
        
        # Rule 1: Restricted Zone Intrusion
        if zone_info and zone_info['is_inside']:
            threat_level = zone_info['threat_level']
            return {
                'person_id': person_id,
                'status': 'Intrusion',
                'threat_level': threat_level,
                'reason': f"Person entered restricted zone: {zone_info['zone_name']}",
                'zone_info': zone_info
            }
        
        # Rule 2: Loitering Detection (optional)
        if not self.enable_loitering_detection:
            return {
                'person_id': person_id,
                'status': 'Normal',
                'threat_level': 'low',
                'reason': 'Loitering detection disabled',
                'zone_info': None
            }

        # Rule 2: Loitering Detection
        # Loitering requires BOTH conditions: duration > threshold AND low movement
        is_loitering = False
        loitering_reasons = []
        
        # Check if BOTH conditions are met for loitering
        duration_exceeded = dwell_duration > self.loitering_duration_threshold
        low_movement = distance_traveled < self.min_movement_meters
        
        if duration_exceeded and low_movement:
            is_loitering = True
            loitering_reasons.append(f"Duration {dwell_duration:.1f}s > {self.loitering_duration_threshold}s")
            loitering_reasons.append(f"Movement {distance_traveled:.1f}m < {self.min_movement_meters}m")
        
        # Rule 3: Combined - Loitering near restricted zone
        threat_level = 'low'
        if is_loitering:
            # Check if person is near a restricted zone
            near_restricted = False
            for zone_name, zone_data in self.zone_polygons.items():
                polygon = zone_data['polygon']
                point = Point(centroid[0], centroid[1])
                # Check if within 50 pixels of zone boundary
                distance_to_zone = polygon.exterior.distance(point)
                if distance_to_zone < 50:
                    near_restricted = True
                    threat_level = 'high'
                    loitering_reasons.append(f"Near restricted zone: {zone_name}")
                    break
            
            if not near_restricted:
                threat_level = 'medium'
            
            return {
                'person_id': person_id,
                'status': 'Loitering',
                'threat_level': threat_level,
                'reason': '; '.join(loitering_reasons),
                'zone_info': None
            }
        
        # Normal behavior
        return {
            'person_id': person_id,
            'status': 'Normal',
            'threat_level': 'low',
            'reason': 'Normal activity',
            'zone_info': None
        }
    
    def update_restricted_zones(self, restricted_zones: List[Dict]):
        """Update restricted zones configuration."""
        self.restricted_zones = restricted_zones
        self.zone_polygons = {}
        
        for zone in self.restricted_zones:
            name = zone.get('name', 'Unknown')
            polygon_coords = zone.get('polygon', [])
            if len(polygon_coords) >= 3:
                self.zone_polygons[name] = {
                    'polygon': Polygon(polygon_coords),
                    'threat_level': zone.get('threat_level', 'medium')
                }

