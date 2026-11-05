"""
Visualization utilities for drawing detections, trajectories, and alerts.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from shapely.geometry import Polygon


class Visualizer:
    """Handles visualization of detections, trajectories, and zones."""
    
    def __init__(self, show_pose_keypoints: bool = False,
                 show_trajectory: bool = True,
                 trajectory_length: int = 30,
                 colors: Dict[str, List[int]] = None):
        """
        Initialize Visualizer.
        
        Args:
            show_pose_keypoints: Whether to draw pose keypoints
            show_trajectory: Whether to draw trajectory paths
            trajectory_length: Number of trajectory points to display
            colors: Dictionary of color mappings
        """
        self.show_pose_keypoints = show_pose_keypoints
        self.show_trajectory = show_trajectory
        self.trajectory_length = trajectory_length
        
        # Default colors (BGR format for OpenCV)
        self.colors = colors or {
            'normal': [0, 255, 0],      # Green
            'loitering': [0, 165, 255],  # Orange
            'intrusion': [0, 0, 255],    # Red
            'restricted_zone': [255, 0, 255]  # Magenta
        }
    
    def draw_restricted_zones(self, frame: np.ndarray, 
                             restricted_zones: List[Dict],
                             alpha: float = 0.3):
        """
        Draw restricted zones on frame.
        
        Args:
            frame: Video frame
            restricted_zones: List of zone definitions
            alpha: Transparency for zone overlay
        """
        overlay = frame.copy()
        
        for zone in restricted_zones:
            polygon_coords = zone.get('polygon', [])
            if len(polygon_coords) >= 3:
                pts = np.array(polygon_coords, np.int32)
                pts = pts.reshape((-1, 1, 2))
                
                # Draw filled polygon
                cv2.fillPoly(overlay, [pts], self.colors['restricted_zone'])
                
                # Draw border
                cv2.polylines(overlay, [pts], True, self.colors['restricted_zone'], 2)
                
                # Add label
                if polygon_coords:
                    x, y = polygon_coords[0]
                    zone_name = zone.get('name', 'Restricted Zone')
                    cv2.putText(overlay, zone_name, (x, y - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Blend overlay
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    
    def draw_detection(self, frame: np.ndarray, detection: Dict,
                      decision_result: Optional[Dict] = None):
        """
        Draw bounding box and information for a detection.
        
        Args:
            frame: Video frame
            detection: Detection dictionary from PoseAgent
            decision_result: Decision result from DecisionAgent
        """
        person_id = detection['person_id']
        bbox = detection['bbox']
        x, y, w, h = bbox
        centroid = detection['centroid']
        
        # Determine color based on status
        if decision_result:
            status = decision_result['status']
            if status == 'Intrusion':
                color = self.colors['intrusion']
            elif status == 'Loitering':
                color = self.colors['loitering']
            else:
                color = self.colors['normal']
        else:
            color = self.colors['normal']
        
        # Draw bounding box
        cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
        
        # Draw centroid
        cv2.circle(frame, centroid, 5, color, -1)
        
        # Draw label
        label = f"Person {person_id}"
        if decision_result:
            label += f" - {decision_result['status']}"
            if decision_result['threat_level'] != 'low':
                label += f" ({decision_result['threat_level']})"
        
        # Background for text
        (text_width, text_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x, y - text_height - 10), 
                     (x + text_width, y), color, -1)
        
        # Text
        cv2.putText(frame, label, (x, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw pose keypoints if enabled
        if self.show_pose_keypoints and detection.get('keypoints'):
            self._draw_keypoints(frame, detection['keypoints'])
    
    def draw_trajectory(self, frame: np.ndarray, trajectory: List[Tuple[int, int, int]],
                       color: Tuple[int, int, int] = None):
        """
        Draw trajectory path on frame.
        
        Args:
            frame: Video frame
            trajectory: List of (x, y, timestamp) points
            color: Trajectory color (default: white)
        """
        if not trajectory or len(trajectory) < 2:
            return
        
        if color is None:
            color = (255, 255, 255)  # White
        
        # Get recent trajectory points
        recent_trajectory = trajectory[-self.trajectory_length:]
        
        # Draw lines connecting trajectory points
        for i in range(1, len(recent_trajectory)):
            x1, y1, _ = recent_trajectory[i-1]
            x2, y2, _ = recent_trajectory[i]
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        
        # Draw points
        for x, y, _ in recent_trajectory:
            cv2.circle(frame, (int(x), int(y)), 3, color, -1)
    
    def draw_info_panel(self, frame: np.ndarray, active_persons: int,
                       alerts: List[Dict] = None):
        """
        Draw information panel on frame.
        
        Args:
            frame: Video frame
            active_persons: Number of active persons
            alerts: List of recent alerts
        """
        h, w = frame.shape[:2]
        
        # Create semi-transparent panel
        panel_height = 100
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (w - 10, panel_height), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Draw text
        y_offset = 30
        cv2.putText(frame, f"Active Persons: {active_persons}", (20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        if alerts:
            y_offset += 25
            recent_alert = alerts[-1]
            alert_text = f"Last Alert: {recent_alert['status']} - Person {recent_alert['person_id']}"
            cv2.putText(frame, alert_text, (20, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors[recent_alert['status'].lower()], 2)
    
    def _draw_keypoints(self, frame: np.ndarray, keypoints: Dict):
        """Draw pose keypoints on frame."""
        if not keypoints or 'points' not in keypoints:
            return
        
        points = keypoints['points']
        confidences = keypoints.get('confidence', [])
        
        # Keypoint connections (simplified skeleton)
        connections = [
            (0, 1), (0, 2), (1, 3), (2, 4),  # Head to shoulders
            (5, 6), (5, 7), (6, 8), (7, 9), (8, 10),  # Arms
            (11, 12), (11, 13), (12, 14), (13, 15), (14, 16)  # Legs
        ]
        
        # Draw connections
        for start_idx, end_idx in connections:
            if (start_idx < len(points) and end_idx < len(points) and
                (not confidences or 
                 (confidences[start_idx] > 0.5 and confidences[end_idx] > 0.5))):
                pt1 = tuple(map(int, points[start_idx]))
                pt2 = tuple(map(int, points[end_idx]))
                cv2.line(frame, pt1, pt2, (0, 255, 255), 2)
        
        # Draw keypoints
        for i, point in enumerate(points):
            if not confidences or confidences[i] > 0.5:
                pt = tuple(map(int, point))
                cv2.circle(frame, pt, 4, (0, 255, 255), -1)

