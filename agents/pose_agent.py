"""
PoseAgent: Detects and tracks humans in video frames using pose estimation.
Uses YOLOv8-Pose for detection and ByteTrack/DeepSORT for tracking.
"""

import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from ultralytics import YOLO
from collections import defaultdict


class PoseAgent:
    """Agent responsible for human detection and pose estimation."""
    
    def __init__(self, model_path: str = "yolov8n-pose.pt", 
                 confidence_threshold: float = 0.5,
                 device: str = "cpu"):
        """
        Initialize PoseAgent.
        
        Args:
            model_path: Path to YOLOv8 pose model
            confidence_threshold: Minimum confidence for detections
            device: Device to run inference on (cpu, cuda, mps)
        """
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.track_history = defaultdict(list)
        
    def detect_and_track(self, frame: np.ndarray, frame_number: int) -> List[Dict]:
        """
        Detect humans and track them across frames.
        
        Args:
            frame: Input video frame (BGR format)
            frame_number: Current frame number
            
        Returns:
            List of detections, each containing:
            {
                'person_id': int,
                'bbox': [x, y, w, h],
                'centroid': (x, y),
                'keypoints': List of keypoint coordinates,
                'confidence': float,
                'timestamp': frame_number
            }
        """
        # Run YOLOv8 pose estimation with tracking
        results = self.model.track(
            frame,
            persist=True,
            conf=self.confidence_threshold,
            device=self.device,
            classes=[0], # Only detect person class (class 0 in COCO)
            verbose=False
        )
        
        detections = []
        
        if results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes
            keypoints = results[0].keypoints
            
            for i in range(len(boxes)):
                # Get bounding box
                box = boxes.xyxy[i].cpu().numpy()
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                
                # Get tracking ID
                if boxes.id is not None:
                    person_id = int(boxes.id[i].cpu().numpy())
                else:
                    # Fallback: use index if tracking not available
                    person_id = i
                
                # Track a lower-center point (near feet) instead of box center.
                foot_x = int((x1 + x2) / 2)
                foot_y = int(y2 - 0.05 * h)  # Slightly above bottom edge to approximate toes
                centroid = (foot_x, foot_y)
                
                # Get keypoints if available
                keypoint_data = None
                if keypoints is not None and len(keypoints) > 0:
                    kp = keypoints.xy[i].cpu().numpy()
                    conf = keypoints.conf[i].cpu().numpy() if keypoints.conf is not None else None
                    keypoint_data = {
                        'points': kp.tolist(),
                        'confidence': conf.tolist() if conf is not None else None
                    }
                
                # Get confidence
                confidence = float(boxes.conf[i].cpu().numpy())
                
                detection = {
                    'person_id': person_id,
                    'bbox': [int(x1), int(y1), int(w), int(h)],
                    'centroid': centroid,
                    'keypoints': keypoint_data,
                    'confidence': confidence,
                    'timestamp': frame_number
                }
                
                detections.append(detection)
                
                # Store tracking history
                self.track_history[person_id].append({
                    'centroid': centroid,
                    'frame': frame_number
                })
        
        return detections
    
    def get_track_history(self, person_id: int) -> List[Dict]:
        """Get tracking history for a specific person."""
        return self.track_history.get(person_id, [])
    
    def clear_old_tracks(self, current_frame: int, max_age: int = 30):
        """Remove tracks that haven't been seen for max_age frames."""
        to_remove = []
        for person_id, history in self.track_history.items():
            if history and current_frame - history[-1]['frame'] > max_age:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            del self.track_history[person_id]

