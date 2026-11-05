"""
Main pipeline for Loitering & Restricted Area Intrusion Detection System.
Orchestrates all agents to process video streams.
"""

import cv2
import yaml
import numpy as np
import os
import time
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

from agents.pose_agent import PoseAgent
from agents.trajectory_agent import TrajectoryAgent
from agents.memory_agent import MemoryAgent
from agents.decision_agent import DecisionAgent
from agents.notification_agent import NotificationAgent
from utils.visualizer import Visualizer


class LoiteringDetectionSystem:
    """Main system orchestrating all agents."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Initialize the detection system.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize agents
        thresholds = self.config['thresholds']
        loitering_enabled = thresholds.get('enable_loitering_detection', True)

        pose_config = self.config['pose_detection']
        selected_model = (
            pose_config.get('loitering_on_model', 'yolov8n-pose.pt')
            if loitering_enabled
            else pose_config.get('loitering_off_model', 'yolov8n.pt')
        )
        self.pose_agent = PoseAgent(
            model_path=selected_model,
            confidence_threshold=pose_config['confidence_threshold'],
            device=pose_config['device']
        )
        
        self.trajectory_agent = TrajectoryAgent()
        self.memory_agent = MemoryAgent()
        
        self.decision_agent = DecisionAgent(
            loitering_duration_threshold=thresholds['loitering_duration_seconds'],
            min_movement_meters=thresholds['min_movement_meters'],
            enable_loitering_detection=loitering_enabled,
            restricted_zones=self.config['restricted_zones']
        )
        
        notif_config = self.config['notifications']
        self.alert_cooldown_frames = int(notif_config.get('alert_cooldown_frames', 30))
        self.person_alert_state = {}
        self.notification_agent = NotificationAgent(
            enable_email=notif_config.get('enable_email', False),
            enable_sms=notif_config.get('enable_sms', False),
            enable_whatsapp=notif_config.get('enable_whatsapp', False),
            enable_console=notif_config.get('enable_console', True),
            enable_visual=notif_config.get('enable_visual_alerts', True),
            email_recipients=notif_config.get('email_recipients', []),
            sms_recipients=notif_config.get('sms_recipients', []),
            whatsapp_recipients=notif_config.get('whatsapp_recipients', [])
        )
        
        # Initialize visualizer
        vis_config = self.config['visualization']
        self.enable_display = vis_config.get('enable_display', True)
        self.visualizer = Visualizer(
            show_pose_keypoints=vis_config['show_pose_keypoints'],
            show_trajectory=vis_config['show_trajectory'],
            trajectory_length=vis_config['trajectory_length'],
            colors=vis_config['colors']
        )
        
        # Video settings
        self.video_config = self.config['video']
        self.fps = self.video_config['fps']
        self.frame_width = int(self.video_config.get('frame_width', 640))
        self.frame_height = int(self.video_config.get('frame_height', 480))

    def _should_send_alert(self, person_id: int, decision_result: dict, frame_number: int) -> bool:
        """Rate-limit region alerts per person to avoid repeated spam."""
        status = decision_result.get('status')
        zone_info = decision_result.get('zone_info')
        current_zone = zone_info.get('zone_name') if zone_info else None

        state = self.person_alert_state.setdefault(
            person_id,
            {
                'last_alert_frame': None,
                'last_alert_zone': None,
                'left_after_last_alert': True
            }
        )

        # Apply the same per-person cooldown policy to loitering alerts.
        if status == 'Loitering':
            if state['last_alert_frame'] is None:
                return True
            return (frame_number - state['last_alert_frame']) >= self.alert_cooldown_frames

        # If person is currently outside all restricted zones, mark as left.
        if current_zone is None:
            state['left_after_last_alert'] = True
            return False

        # For intrusion, allow first alert immediately.
        if state['last_alert_frame'] is None:
            return True

        enough_cooldown = (frame_number - state['last_alert_frame']) >= self.alert_cooldown_frames
        revisited_or_new_zone = state['left_after_last_alert'] or (state['last_alert_zone'] != current_zone)
        return enough_cooldown and revisited_or_new_zone

    def _mark_alert_sent(self, person_id: int, decision_result: dict, frame_number: int):
        """Update per-person alert state after a notification is sent."""
        zone_info = decision_result.get('zone_info')
        zone_name = zone_info.get('zone_name') if zone_info else None
        state = self.person_alert_state.setdefault(
            person_id,
            {
                'last_alert_frame': None,
                'last_alert_zone': None,
                'left_after_last_alert': True
            }
        )
        state['last_alert_frame'] = frame_number
        state['last_alert_zone'] = zone_name
        state['left_after_last_alert'] = False
        
    def process_frame(self, frame: np.ndarray, frame_number: int) -> np.ndarray:
        """
        Process a single frame through the pipeline.
        
        Args:
            frame: Input video frame
            frame_number: Current frame number
            
        Returns:
            Annotated frame with detections and visualizations
        """
        # Step 1: PoseAgent - Detect and track humans
        detections = self.pose_agent.detect_and_track(frame, frame_number)
        
        # Step 2: Draw restricted zones
        if self.enable_display:
            self.visualizer.draw_restricted_zones(frame, self.config['restricted_zones'])
        
        # Step 3: Process each detection
        active_persons = []
        alerts = []
        
        for detection in detections:
            person_id = detection['person_id']
            centroid = detection['centroid']
            
            # Step 3a: TrajectoryAgent - Update trajectory
            trajectory_info = self.trajectory_agent.update_trajectory(
                person_id, centroid, frame_number, self.fps
            )
            
            # Step 3b: MemoryAgent - Update person state
            self.memory_agent.update_person_state(
                person_id, centroid, frame_number, self.fps
            )
            
            # Step 3c: Get person state
            person_state = self.memory_agent.get_person_state(person_id)
            if not person_state:
                continue
            
            dwell_duration = person_state['dwell_duration']
            distance_traveled = trajectory_info['distance_traveled']
            is_moving = trajectory_info['is_moving']
            
            # Step 3d: DecisionAgent - Evaluate behavior
            decision_result = self.decision_agent.evaluate_person(
                person_id=person_id,
                dwell_duration=dwell_duration,
                distance_traveled=distance_traveled,
                centroid=centroid,
                is_moving=is_moving
            )
            
            # Step 3e: NotificationAgent - Send alerts if needed
            if decision_result['status'] in ['Loitering', 'Intrusion'] and \
               self._should_send_alert(person_id, decision_result, frame_number):
                self.notification_agent.send_alert(
                    decision_result, frame, 
                    camera_id=self.video_config.get('camera_id', 'Camera#1')
                )
                self._mark_alert_sent(person_id, decision_result, frame_number)
                alerts.append({
                    'person_id': person_id,
                    'status': decision_result['status'],
                    'threat_level': decision_result['threat_level']
                })
            
            # Step 3f: Visualization
            if self.enable_display:
                self.visualizer.draw_detection(frame, detection, decision_result)
            
            # Draw trajectory if enabled
            if self.enable_display and self.visualizer.show_trajectory:
                trajectory = self.trajectory_agent.get_trajectory(person_id)
                if decision_result['status'] == 'Intrusion':
                    traj_color = tuple(self.visualizer.colors['intrusion'])
                elif decision_result['status'] == 'Loitering':
                    traj_color = tuple(self.visualizer.colors['loitering'])
                else:
                    traj_color = tuple(self.visualizer.colors['normal'])
                
                self.visualizer.draw_trajectory(frame, trajectory, traj_color)
            
            active_persons.append(person_id)
        
        # Step 4: Cleanup inactive persons
        self.pose_agent.clear_old_tracks(frame_number, 
                                         self.config['tracking']['max_age'])
        self.memory_agent.cleanup_inactive(
            frame_number,
            self.config['tracking']['max_age'],
            self.fps
        )
        
        # Step 5: Draw info panel
        if self.enable_display:
            self.visualizer.draw_info_panel(frame, len(active_persons), alerts)
        
        return frame
    
    def run(self, input_source, output_path: Optional[str] = None):
        """
        Run the detection system on video input.
        
        Args:
            input_source: Video file path or camera index
            output_path: Optional path to save output video
        """
        # Open video source
        if isinstance(input_source, (int, str)) and str(input_source).isdigit():
            cap = cv2.VideoCapture(int(input_source))
        else:
            cap = cv2.VideoCapture(input_source)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {input_source}")
        
        # Get video properties and cap processing to 15 FPS
        source_fps = int(cap.get(cv2.CAP_PROP_FPS)) or self.fps
        target_fps = 15
        processing_fps = min(source_fps, target_fps) if source_fps > 0 else target_fps
        self.fps = processing_fps
        min_process_interval = 1.0 / target_fps

        # Try to request 15 FPS on camera sources (not all backends honor this).
        if isinstance(input_source, (int, str)) and str(input_source).isdigit():
            cap.set(cv2.CAP_PROP_FPS, target_fps)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.frame_width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.frame_height)
        elif isinstance(input_source, str) and input_source.lower().startswith("rtsp://"):
            # Reduce buffering for lower-latency RTSP streams when backend supports it.
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        output_size = (self.frame_width, self.frame_height)
        
        # Setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, processing_fps, output_size)
        
        frame_number = 0
        
        print("Starting detection system...")
        if self.enable_display:
            print("Press 'q' to quit, 'p' to pause")
        else:
            print("Display disabled (headless mode). Press Ctrl+C to stop.")
        
        paused = False
        
        last_processed_time = 0.0
        try:
            while True:
                if not paused:
                    # Continuously grab frames, and only retrieve/process at target interval.
                    # This keeps inference load fixed while using the freshest frame.
                    grabbed = cap.grab()
                    if not grabbed:
                        break

                    now = time.monotonic()
                    if now - last_processed_time < min_process_interval:
                        continue
                    last_processed_time = now

                    ret, frame = cap.retrieve()
                    if not ret:
                        continue

                    # Normalize processing resolution for consistent inference behavior.
                    if frame.shape[1] != self.frame_width or frame.shape[0] != self.frame_height:
                        frame = cv2.resize(frame, (self.frame_width, self.frame_height), interpolation=cv2.INTER_LINEAR)
                    
                    frame_number += 1
                    
                    # Process frame
                    annotated_frame = self.process_frame(frame, frame_number)
                    
                    # Write frame if writer exists
                    if writer:
                        writer.write(annotated_frame)
                    
                    # Display frame
                    if self.enable_display:
                        cv2.imshow('Loitering & Intrusion Detection', annotated_frame)
                
                # Handle keyboard input
                if self.enable_display:
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        break
                    elif key == ord('p'):
                        paused = not paused
                        print("Paused" if paused else "Resumed")
        
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        
        finally:
            # Cleanup
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            print(f"\nProcessed {frame_number} frames")
            print(f"Total alerts: {len(self.notification_agent.alert_history)}")


def main():
    """Main entry point."""
    load_dotenv()

    # Runtime configuration is sourced from environment variables:
    # INPUT_SOURCE, OUTPUT_PATH, CONFIG_PATH.
    input_source = os.getenv('INPUT_SOURCE', '0')
    output_path = os.getenv('OUTPUT_PATH') or None
    config_path = os.getenv('CONFIG_PATH', 'config/config.yaml')

    # Initialize system
    system = LoiteringDetectionSystem(config_path=config_path)
    
    # Run detection
    system.run(input_source, output_path)


if __name__ == "__main__":
    main()

