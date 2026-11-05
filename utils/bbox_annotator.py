"""
Bounding Box Annotator: Interactive tool to draw polygons on video frames
by clicking vertices. Saves coordinates to a text file.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple
import argparse
import yaml


class BBoxAnnotator:
    """Interactive tool for drawing polygons by clicking vertices on video frames."""
    
    def __init__(self, video_path: str, output_file: str = "polygon_coordinates.txt",
                 config_path: str = "config/config.yaml"):
        """
        Initialize the annotator.
        
        Args:
            video_path: Path to input video file
            output_file: Path to save polygon coordinates (TXT file)
            config_path: Path to system config file for target frame size
        """
        self.video_path = video_path
        self.output_file = output_file
        self.config_path = config_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.source_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.source_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.width = self.source_width
        self.height = self.source_height

        # Load target size from config if available.
        self._load_target_size_from_config()
        
        # Polygon points (vertices)
        self.polygon_points = []
        
        # Load first frame
        ret, self.frame = self.cap.read()
        if not ret:
            raise ValueError("Could not read first frame from video")

        # Match annotation frame size with main pipeline preprocessing size.
        if (self.frame.shape[1], self.frame.shape[0]) != (self.width, self.height):
            self.frame = cv2.resize(self.frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        
        # Window name
        self.window_name = "Polygon Annotator - Click vertices, Press ENTER to save, 'c' to clear, 'q' to quit"
        
        print(f"Video loaded: source {self.source_width}x{self.source_height}, "
              f"annotation {self.width}x{self.height}, {self.total_frames} frames, {self.fps} fps")
        print("\nInstructions:")
        print("  1. Click on the frame to add polygon vertices")
        print("  2. Each click adds a new vertex")
        print("  3. Press ENTER to save the polygon coordinates")
        print("  4. Press 'c' to clear and start over")
        print("  5. Press 'q' to quit")
    
    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse click events to add polygon vertices."""
        if event == cv2.EVENT_LBUTTONDOWN:
            # Add vertex to polygon
            self.polygon_points.append((x, y))
            print(f"Vertex {len(self.polygon_points)} added at ({x}, {y})")
    
    def draw_polygon(self, frame: np.ndarray) -> np.ndarray:
        """Draw the current polygon being created on the frame."""
        display_frame = frame.copy()
        
        if len(self.polygon_points) > 0:
            # Draw vertices as circles
            for i, point in enumerate(self.polygon_points):
                cv2.circle(display_frame, point, 5, (0, 255, 0), -1)
                # Draw vertex number
                cv2.putText(display_frame, str(i + 1), (point[0] + 10, point[1] - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw lines connecting vertices
            if len(self.polygon_points) > 1:
                for i in range(len(self.polygon_points) - 1):
                    cv2.line(display_frame, self.polygon_points[i], 
                            self.polygon_points[i + 1], (0, 255, 0), 2)
            
            # Draw closing line if we have at least 3 points
            if len(self.polygon_points) >= 3:
                cv2.line(display_frame, self.polygon_points[-1], 
                        self.polygon_points[0], (0, 255, 0), 2)
        
        # Draw info overlay
        info_text = [
            f"Vertices: {len(self.polygon_points)}",
            "Click to add vertices, ENTER to save, 'c' to clear, 'q' to quit"
        ]
        
        y_offset = 30
        for text in info_text:
            # Draw background for text
            (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(display_frame, (10, y_offset - text_height - 5), 
                         (10 + text_width + 10, y_offset + 5), (0, 0, 0), -1)
            cv2.putText(display_frame, text, (15, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            y_offset += 30
        
        return display_frame
    
    def save_polygon(self):
        """Save the polygon coordinates to a text file as a nested list."""
        if len(self.polygon_points) < 3:
            print("Error: Need at least 3 vertices to form a polygon")
            return
        
        # Convert to nested list format
        nested_list = [[x, y] for x, y in self.polygon_points]
        
        # Save to text file
        with open(self.output_file, 'w') as f:
            f.write(f"# Polygon coordinates from video: {self.video_path}\n")
            f.write(f"# Video dimensions: {self.width}x{self.height}\n")
            f.write(f"# Number of vertices: {len(self.polygon_points)}\n")
            f.write("# Nested list format: [[x1, y1], [x2, y2], ...]\n\n")
            
            # Write the nested list
            f.write(str(nested_list))
            f.write("\n")
        
        print(f"\n✓ Polygon saved to {self.output_file}")
        print(f"  Total vertices: {len(self.polygon_points)}")
        print(f"  Coordinates (nested list): {nested_list}")

    def _load_target_size_from_config(self):
        """Load annotation target width/height from config video settings."""
        try:
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f) or {}
            video_cfg = config.get('video', {})
            cfg_w = int(video_cfg.get('frame_width', self.width))
            cfg_h = int(video_cfg.get('frame_height', self.height))
            if cfg_w > 0 and cfg_h > 0:
                self.width = cfg_w
                self.height = cfg_h
        except Exception:
            # Keep source resolution if config cannot be read.
            pass
    
    def clear_polygon(self):
        """Clear current polygon being drawn."""
        self.polygon_points = []
        print("Polygon cleared. Start clicking to create a new polygon.")
    
    def run(self):
        """Run the annotation tool."""
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        print("\nStarting annotation tool...")
        print("Click on the frame to add polygon vertices")
        
        while True:
            display_frame = self.draw_polygon(self.frame)
            cv2.imshow(self.window_name, display_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # Ask to save before quitting
                if len(self.polygon_points) >= 3:
                    save = input("\nSave polygon before quitting? (y/n): ").strip().lower()
                    if save == 'y':
                        self.save_polygon()
                break
            elif key == 13 or key == 10:  # ENTER key
                self.save_polygon()
                # Ask if user wants to create another polygon
                another = input("Create another polygon? (y/n): ").strip().lower()
                if another == 'y':
                    self.clear_polygon()
                else:
                    break
            elif key == ord('c'):
                self.clear_polygon()
        
        # Cleanup
        self.cap.release()
        cv2.destroyAllWindows()
        print("\nAnnotation tool closed.")


def main():
    """Main entry point for the annotation tool."""
    parser = argparse.ArgumentParser(
        description='Interactive tool to draw polygons on video frames by clicking vertices'
    )
    parser.add_argument('video', type=str, help='Path to input video file')
    parser.add_argument('--output', '-o', type=str, default='polygon_coordinates.txt',
                       help='Output text file path (default: polygon_coordinates.txt)')
    parser.add_argument('--config', type=str, default='config/config.yaml',
                       help='Path to config file for target frame size')
    
    args = parser.parse_args()
    
    try:
        annotator = BBoxAnnotator(args.video, args.output, args.config)
        annotator.run()
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

