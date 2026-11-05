"""
MemoryAgent: Stores and maintains movement history for each person.
Acts as short-term memory for loitering detection.
"""

from typing import Dict, Optional
from collections import defaultdict
from datetime import datetime, timedelta
import time


class MemoryAgent:
    """Agent responsible for maintaining person state and history."""
    
    def __init__(self):
        """Initialize MemoryAgent."""
        self.person_states = defaultdict(dict)  # person_id -> state dict
        self.current_time = None
        
    def update_person_state(self, person_id: int, centroid: tuple, 
                           timestamp: int, fps: float = 30.0):
        """
        Update state for a person.
        
        Args:
            person_id: Unique identifier for the person
            centroid: Current (x, y) position
            timestamp: Frame number or timestamp
            fps: Frames per second
        """
        if person_id not in self.person_states:
            # First time seeing this person
            self.person_states[person_id] = {
                'entry_time': timestamp / fps,  # Convert to seconds
                'entry_frame': timestamp,
                'first_position': centroid,
                'last_seen': timestamp / fps,
                'last_frame': timestamp,
                'last_position': centroid,
                'dwell_duration': 0.0,
                'is_active': True
            }
        else:
            # Update existing person
            state = self.person_states[person_id]
            state['last_seen'] = timestamp / fps
            state['last_frame'] = timestamp
            state['last_position'] = centroid
            state['dwell_duration'] = (timestamp / fps) - state['entry_time']
            state['is_active'] = True
    
    def get_person_state(self, person_id: int) -> Optional[Dict]:
        """
        Get current state for a person.
        
        Returns:
            Dictionary with state information:
            {
                'entry_time': float (seconds),
                'entry_frame': int,
                'first_position': (x, y),
                'last_seen': float (seconds),
                'last_frame': int,
                'last_position': (x, y),
                'dwell_duration': float (seconds),
                'is_active': bool
            }
        """
        return self.person_states.get(person_id)
    
    def get_dwell_duration(self, person_id: int) -> float:
        """Get dwell duration for a person in seconds."""
        state = self.get_person_state(person_id)
        if state:
            return state['dwell_duration']
        return 0.0
    
    def mark_inactive(self, person_id: int):
        """Mark a person as inactive (no longer detected)."""
        if person_id in self.person_states:
            self.person_states[person_id]['is_active'] = False
    
    def remove_person(self, person_id: int):
        """Remove person from memory."""
        if person_id in self.person_states:
            del self.person_states[person_id]
    
    def cleanup_inactive(self, current_frame: int, max_age_frames: int = 30, fps: float = 30.0):
        """
        Remove persons that haven't been seen for max_age_frames.
        
        Args:
            current_frame: Current frame number
            max_age_frames: Maximum frames to keep inactive persons
            fps: Frames per second
        """
        to_remove = []
        for person_id, state in self.person_states.items():
            if not state.get('is_active', False):
                frames_since_last_seen = current_frame - state.get('last_frame', 0)
                if frames_since_last_seen > max_age_frames:
                    to_remove.append(person_id)
        
        for person_id in to_remove:
            self.remove_person(person_id)
    
    def get_all_active_persons(self) -> Dict[int, Dict]:
        """Get all currently active persons."""
        return {pid: state for pid, state in self.person_states.items() 
                if state.get('is_active', False)}
    
    def get_all_persons(self) -> Dict[int, Dict]:
        """Get all persons in memory."""
        return dict(self.person_states)

