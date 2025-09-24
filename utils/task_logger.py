"""
Task Logger for AIM/PROGRESS Notes

This module provides utilities for logging AIM notes at the start of the day
and PROGRESS notes at the end of the day, as required by the BHIV team process.
"""

import os
import json
from datetime import datetime
from typing import Dict, Any, Optional
from utils.logger import get_logger

logger = get_logger(__name__)

class TaskLogger:
    """Utility class for logging AIM and PROGRESS notes."""
    
    def __init__(self, log_file: str = "logs/task_log.json"):
        self.log_file = log_file
        self._ensure_log_directory()
    
    def _ensure_log_directory(self):
        """Ensure the log directory exists."""
        log_dir = os.path.dirname(self.log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
    
    def log_aim(self, task_description: str, goals: list, resources: list, 
                challenges: list, notes: str = "") -> Dict[str, Any]:
        """
        Log AIM note at the start of the day.
        
        Args:
            task_description: Description of the main task
            goals: List of goals to achieve
            resources: List of resources available
            challenges: List of anticipated challenges
            notes: Additional notes
            
        Returns:
            Dictionary with log entry details
        """
        timestamp = datetime.now().isoformat()
        aim_entry = {
            "type": "AIM",
            "timestamp": timestamp,
            "task_description": task_description,
            "goals": goals,
            "resources": resources,
            "challenges": challenges,
            "notes": notes
        }
        
        self._write_log_entry(aim_entry)
        logger.info(f"AIM note logged: {task_description}")
        return aim_entry
    
    def log_progress(self, tasks_completed: list, tasks_failed: list, 
                     lessons_learned: list, grateful_for: list, notes: str = "") -> Dict[str, Any]:
        """
        Log PROGRESS note at the end of the day.
        
        Args:
            tasks_completed: List of tasks completed
            tasks_failed: List of tasks that failed
            lessons_learned: List of lessons learned
            grateful_for: List of things grateful for
            notes: Additional notes
            
        Returns:
            Dictionary with log entry details
        """
        timestamp = datetime.now().isoformat()
        progress_entry = {
            "type": "PROGRESS",
            "timestamp": timestamp,
            "tasks_completed": tasks_completed,
            "tasks_failed": tasks_failed,
            "lessons_learned": lessons_learned,
            "grateful_for": grateful_for,
            "notes": notes
        }
        
        self._write_log_entry(progress_entry)
        logger.info(f"PROGRESS note logged with {len(tasks_completed)} completed tasks")
        return progress_entry
    
    def _write_log_entry(self, entry: Dict[str, Any]):
        """
        Write a log entry to the log file.
        
        Args:
            entry: Dictionary with log entry data
        """
        # Read existing entries
        entries = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    entries = json.load(f)
            except Exception as e:
                logger.warning(f"Error reading existing log file: {e}")
        
        # Add new entry
        entries.append(entry)
        
        # Write back to file
        try:
            with open(self.log_file, 'w') as f:
                json.dump(entries, f, indent=2)
        except Exception as e:
            logger.error(f"Error writing to log file: {e}")
    
    def get_daily_summary(self, date: Optional[str] = None) -> Dict[str, Any]:
        """
        Get daily summary of AIM and PROGRESS notes.
        
        Args:
            date: Date to get summary for (YYYY-MM-DD format), defaults to today
            
        Returns:
            Dictionary with daily summary
        """
        if date is None:
            date = datetime.now().strftime("%Y-%m-%d")
        
        entries = []
        if os.path.exists(self.log_file):
            try:
                with open(self.log_file, 'r') as f:
                    all_entries = json.load(f)
                
                # Filter entries for the specified date
                entries = [
                    entry for entry in all_entries 
                    if entry.get('timestamp', '').startswith(date)
                ]
            except Exception as e:
                logger.warning(f"Error reading log file: {e}")
        
        aim_entries = [e for e in entries if e.get('type') == 'AIM']
        progress_entries = [e for e in entries if e.get('type') == 'PROGRESS']
        
        return {
            "date": date,
            "aim_notes": aim_entries,
            "progress_notes": progress_entries,
            "total_entries": len(entries)
        }

# Global task logger instance
task_logger = TaskLogger()

def log_aim(task_description: str, goals: list, resources: list, 
            challenges: list, notes: str = "") -> Dict[str, Any]:
    """Convenience function to log AIM note."""
    return task_logger.log_aim(task_description, goals, resources, challenges, notes)

def log_progress(tasks_completed: list, tasks_failed: list, 
                 lessons_learned: list, grateful_for: list, notes: str = "") -> Dict[str, Any]:
    """Convenience function to log PROGRESS note."""
    return task_logger.log_progress(tasks_completed, tasks_failed, lessons_learned, grateful_for, notes)

def get_daily_summary(date: Optional[str] = None) -> Dict[str, Any]:
    """Convenience function to get daily summary."""
    return task_logger.get_daily_summary(date)

if __name__ == "__main__":
    # Example usage
    print("Testing task logger...")
    
    # Log AIM note
    aim_entry = log_aim(
        task_description="Implement BHIV Core Integration",
        goals=[
            "Define standard agent interface",
            "Create orchestration layer",
            "Write documentation"
        ],
        resources=[
            "Existing agent codebase",
            "Team documentation",
            "Python development environment"
        ],
        challenges=[
            "Ensuring compatibility with existing agents",
            "Handling different input/output formats",
            "Integration with RL system"
        ],
        notes="Starting with text and audio agents as examples"
    )
    print(f"AIM entry: {aim_entry}")
    
    # Log PROGRESS note
    progress_entry = log_progress(
        tasks_completed=[
            "Defined standard agent interface",
            "Updated text agent to follow interface",
            "Created orchestration layer"
        ],
        tasks_failed=[
            "Some integration tests failed due to missing services"
        ],
        lessons_learned=[
            "Standard interfaces improve code maintainability",
            "Orchestration layer needs better error handling"
        ],
        grateful_for=[
            "Team collaboration on design decisions",
            "Existing codebase structure"
        ],
        notes="Ready for team review"
    )
    print(f"PROGRESS entry: {progress_entry}")
    
    # Get daily summary
    summary = get_daily_summary()
    print(f"Daily summary: {summary}")