"""
Checkpoint management for resumable dataset preprocessing.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Optional, Dict, Any, List


class CheckpointManager:
    """Manages reading and writing checkpoint state to disk."""

    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = checkpoint_path

    def exists(self) -> bool:
        """Check if a checkpoint file exists."""
        return os.path.exists(self.checkpoint_path)

    def load(self) -> Optional[Dict[str, Any]]:
        """Load checkpoint state if available."""
        if not self.exists():
            return None
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None

    def save(
        self,
        input_file: str,
        source: str,
        rows_processed: int,
        current_part: int,
        part_files: List[str],
        status: str = "in_progress",
        error_count: int = 0,
    ) -> None:
        """Persist checkpoint state atomically."""
        state = {
            "input_file": input_file,
            "source": source,
            "rows_processed": rows_processed,
            "current_part": current_part,
            "part_files": part_files,
            "status": status,
            "error_count": error_count,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        os.makedirs(os.path.dirname(os.path.abspath(self.checkpoint_path)), exist_ok=True)
        temp_path = f"{self.checkpoint_path}.tmp"
        with open(temp_path, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2)
        os.replace(temp_path, self.checkpoint_path)

    def mark_completed(self, rows_processed: int, current_part: int, part_files: List[str], error_count: int = 0) -> None:
        """Update checkpoint status to completed."""
        state = self.load() or {}
        input_file = state.get("input_file", "")
        source = state.get("source", "")
        self.save(
            input_file=input_file,
            source=source,
            rows_processed=rows_processed,
            current_part=current_part,
            part_files=part_files,
            status="completed",
            error_count=error_count,
        )
