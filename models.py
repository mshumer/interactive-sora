from __future__ import annotations

import enum
from datetime import datetime
from typing import List, Optional

from sqlalchemy import (
    JSON,
    Column,
    DateTime,
    Enum,
    Integer,
    String,
    Text,
    UniqueConstraint,
)

from database import Base


class SceneStatus(str, enum.Enum):
    PENDING = "pending"
    QUEUED = "queued"
    READY = "ready"
    FAILED = "failed"


JSON_TYPE = JSON


class Scene(Base):
    __tablename__ = "scenes"
    __table_args__ = (
        UniqueConstraint("world_id", "path", name="uq_scene_world_path"),
    )

    id = Column(Integer, primary_key=True, autoincrement=True)
    world_id = Column(String(128), nullable=False, index=True)
    path = Column(String(512), nullable=False)
    depth = Column(Integer, nullable=False, default=0)
    status = Column(Enum(SceneStatus), nullable=False, default=SceneStatus.PENDING)
    scenario_display = Column(Text, nullable=True)
    sora_prompt = Column(Text, nullable=True)
    choices = Column(JSON_TYPE, nullable=True)
    trigger_choice = Column(Text, nullable=True)
    video_url = Column(Text, nullable=True)
    poster_url = Column(Text, nullable=True)
    video_seconds = Column(Integer, nullable=True)
    planner_model = Column(String(128), nullable=True)
    planner_raw = Column(JSON_TYPE, nullable=True)
    failure_code = Column(String(128), nullable=True)
    failure_detail = Column(Text, nullable=True)
    contributor_hash = Column(String(256), nullable=True)
    started_at = Column(DateTime(timezone=True), nullable=True)
    state_summary = Column(Text, nullable=True)
    progress = Column(Integer, nullable=True)
    progress_updated_at = Column(DateTime(timezone=True), nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
    )

    def child_path(self, index: int) -> str:
        if not self.path:
            return str(index)
        return f"{self.path}/{index}"


class SceneMetric(Base):
    __tablename__ = "scene_metrics"

    id = Column(Integer, primary_key=True, autoincrement=True)
    scene_id = Column(Integer, nullable=False, index=True)
    rendered = Column(Integer, nullable=False, default=0)
    render_time_ms = Column(Integer, nullable=True)
    storage_bytes = Column(Integer, nullable=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=datetime.utcnow)


def split_path(path: str) -> List[int]:
    if not path:
        return []
    return [int(part) for part in path.split("/") if part != ""]


def compute_depth(path: str) -> int:
    return len(split_path(path))


def parent_path(path: str) -> Optional[str]:
    parts = split_path(path)
    if not parts:
        return None
    return "/".join(str(part) for part in parts[:-1])


def last_choice_index(path: str) -> Optional[int]:
    parts = split_path(path)
    if not parts:
        return None
    return parts[-1]
