from __future__ import annotations

import json
import logging
import mimetypes
import os
import re
import subprocess
import tempfile
import threading
import time
from urllib.parse import urlparse
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from database import init_db, session_scope
from models import (
    Scene,
    SceneMetric,
    SceneStatus,
    compute_depth,
    last_choice_index,
    parent_path,
    split_path,
)
from storage import LocalStorageClient, StoredAsset, build_storage_client

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover
    cv2 = None

try:
    import imageio_ffmpeg  # type: ignore

    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover
    FFMPEG_BIN = None

APP_TITLE = "Sora Shared World API"

DEFAULT_SECONDS = 8

OPENAI_API_BASE = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")
SORA_VIDEOS_ENDPOINT = f"{OPENAI_API_BASE}/videos"
RESPONSES_ENDPOINT = f"{OPENAI_API_BASE}/responses"

WORLD_ID = os.environ.get("WORLD_ID", "default")

DEFAULT_WORLD_BASE_PROMPT = (
    "A multiverse adventure where shimmering portals splice iconic game-inspired realms together. "
    "Our protagonist is the Courier, an agile dimension runner collecting chronoglyph shards that stabilize reality. "
    "We begin inside the Nexus Gate, a concentric chamber housing three unlabelled portals:"
    " one amber-glass gateway echoing with synth bass, one cerulean rune vortex crackling with arcane energy, and one obsidian aperture breathing neon steam."
    " Each portal deposits the Courier into a distinct world—neon Vice City highways, rune-lit gothic battlegrounds, clockwork fantasy metropolises—"
    "all remixing familiar vibes without naming trademarks. The Courier is guided by an AI companion, Luma, who tracks shard resonance."
    " The stakes: close the Cataclysm Rift by assembling three legendary relics hidden across worlds;"
    " each scene should propel the chase, reveal cross-world cause-and-effect, or introduce allies/enemies reacting to the Courier's interference."
)

BASE_PROMPT = os.environ.get("WORLD_BASE_PROMPT", DEFAULT_WORLD_BASE_PROMPT)
PLANNER_MODEL = os.environ.get("PLANNER_MODEL", "gpt-5")
SORA_MODEL = os.environ.get("SORA_MODEL", "sora-2")
VIDEO_SIZE = os.environ.get("VIDEO_SIZE", "1280x720")
SCENE_TIMEOUT_SECONDS = int(os.environ.get("SCENE_TIMEOUT_SECONDS", "900"))
WATCHDOG_INTERVAL_SECONDS = int(os.environ.get("WATCHDOG_INTERVAL_SECONDS", "60"))
CONTRIBUTOR_SALT = os.environ.get("CONTRIBUTOR_SALT", "sora-shared-world")

DEFAULT_PROMPT_GUIDANCE = (
    "\n".join(
        [
            "Tone: Cinematic, high-energy multiverse heist. Every shot should highlight a recognizable-but-remixed world feature (vehicles, creatures, tech).",
            "Portals: Visualise swirling anomalies linking worlds. If a portal appears, show its activation, traversal, or aftermath in the same shot.",
            "Momentum: Show large movements—dashing, driving, grappling, spell bursts—rather than static observation.",
            "Quest Focus: We are chasing chronoglyph shards and the Cataclysm Rift. Each scene should reveal progress, a clue, or a complication tied to that quest.",
            "Allies & Foes: Introduce colorful companions or antagonists from different worlds reacting to portals; show how their abilities influence the beat.",
            "Hook: End with a striking twist (new world glimpsed, portal destabilising, relic reacting) that makes the next choice consequential.",
            "Portal Hub: When at the Nexus Gate, surface three distinct unlabeled portals as the primary choices; once a world is entered, momentum should push forward until the player restarts.",
        ]
    )
)

PROMPT_GUIDANCE = os.environ.get("WORLD_PROMPT_GUIDANCE", "").strip() or DEFAULT_PROMPT_GUIDANCE
STATE_SUMMARY_MODEL = os.environ.get("STATE_SUMMARY_MODEL", "gpt-5-mini").strip()

VIDEO_DIR = Path("sora_cyoa_videos")
FRAME_DIR = Path("sora_cyoa_frames")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)

storage_client = build_storage_client()
logger = logging.getLogger("sora_shared_world")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
level = logging.getLevelName(os.environ.get("LOG_LEVEL", "INFO"))
logger.setLevel(level if isinstance(level, int) else logging.INFO)

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

if isinstance(storage_client, LocalStorageClient):
    app.mount("/storage", StaticFiles(directory=storage_client.base_dir), name="storage")


class SceneResponse(BaseModel):
    world_id: str = Field(..., alias="worldId")
    path: str
    depth: int
    status: str
    scenario_display: Optional[str] = Field(None, alias="scenarioDisplay")
    sora_prompt: Optional[str] = Field(None, alias="soraPrompt")
    trigger_choice: Optional[str] = Field(None, alias="triggerChoice")
    choices: List[str] = Field(default_factory=list)
    choices_status: List[str] = Field(default_factory=list, alias="choicesStatus")
    children_paths: List[str] = Field(default_factory=list, alias="childrenPaths")
    video_url: Optional[str] = Field(None, alias="videoUrl")
    poster_url: Optional[str] = Field(None, alias="posterUrl")
    failure_code: Optional[str] = Field(None, alias="failureCode")
    failure_detail: Optional[str] = Field(None, alias="failureDetail")
    queued_since: Optional[datetime] = Field(None, alias="queuedSince")
    updated_at: Optional[datetime] = Field(None, alias="updatedAt")
    progress: Optional[int] = None
    progress_updated_at: Optional[datetime] = Field(None, alias="progressUpdatedAt")
    state_summary: Optional[str] = Field(None, alias="stateSummary")


class SceneGenerationRequest(BaseModel):
    path: str = ""
    api_key: str = Field(..., alias="apiKey")

    @validator("path")
    def validate_path(cls, value: str) -> str:
        if value == "":
            return ""
        if not re.fullmatch(r"(\d+)(/\d+)*", value):
            raise ValueError("path must be slash-separated numeric indexes, e.g. '0/1'")
        return value


class WorldResponse(BaseModel):
    world_id: str = Field(..., alias="worldId")
    base_prompt: str = Field(..., alias="basePrompt")
    planner_model: str = Field(..., alias="plannerModel")
    sora_model: str = Field(..., alias="soraModel")
    video_size: str = Field(..., alias="videoSize")


class WorldMetricsResponse(BaseModel):
    world_id: str = Field(..., alias="worldId")
    scene_count: int = Field(..., alias="sceneCount")
    ready_count: int = Field(..., alias="readyCount")
    queued_count: int = Field(..., alias="queuedCount")
    failed_count: int = Field(..., alias="failedCount")
    storage_bytes: int = Field(..., alias="storageBytes")
    success_rate: float = Field(..., alias="successRate")


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def normalize_seconds(secs: int) -> int:
    allowed = (4, 8, 12)
    return min(allowed, key=lambda value: abs(value - int(secs)))


@dataclass
class GenerationHandle:
    world_id: str
    path: str
    cancel_event: threading.Event
    thread: threading.Thread


_RUNNING_GENERATIONS: Dict[Tuple[str, str], GenerationHandle] = {}
_RUN_LOCK = threading.Lock()


def register_generation(handle: GenerationHandle) -> None:
    with _RUN_LOCK:
        _RUNNING_GENERATIONS[(handle.world_id, handle.path)] = handle


def clear_generation(world_id: str, path: str) -> None:
    with _RUN_LOCK:
        _RUNNING_GENERATIONS.pop((world_id, path), None)


def request_cancel(world_id: str, path: str) -> None:
    with _RUN_LOCK:
        handle = _RUNNING_GENERATIONS.get((world_id, path))
        if handle:
            handle.cancel_event.set()


@app.on_event("startup")
def on_startup() -> None:
    init_db()
    threading.Thread(target=_timeout_watchdog, daemon=True).start()


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/worlds/{world_id}", response_model=WorldResponse)
def get_world(world_id: str) -> WorldResponse:
    if world_id != WORLD_ID:
        raise HTTPException(status_code=404, detail="World not found")
    return WorldResponse(
        worldId=WORLD_ID,
        basePrompt=BASE_PROMPT,
        plannerModel=PLANNER_MODEL,
        soraModel=SORA_MODEL,
        videoSize=VIDEO_SIZE,
    )


@app.get("/worlds/{world_id}/metrics", response_model=WorldMetricsResponse)
def get_world_metrics(world_id: str) -> WorldMetricsResponse:
    if world_id != WORLD_ID:
        raise HTTPException(status_code=404, detail="World not found")

    with session_scope() as session:
        total = session.execute(
            select(func.count()).where(Scene.world_id == world_id)
        ).scalar_one()
        ready = session.execute(
            select(func.count()).where(Scene.world_id == world_id, Scene.status == SceneStatus.READY)
        ).scalar_one()
        queued = session.execute(
            select(func.count()).where(Scene.world_id == world_id, Scene.status == SceneStatus.QUEUED)
        ).scalar_one()
        failed = session.execute(
            select(func.count()).where(Scene.world_id == world_id, Scene.status == SceneStatus.FAILED)
        ).scalar_one()
        storage_bytes = session.execute(
            select(func.coalesce(func.sum(SceneMetric.storage_bytes), 0))
            .join(Scene, SceneMetric.scene_id == Scene.id)
            .where(Scene.world_id == world_id)
        ).scalar_one()

    success_denominator = max(ready + failed, 1)
    success_rate = ready / success_denominator

    return WorldMetricsResponse(
        worldId=world_id,
        sceneCount=total,
        readyCount=ready,
        queuedCount=queued,
        failedCount=failed,
        storageBytes=int(storage_bytes or 0),
        successRate=success_rate,
    )


@app.get("/worlds/{world_id}/scenes", response_model=SceneResponse)
def get_scene(world_id: str, path: str = Query("")) -> SceneResponse:
    if world_id != WORLD_ID:
        raise HTTPException(status_code=404, detail="World not found")

    with session_scope() as session:
        scene = ensure_scene_exists(session, world_id, path)
        return build_scene_response(session, scene)


@app.post("/worlds/{world_id}/scenes", response_model=SceneResponse)
def generate_scene_endpoint(world_id: str, payload: SceneGenerationRequest) -> SceneResponse:
    if world_id != WORLD_ID:
        raise HTTPException(status_code=404, detail="World not found")

    path = payload.path or ""
    api_key = payload.api_key.strip()
    if not api_key:
        raise HTTPException(status_code=400, detail="API key required for generation")

    should_start = False
    with session_scope() as session:
        scene = ensure_scene_exists(session, world_id, path)
        if scene.status == SceneStatus.READY:
            logger.info("scene already ready world=%s path=%s", world_id, path or "root")
            return build_scene_response(session, scene)
        if scene.status == SceneStatus.QUEUED:
            logger.info("scene already queued world=%s path=%s", world_id, path or "root")
            return build_scene_response(session, scene)

        # pending or failed
        logger.info("scene claim queued world=%s path=%s", world_id, path or "root")
        scene.status = SceneStatus.QUEUED
        scene.failure_code = None
        scene.failure_detail = None
        scene.started_at = utcnow()
        scene.progress = 0
        scene.progress_updated_at = utcnow()
        session.flush()
        should_start = True
        response = build_scene_response(session, scene)

    if should_start:
        start_generation(world_id, path, api_key)
    return response


@app.post("/worlds/{world_id}/scenes/{path:path}/retry", response_model=SceneResponse)
def retry_scene(world_id: str, path: str, payload: SceneGenerationRequest) -> SceneResponse:
    payload.path = path
    return generate_scene_endpoint(world_id, payload)


def ensure_scene_exists(session: Session, world_id: str, path: str) -> Scene:
    stmt = select(Scene).where(Scene.world_id == world_id, Scene.path == path)
    scene = session.execute(stmt).scalars().first()
    if scene:
        return scene

    scene = Scene(
        world_id=world_id,
        path=path,
        depth=compute_depth(path),
        status=SceneStatus.PENDING,
    )
    parent = parent_path(path)
    if parent is not None:
        scene.trigger_choice = None
    session.add(scene)
    session.flush()
    return scene


def _resolve_asset_url(value: Optional[str], *, variant: str) -> Optional[str]:
    if not value:
        return None
    try:
        return storage_client.resolve_url(value, variant=variant)
    except AttributeError:
        # Back-compat for older StorageClient implementations
        return value


def _update_scene_progress(world_id: str, path: str, progress: Optional[int]) -> None:
    if progress is None:
        return
    with session_scope() as session:
        scene = (
            session.execute(select(Scene).where(Scene.world_id == world_id, Scene.path == path))
            .scalars()
            .first()
        )
        if not scene or scene.status != SceneStatus.QUEUED:
            return
        scene.progress = int(progress)
        scene.progress_updated_at = utcnow()


def ensure_action_beat(scene: Dict[str, Any], fallback_choice: Optional[str]) -> None:
    prompt = scene.get("sora_prompt") or ""
    if "Action Beat:" in prompt:
        logger.info("[prompt] action beat already present")
        return
    candidate = fallback_choice or ""
    if not candidate:
        choices = scene.get("choices") or []
        if choices:
            candidate = choices[0]
        else:
            candidate = (scene.get("scenario_display") or "")[:160]
    candidate = candidate.strip()
    if not candidate:
        candidate = "Trigger a dramatic cross-world portal event within 8 seconds."
    scene["sora_prompt"] = prompt.rstrip() + f"\nAction Beat: {candidate}"
    logger.info("[prompt] appended action beat: %s", candidate)


def build_scene_response(session: Session, scene: Scene) -> SceneResponse:
    choices = scene.choices or []
    child_statuses: List[str] = []
    child_paths: List[str] = []
    for idx in range(len(choices) or 3):
        child = scene.child_path(idx) if hasattr(scene, "child_path") else _child_path(scene.path, idx)
        child_paths.append(child)
        child_scene = (
            session.execute(
                select(Scene).where(Scene.world_id == scene.world_id, Scene.path == child)
            ).scalars().first()
        )
        if child_scene is None:
            child_statuses.append(SceneStatus.PENDING.value)
        else:
            child_statuses.append(child_scene.status.value)

    return SceneResponse(
        worldId=scene.world_id,
        path=scene.path,
        depth=scene.depth,
        status=scene.status.value,
        scenarioDisplay=scene.scenario_display,
        soraPrompt=scene.sora_prompt,
        triggerChoice=scene.trigger_choice,
        choices=choices if isinstance(choices, list) else [],
        choicesStatus=child_statuses,
        childrenPaths=child_paths,
        videoUrl=_resolve_asset_url(scene.video_url, variant="video"),
        posterUrl=_resolve_asset_url(scene.poster_url, variant="poster"),
        failureCode=scene.failure_code,
        failureDetail=scene.failure_detail,
        queuedSince=scene.started_at,
        updatedAt=scene.updated_at,
        progress=getattr(scene, "progress", None),
        progressUpdatedAt=getattr(scene, "progress_updated_at", None),
        stateSummary=getattr(scene, "state_summary", None),
    )


def start_generation(world_id: str, path: str, api_key: str) -> None:
    cancel_event = threading.Event()
    thread = threading.Thread(
        target=_generate_scene,
        args=(world_id, path, api_key, cancel_event),
        daemon=True,
        name=f"gen-{world_id}-{path or 'root'}",
    )
    handle = GenerationHandle(world_id=world_id, path=path, cancel_event=cancel_event, thread=thread)
    register_generation(handle)
    logger.info("generation queued world=%s path=%s", world_id, path or "root")
    thread.start()


def _generate_scene(world_id: str, path: str, api_key: str, cancel_event: threading.Event) -> None:
    contributor_hash = hash_contributor(api_key, path)
    try:
        logger.info("generation started world=%s path=%s", world_id, path or "root")
        try:
            _generate_scene_inner(world_id, path, api_key, cancel_event, contributor_hash)
        except SceneCancelled:
            logger.info("generation cancelled world=%s path=%s", world_id, path or "root")
            _mark_pending(world_id, path)
        except Exception as exc:
            logger.exception("generation error world=%s path=%s", world_id, path or "root")
            _mark_failed(world_id, path, "generation_error", str(exc))
    finally:
        logger.info("generation finished world=%s path=%s", world_id, path or "root")
        clear_generation(world_id, path)


def _generate_scene_inner(
    world_id: str,
    path: str,
    api_key: str,
    cancel_event: threading.Event,
    contributor_hash: str,
) -> None:
    with session_scope() as session:
        scene = (
            session.execute(
                select(Scene).where(Scene.world_id == world_id, Scene.path == path).with_for_update()
            )
            .scalars()
            .one()
        )
        if scene.status != SceneStatus.QUEUED:
            return
        scene.started_at = scene.started_at or utcnow()
        session.flush()

    if cancel_event.is_set():
        _mark_pending(world_id, path)
        return

    planner_result = plan_scene(world_id, path, api_key)
    if planner_result.get("_planner_missing_prompt"):
        _mark_failed(world_id, path, "planner_missing_prompt", planner_result.get("_planner_missing_prompt_reason", ""))
        return

    if cancel_event.is_set():
        _mark_pending(world_id, path)
        return

    ensure_action_beat(planner_result, planner_result.get("_chosen_choice"))

    try:
        asset = render_scene_video(world_id, path, planner_result["sora_prompt"], api_key, cancel_event)
    except SceneCancelled:
        _mark_pending(world_id, path)
        return
    except Exception as exc:
        _mark_failed(world_id, path, "sora_error", str(exc))
        return

    if cancel_event.is_set():
        _mark_pending(world_id, path)
        return

    prior_state_summaries = collect_state_summaries(world_id, path)
    state_summary_text = summarise_scene_state(
        api_key=api_key,
        base_prompt=BASE_PROMPT,
        scenario_display=planner_result["scenario_display"],
        choices=planner_result["choices"],
        prior_summaries=prior_state_summaries,
    )
    if not state_summary_text:
        state_summary_text = planner_result["scenario_display"]

    with session_scope() as session:
        scene = (
            session.execute(
                select(Scene).where(Scene.world_id == world_id, Scene.path == path).with_for_update()
            )
            .scalars()
            .one()
        )
        scene.scenario_display = planner_result["scenario_display"]
        scene.sora_prompt = planner_result["sora_prompt"]
        scene.choices = planner_result["choices"]
        scene.planner_model = PLANNER_MODEL
        scene.planner_raw = planner_result.get("_raw_planner_output")
        if isinstance(storage_client, LocalStorageClient):
            scene.video_url = asset.video_url
            scene.poster_url = asset.poster_url
        else:
            scene.video_url = asset.video_key
            scene.poster_url = asset.poster_key
        scene.video_seconds = DEFAULT_SECONDS
        scene.status = SceneStatus.READY
        scene.failure_code = None
        scene.failure_detail = None
        scene.contributor_hash = contributor_hash
        scene.started_at = None
        scene.state_summary = state_summary_text
        scene.progress = 100
        scene.progress_updated_at = utcnow()
        scene.trigger_choice = determine_trigger_choice(session, world_id, path)
        session.add(
            SceneMetric(
                scene_id=scene.id,
                rendered=1,
                render_time_ms=None,
                storage_bytes=asset.bytes_written,
            )
        )


def determine_trigger_choice(session: Session, world_id: str, path: str) -> Optional[str]:
    parent = parent_path(path)
    if parent is None:
        return None
    parent_scene = (
        session.execute(select(Scene).where(Scene.world_id == world_id, Scene.path == parent))
        .scalars()
        .first()
    )
    if parent_scene is None or not parent_scene.choices:
        return None
    idx = last_choice_index(path)
    if idx is None:
        return None
    if idx < len(parent_scene.choices):
        return parent_scene.choices[idx]
    return None


def plan_scene(world_id: str, path: str, api_key: str) -> Dict[str, Any]:
    if not path:
        result = plan_initial_scene(api_key=api_key, base_prompt=BASE_PROMPT, model=PLANNER_MODEL)
        first_choice = (result.get("choices") or [None])[0]
        result["_chosen_choice"] = first_choice
        result["_state_context"] = []
    else:
        parent_path_value = parent_path(path)
        if parent_path_value is None:
            raise RuntimeError("Path has no parent; cannot continue")
        ancestor_paths = ancestor_path_list(path)
        with session_scope() as session:
            stmt = select(Scene).where(Scene.world_id == world_id, Scene.path.in_(ancestor_paths))
            rows = session.execute(stmt).scalars().all()
        by_path = {row.path: row for row in rows}
        parent = by_path.get(parent_path_value)
        if parent is None or not parent.choices:
            raise RuntimeError("Parent scene lacks choices; cannot continue")
        prior_prompts = []
        state_context: List[str] = []
        for anc_path in ancestor_paths:
            scene = by_path.get(anc_path)
            if scene and scene.sora_prompt:
                prior_prompts.append(scene.sora_prompt)
            if anc_path != path and scene and getattr(scene, "state_summary", None):
                state_context.append(scene.state_summary)
        idx = last_choice_index(path)
        if idx is None or idx >= len(parent.choices):
            raise RuntimeError("Invalid choice index for path")
        chosen_choice = parent.choices[idx]
        logger.info("[planner] continue world=%s path=%s choice=%s state_context=%s", world_id, path, chosen_choice, state_context)
        result = plan_next_scene(
            api_key=api_key,
            base_prompt=BASE_PROMPT,
            prior_sora_prompts=prior_prompts,
            chosen_choice=chosen_choice,
            state_summaries=state_context,
            model=PLANNER_MODEL,
        )
        result["_chosen_choice"] = chosen_choice
        result["_state_context"] = state_context
    return result


class SceneCancelled(Exception):
    pass


def ancestor_path_list(path: str) -> List[str]:
    parts = split_path(path)
    ancestors: List[str] = []
    for end in range(1, len(parts) + 1):
        ancestor = "/".join(str(part) for part in parts[:end])
        ancestors.append(ancestor)
    if ancestors:
        # Always include root path "" as the base context
        ancestors.insert(0, "")
    else:
        ancestors.append("")
    return ancestors


def collect_state_summaries(world_id: str, path: str) -> List[str]:
    ancestor_paths = ancestor_path_list(path)
    # Exclude the current path; we only need previously locked scenes
    ancestor_context = [p for p in ancestor_paths if p != path]
    if not ancestor_context:
        return []
    with session_scope() as session:
        rows = (
            session.execute(
                select(Scene).where(Scene.world_id == world_id, Scene.path.in_(ancestor_context))
            )
            .scalars()
            .all()
        )
    rows.sort(key=lambda scene: scene.depth)
    summaries = [row.state_summary for row in rows if getattr(row, "state_summary", None)]
    return summaries


STATE_SUMMARY_SYSTEM = """
You are the chronicler for an expansive multiverse adventure.

Summarise the current state in at most three short bullet points.
- Track key elements: chronoglyph shards remaining/found, portal stability, allies or foes involved, immediate threats, and location shifts between worlds.
- Highlight cause and effect (e.g., how actions in one world impact another).
- Keep bullets under 160 characters, starting each with "- ". No extra commentary.
""".strip()


def summarise_scene_state(
    api_key: str,
    base_prompt: str,
    scenario_display: str,
    choices: List[str],
    prior_summaries: List[str],
) -> Optional[str]:
    model = STATE_SUMMARY_MODEL or ""
    if not model or model.lower() == "none":
        return None

    prior_section = "\n".join(f"- {summary}" for summary in prior_summaries) if prior_summaries else "(none yet)"
    choices_section = "\n".join(f"- {choice}" for choice in choices)
    user_input = f"""
WORLD BASE PROMPT (trimmed):
{base_prompt[:800]}

PRIOR STATE SNAPSHOT:
{prior_section}

CURRENT SCENE NARRATION:
{scenario_display}

CHOICES OFFERED NEXT:
{choices_section}

TASK: Summarise the evolving state using at most three bullets as instructed.
""".strip()

    try:
        summary_text = responses_create(
            api_key=api_key,
            model=model,
            instructions=STATE_SUMMARY_SYSTEM,
            user_input=user_input,
        )
        cleaned = summary_text.strip()
        return cleaned if cleaned else None
    except Exception as exc:  # pragma: no cover - best effort
        logger.warning("state summary generation failed: %s", exc)
        return None


def render_scene_video(
    world_id: str,
    path: str,
    sora_prompt: str,
    api_key: str,
    cancel_event: threading.Event,
) -> StoredAsset:
    video_id, video_path = None, None
    parent_last_frame: Optional[Path] = None
    parent = parent_path(path)
    if parent is not None:
        with session_scope() as session:
            parent_scene = (
                session.execute(
                    select(Scene).where(Scene.world_id == world_id, Scene.path == parent)
                )
                .scalars()
                .first()
            )
        if parent_scene:
            logger.info("[continuity] parent scene world=%s parent_path=%s status=%s", world_id, parent, getattr(parent_scene, "status", None))
        else:
            logger.warning("[continuity] missing parent scene world=%s parent_path=%s", world_id, parent)
        if parent_scene and parent_scene.poster_url:
            logger.info("[continuity] fetching last frame for world=%s parent_path=%s url=%s", world_id, parent or "root", parent_scene.poster_url)
            parent_last_frame = download_asset(parent_scene.poster_url, variant="poster")
            logger.info("[continuity] download path=%s type=%s exists=%s", parent_last_frame, type(parent_last_frame), parent_last_frame.exists() if isinstance(parent_last_frame, Path) else None)
            if isinstance(parent_last_frame, Path) and parent_last_frame.exists():
                logger.info("[continuity] last frame ready at %s", parent_last_frame)
            else:
                logger.warning("[continuity] failed to obtain last frame for world=%s path=%s", world_id, path or "root")

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            if parent_last_frame:
                logger.info("[continuity] sending input_reference=%s", parent_last_frame)
            else:
                logger.info("[continuity] no input_reference available for world=%s path=%s", world_id, path or "root")
            video_job = sora_create_video(
                api_key=api_key,
                sora_prompt=sora_prompt,
                model=SORA_MODEL,
                size=VIDEO_SIZE,
                seconds=DEFAULT_SECONDS,
                input_reference_path=parent_last_frame,
            )
            _update_scene_progress(world_id, path, video_job.get("progress"))
            video = sora_poll_until_complete(
                api_key,
                video_job,
                cancel_event,
                progress_callback=lambda prog: _update_scene_progress(world_id, path, prog),
            )
            if cancel_event.is_set():
                raise SceneCancelled()
            video_id = video["id"]

            video_file = tmp_dir_path / f"{video_id}.mp4"
            sora_download_content(api_key, video_id, video_file, variant="video")
            frame_file = tmp_dir_path / f"{video_id}_last.jpg"
            extract_last_frame(video_file, frame_file)

            key_prefix = f"{world_id}/{path or 'root'}"
            asset = storage_client.upload(video_file, frame_file, key_prefix=key_prefix)
            return asset
    finally:
        if parent_last_frame and parent_last_frame.exists():
            parent_last_frame.unlink(missing_ok=True)


def download_asset(stored_value: str, variant: str) -> Optional[Path]:
    resolved_url = _resolve_asset_url(stored_value, variant=variant)
    if not resolved_url:
        logger.warning("[continuity] resolve failed for variant=%s value=%s", variant, stored_value)
        return None
    logger.info("[continuity] downloading asset variant=%s url=%s", variant, resolved_url)
    try:
        response = requests.get(resolved_url, timeout=30)
        logger.info("[continuity] download status=%s url=%s", response.status_code, resolved_url)
        if response.status_code >= 400:
            logger.warning("[continuity] download failed status=%s body=%s", response.status_code, response.text[:200])
            return None
        parsed = urlparse(resolved_url)
        path_suffix = Path(parsed.path).suffix.lower()
        if path_suffix in {".jpg", ".jpeg", ".png", ".webp", ".mp4"}:
            suffix = path_suffix
        else:
            suffix = ".mp4" if variant == "video" else ".jpg"
        fd, tmp_path = tempfile.mkstemp(suffix=suffix)
        os.close(fd)
        tmp = Path(tmp_path)
        with tmp.open("wb") as fh:
            fh.write(response.content)
        logger.info("[continuity] download saved to %s", tmp)
        return tmp
    except Exception as exc:
        logger.warning("[continuity] exception downloading asset: %s", exc)
        return None


def hash_contributor(api_key: str, path: str) -> str:
    import hashlib

    payload = f"{CONTRIBUTOR_SALT}:{path}:{api_key}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _mark_pending(world_id: str, path: str) -> None:
    with session_scope() as session:
        scene = (
            session.execute(select(Scene).where(Scene.world_id == world_id, Scene.path == path))
            .scalars()
            .first()
        )
        if not scene:
            return
        scene.status = SceneStatus.PENDING
        scene.started_at = None
        scene.failure_code = None
        scene.failure_detail = None
        scene.contributor_hash = None
        scene.progress = None
        scene.progress_updated_at = None
        logger.info("scene reset to pending world=%s path=%s", world_id, path or "root")


def _mark_failed(world_id: str, path: str, code: str, detail: str) -> None:
    with session_scope() as session:
        scene = (
            session.execute(select(Scene).where(Scene.world_id == world_id, Scene.path == path))
            .scalars()
            .first()
        )
        if not scene:
            return
        scene.status = SceneStatus.FAILED
        scene.failure_code = code
        scene.failure_detail = detail
        scene.started_at = None
        scene.contributor_hash = None
        scene.progress = None
        scene.progress_updated_at = None
        logger.warning("scene failed world=%s path=%s code=%s detail=%s", world_id, path or "root", code, detail)


def _timeout_watchdog() -> None:
    while True:
        time.sleep(WATCHDOG_INTERVAL_SECONDS)
        cutoff = utcnow() - timedelta(seconds=SCENE_TIMEOUT_SECONDS)
        try:
            with session_scope() as session:
                stmt = select(Scene).where(
                    Scene.world_id == WORLD_ID,
                    Scene.status == SceneStatus.QUEUED,
                    Scene.started_at.isnot(None),
                )
                rows = session.execute(stmt).scalars().all()
                for scene in rows:
                    if scene.started_at and scene.started_at < cutoff:
                        request_cancel(scene.world_id, scene.path)
                        scene.status = SceneStatus.PENDING
                        scene.started_at = None
        except Exception:
            continue


def _child_path(path: str, index: int) -> str:
    if not path:
        return str(index)
    return f"{path}/{index}"


# === Planner Helpers ===

PLANNER_SYSTEM = """
You are the Scenario Planner for a Sora-powered choose-your-own-adventure game.

Your job:
- Given a BASE PROMPT (world/tone) or a CONTINUATION (previous scene prompts + the player's chosen action),
- Produce a JSON object that contains:
  {
    "scenario_display": "A short paragraph (<= 120 words) narrating the current scene to show in the UI.",
    "sora_prompt": "<A detailed Sora prompt for generating an 8-second video.>",
    "choices": ["<choice 1>", "<choice 2>", "<choice 3>"]
  }

Rules:
1) The 'sora_prompt' must be the exact text we send to Sora's /videos API.
   - Include a line: "Context (not visible in video, only for AI guidance): ..." to carry forward continuity and constraints.
   - Include a line: "Prompt: ..." with concrete, cinematic directions (camera, subject, motion, lighting).
   - Keep 'Prompt' specific to a single 8-second shot.
   - For steps after the first, begin exactly from the final frame of the previous scene.

2) Safety & platform constraints (strict):
   - Content must be suitable for audiences under 18.
   - Do NOT depict real people (including public figures) or copyrighted/fictional characters.
   - Avoid copyrighted music and explicit logos/trademarks. Use generic brand cues only.
   - Avoid hate, sexual content, excessive violence, or self-harm.

3) Continuity:
   - Maintain consistent characters, setting, tone, camera language, and lighting unless the choice implies a justified shift.
   - Ensure smooth shot-to-shot transitions (same time of day, matching positions/poses as appropriate).

4) Choices:
   - Provide exactly three distinct options for what the player can do next.
   - Make each option feasible in the next short shot, and clearly different in intent.
   - Keep each choice concise (<= 22 words).
   - Aim for options that open visibly different paths (new discoveries, escalations, or dramatic reactions); avoid three small variations of the same move.

5) Multiverse context:
   - Portals can appear, destabilise, or be traversed in any scene. Highlight iconic world mashups (futuristic vehicles vs. dark fantasy adversaries, etc.) without naming trademarks.
   - Tie choices to the chronoglyph shard hunt and the Cataclysm Rift stakes—progress, setbacks, or intel should be obvious on-screen.
   - Allies/enemies from other worlds should react believably to cross-world physics or tech clashes.

6) Pacing & shot design:
   - Each 8-second shot must deliver a complete beat (setup → escalation → visible outcome) that meaningfully changes the situation.
   - Start in motion—skip drawn-out establishing frames. Hit the key moment within the first 3 seconds and carry energy through the remainder.
   - End with a fresh reveal, reaction, or consequence that sets up the next decision.

7) Output strictly JSON. No markdown, no commentary, no code fences.
""".strip()


def responses_create(api_key: str, model: str, instructions: str, user_input: str) -> str:
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": model,
        "instructions": instructions,
        "input": user_input,
    }
    response = requests.post(RESPONSES_ENDPOINT, headers=headers, json=payload, timeout=120)
    if response.status_code >= 400:
        raise RuntimeError(f"Responses API error {response.status_code}: {response.text}")
    data = response.json()

    text = data.get("output_text", "")
    if text:
        return text

    try:
        items = data.get("output", [])
        builder: List[str] = []
        for item in items:
            blocks = item.get("content") or []
            for block in blocks:
                b_type = block.get("type")
                if b_type in {"output_text", "text"}:
                    builder.append(block.get("text", ""))
                elif isinstance(block.get("text"), list):
                    for segment in block["text"]:
                        if isinstance(segment, dict) and segment.get("type") in {"output_text", "text"}:
                            builder.append(segment.get("text", ""))
        if builder:
            return "".join(builder)
    except Exception:
        pass

    return json.dumps(data)


def extract_first_json(text: str) -> dict:
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        raise ValueError("Planner did not return JSON. Received:\n" + text[:800])
    return json.loads(match.group(0))


def normalize_scene_payload(scene: Dict[str, Any]) -> Dict[str, Any]:
    def _pick(keys: List[str]) -> Any:
        for key in keys:
            if key in scene:
                value = scene[key]
                if value is None:
                    continue
                if isinstance(value, str):
                    value = value.strip()
                    if not value:
                        continue
                return value
        return None

    scenario_display_keys = [
        "scenario_display",
        "scene_display",
        "scene_description",
        "scenario_description",
        "narration",
        "description",
        "display",
        "story",
    ]
    sora_prompt_keys = [
        "sora_prompt",
        "soraPrompt",
        "prompt",
        "video_prompt",
        "videoPrompt",
        "scene_prompt",
        "scenePrompt",
        "shot_prompt",
        "shotPrompt",
    ]
    choices_keys = ["choices", "options", "next_choices", "actions", "nextOptions"]

    scenario_display = _pick(scenario_display_keys)
    if isinstance(scenario_display, list):
        parts = [str(item).strip() for item in scenario_display if str(item).strip()]
        scenario_display = " ".join(parts)
    if not scenario_display:
        scenario_display = "Planner response missing scene description. Adjust your prompt and retry."

    sora_prompt_raw = _pick(sora_prompt_keys)
    sora_prompt_missing = False
    sora_prompt_missing_reason = ""

    sora_prompt_value: Any = sora_prompt_raw
    if isinstance(sora_prompt_value, dict):
        lines: List[str] = []
        for key, value in sora_prompt_value.items():
            if value is None:
                continue
            text_val = str(value).strip()
            if not text_val:
                continue
            lines.append(f"{key}: {text_val}")
        if lines:
            sora_prompt_value = "\n".join(lines).strip()
        else:
            sora_prompt_missing = True
            sora_prompt_missing_reason = "Planner returned prompt dict but it had no usable values."
            sora_prompt_value = ""
    elif isinstance(sora_prompt_value, list):
        joined = "\n".join(str(item).strip() for item in sora_prompt_value if str(item).strip())
        if joined:
            sora_prompt_value = joined
        else:
            sora_prompt_missing = True
            sora_prompt_missing_reason = "Planner returned prompt list but all entries were empty."
            sora_prompt_value = ""

    if sora_prompt_value is None:
        sora_prompt_missing = True
        if not sora_prompt_missing_reason:
            sora_prompt_missing_reason = "Planner response missing recognized Sora prompt field."
        sora_prompt_value = ""
    elif not isinstance(sora_prompt_value, str):
        sora_prompt_value = str(sora_prompt_value).strip()
        if not sora_prompt_value:
            sora_prompt_missing = True
            if not sora_prompt_missing_reason:
                sora_prompt_missing_reason = "Planner returned non-string prompt that was empty after casting."
    else:
        sora_prompt_value = sora_prompt_value.strip()
        if not sora_prompt_value:
            sora_prompt_missing = True
            if not sora_prompt_missing_reason:
                sora_prompt_missing_reason = "Planner Sora prompt string was blank."

    sora_prompt = (
        "Planner response missing Sora prompt details. Please tweak your base prompt or retry."
        if sora_prompt_missing
        else sora_prompt_value
    )

    raw_choices = _pick(choices_keys)
    choices: List[str] = []
    if isinstance(raw_choices, list):
        choices = [str(choice).strip() for choice in raw_choices if str(choice).strip()]
    elif isinstance(raw_choices, str):
        fragments = re.split(r"[\n|]", raw_choices)
        choices = [frag.strip(" •-\t").strip() for frag in fragments if frag.strip()]

    while len(choices) < 3:
        choices.append(f"Missing choice {len(choices) + 1}. Update prompt and regenerate.")
    if len(choices) > 3:
        choices = choices[:3]

    normalized = dict(scene)
    normalized["scenario_display"] = scenario_display
    normalized["sora_prompt"] = sora_prompt
    normalized["choices"] = choices
    normalized["_planner_missing_prompt"] = sora_prompt_missing
    normalized["_planner_missing_prompt_reason"] = sora_prompt_missing_reason
    return normalized


def plan_initial_scene(api_key: str, base_prompt: str, model: str) -> dict:
    guidance_section = ""
    if PROMPT_GUIDANCE:
        guidance_section = f"\n\nADDITIONAL WORLD GUIDANCE:\n{PROMPT_GUIDANCE}"

    user_input = f"""
TASK: Create the opening scene with three choices.

BASE PROMPT:
{base_prompt}

Shot length: 8 seconds.
Return JSON with keys: scenario_display, sora_prompt, choices (3).
{guidance_section}
""".strip()
        raw = responses_create(api_key=api_key, model=model, instructions=PLANNER_SYSTEM, user_input=user_input)
    scene = normalize_scene_payload(extract_first_json(raw))
    scene["_raw_planner_output"] = raw.strip()
    scene["_planner_model"] = model
    scene["_planner_stage"] = "initial"
    return scene


def plan_next_scene(
    api_key: str,
    base_prompt: str,
    prior_sora_prompts: List[str],
    chosen_choice: str,
    state_summaries: List[str],
    model: str,
) -> dict:
    prior_joined = "\n\n---\n\n".join(prior_sora_prompts) if prior_sora_prompts else "(first continuation)"
    state_section = (
        "\n".join(f"- {summary}" for summary in state_summaries)
        if state_summaries
        else "- No prior state summary available yet."
    )
    guidance_section = f"\n\nADDITIONAL WORLD GUIDANCE:\n{PROMPT_GUIDANCE}" if PROMPT_GUIDANCE else ""

    user_input = f"""
TASK: Create the next scene with three choices, continuing the story.

BASE PROMPT:
{base_prompt}

PRIOR SORA PROMPTS (in order; each was used to generate an 8s video):
{prior_joined}

CURRENT STATE SNAPSHOT (bullet list):
{state_section}

PLAYER'S CHOSEN ACTION TO CONTINUE:
{chosen_choice}

Note: The next 8-second shot MUST begin exactly from the final frame of the previous shot,
preserving continuity (subjects, camera position, lighting, motion direction), unless the chosen action implies a change.

Return JSON with keys: scenario_display, sora_prompt, choices (3).
{guidance_section}
""".strip()
        raw = responses_create(api_key=api_key, model=model, instructions=PLANNER_SYSTEM, user_input=user_input)
    scene = normalize_scene_payload(extract_first_json(raw))
    scene["_raw_planner_output"] = raw.strip()
    scene["_planner_model"] = model
    scene["_planner_stage"] = "continuation"
    return scene


# === Sora Helpers ===


def _auth_headers(api_key: str) -> Dict[str, str]:
    if not api_key:
        raise RuntimeError("OpenAI API key is required")
    return {
        "Authorization": f"Bearer {api_key}",
    }


def sora_create_video(
    api_key: str,
    sora_prompt: str,
    model: str,
    size: str,
    seconds: int,
    input_reference_path: Optional[Path] = None,
) -> dict:
    files = {
        "model": (None, model),
        "prompt": (None, sora_prompt),
        "size": (None, size),
        "seconds": (None, str(seconds)),
    }
    if input_reference_path and input_reference_path.exists():
        files["input_reference"] = (
            input_reference_path.name,
            open(input_reference_path, "rb"),
            _guess_mime(input_reference_path),
        )
    response = requests.post(SORA_VIDEOS_ENDPOINT, headers=_auth_headers(api_key), files=files, timeout=600)
    if response.status_code >= 400:
        raise RuntimeError(f"Sora create failed ({response.status_code}): {response.text}")
    return response.json()


def sora_retrieve_video(api_key: str, video_id: str) -> dict:
    url = f"{SORA_VIDEOS_ENDPOINT}/{video_id}"
    last_error: Optional[Exception] = None
    for attempt in range(5):
        try:
            response = requests.get(url, headers=_auth_headers(api_key), timeout=120)
        except requests.RequestException as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 8))
            continue

        if response.status_code >= 500 or response.status_code in (429, 520):
            last_error = RuntimeError(f"Sora retrieve failed ({response.status_code}): {response.text[:200]}")
            time.sleep(min(2 ** attempt, 8))
            continue

        if response.status_code >= 400:
            raise RuntimeError(f"Sora retrieve failed ({response.status_code}): {response.text}")

        try:
            return response.json()
        except ValueError as exc:
            last_error = exc
            time.sleep(min(2 ** attempt, 8))

    if last_error:
        raise RuntimeError(f"Sora retrieve failed after retries: {last_error}")
    raise RuntimeError("Sora retrieve failed after retries: unknown error")


def sora_download_content(api_key: str, video_id: str, out_path: Path, variant: str = "video") -> Path:
    url = f"{SORA_VIDEOS_ENDPOINT}/{video_id}/content"
    with requests.get(
        url,
        headers=_auth_headers(api_key),
        params={"variant": variant},
        stream=True,
        timeout=1800,
    ) as response:
        if response.status_code >= 400:
            raise RuntimeError(f"Sora download failed ({response.status_code}): {response.text}")
        with open(out_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
    return out_path


def sora_poll_until_complete(
    api_key: str,
    job: dict,
    cancel_event: threading.Event,
    progress_callback: Optional[Callable[[Optional[int]], None]] = None,
) -> dict:
    video = job
    video_id = video["id"]
    if progress_callback:
        progress_callback(video.get("progress"))
    while video.get("status") in ("queued", "in_progress"):
        if cancel_event.is_set():
            raise SceneCancelled()
        time.sleep(2)
        video = sora_retrieve_video(api_key, video_id)
        if progress_callback:
            progress_callback(video.get("progress"))

    if video.get("status") != "completed":
        message = (video.get("error") or {}).get("message", f"Job {video_id} failed")
        raise RuntimeError(message)
    return video


def extract_last_frame(video_path: Path, out_image_path: Path) -> Path:
    if cv2 is not None:
        cap = cv2.VideoCapture(str(video_path))
        if cap.isOpened():
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
            success, frame = False, None
            if total > 0:
                cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
                success, frame = cap.read()
            if not success or frame is None:
                cap.release()
                cap = cv2.VideoCapture(str(video_path))
                while True:
                    ret, fr = cap.read()
                    if not ret:
                        break
                    frame = fr
                    success = True
            cap.release()
            if success and frame is not None:
                if cv2.imwrite(str(out_image_path), frame):
                    return out_image_path

    if FFMPEG_BIN:
        cmd = [
            FFMPEG_BIN,
            "-y",
            "-sseof",
            "-0.05",
            "-i",
            str(video_path),
            "-frames:v",
            "1",
            str(out_image_path),
        ]
        subprocess.check_call(cmd)
        if out_image_path.exists():
            return out_image_path

    raise RuntimeError("Failed to extract last frame: OpenCV/FFmpeg unavailable or video unreadable.")


def _guess_mime(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0]
    return mime or "application/octet-stream"


def generate_scene_video(
    api_key: str,
    sora_prompt: str,
    model: str,
    size: str,
    seconds: int,
    input_reference: Optional[Path],
) -> Tuple[str, Path, Path]:
    seconds = normalize_seconds(seconds)
    job = sora_create_video(
        api_key=api_key,
        sora_prompt=sora_prompt,
        model=model,
        size=size,
        seconds=seconds,
        input_reference_path=input_reference,
    )

    video = sora_poll_until_complete(api_key, job, threading.Event())
    video_id = video["id"]

    video_path = VIDEO_DIR / f"{video_id}.mp4"
    sora_download_content(api_key, video_id, video_path, variant="video")

    last_frame_path = FRAME_DIR / f"{video_id}_last.jpg"
    extract_last_frame(video_path, last_frame_path)
    return video_id, video_path, last_frame_path


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
