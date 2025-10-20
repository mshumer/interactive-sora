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
try:
    from pydantic import ConfigDict
except ImportError:  # pragma: no cover - Pydantic v1 fallback
    ConfigDict = None
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
from nyc_catalog import format_area_context, get_area, list_adjacent, load_catalog
from nyc_state import normalize_state, seed_root_state, update_state
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
    "Experience an apocalyptic cyberpunk New York City from the front of an open-air skytram pod. "
    "Mag-rails thread through authentic borough layouts—Times Square to Harlem to SoHo—with neon ruins, overgrown canopies, and AI sentinels flickering below. "
    "You lean into the wind to survey districts, reactivate beacon relays, and uncover hidden enclaves. Faces stay obscured, but the city’s transformation is vivid and photorealistic."
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
            "Perspective: First-person view from the leading edge of an open-air skytram pod—nothing blocking the skyline.",
            "Pace: Glide 1–3 city blocks per shot—smooth acceleration, no sudden collisions, no shakes.",
            "Rails: Emphasize branching mag-rail junctions that let the pilot choose diverging paths through the borough.",
            "Exploration: Showcase landmarks, inhabitants, and ambient stories rather than combat or obstacle dodging.",
            "Photorealism: Cinematic HDR lighting, physically-based materials, volumetric depth—never stylised or toy-like.",
            "Audio: Continuous, heart-pounding soundscape blended with wind rush, rail resonance, and district ambience.",
            "Discovery Hook: End each beat on a compelling reveal (new vista, hidden enclave, signal spike) prompting the next choice.",
            "Show Junction: Hold the final seconds on the three diverging rails themselves (no signage/holograms) so each path is clearly framed.",
        ]
    )
)

PROMPT_GUIDANCE = os.environ.get("WORLD_PROMPT_GUIDANCE", "").strip() or DEFAULT_PROMPT_GUIDANCE
STATE_SUMMARY_MODEL = os.environ.get("STATE_SUMMARY_MODEL", "gpt-5-mini").strip()

VIDEO_DIR = Path("sora_cyoa_videos")
FRAME_DIR = Path("sora_cyoa_frames")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)

ROOT_REFERENCE_IMAGE = Path("data/source_image.png")

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
    if ConfigDict is not None:
        model_config = ConfigDict(populate_by_name=True)
    else:  # pragma: no cover - Pydantic v1 support
        class Config:
            allow_population_by_field_name = True
    world_id: str = Field(..., alias="worldId")
    path: str
    depth: int
    status: str
    scenario_display: Optional[str] = Field(None, alias="scenarioDisplay")
    sora_prompt: Optional[str] = Field(None, alias="soraPrompt")
    trigger_choice: Optional[str] = Field(None, alias="triggerChoice")
    choices: List[str] = Field(default_factory=list)
    choices_short: List[str] = Field(default_factory=list, alias="choicesShort")
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
    state_json: Optional[dict] = Field(None, alias="stateJson")


class SceneGenerationRequest(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(populate_by_name=True)
    else:  # pragma: no cover - Pydantic v1 support
        class Config:
            allow_population_by_field_name = True
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
    if ConfigDict is not None:
        model_config = ConfigDict(populate_by_name=True)
    else:  # pragma: no cover - Pydantic v1 support
        class Config:
            allow_population_by_field_name = True
    world_id: str = Field(..., alias="worldId")
    base_prompt: str = Field(..., alias="basePrompt")
    planner_model: str = Field(..., alias="plannerModel")
    sora_model: str = Field(..., alias="soraModel")
    video_size: str = Field(..., alias="videoSize")


class WorldMetricsResponse(BaseModel):
    if ConfigDict is not None:
        model_config = ConfigDict(populate_by_name=True)
    else:  # pragma: no cover - Pydantic v1 support
        class Config:
            allow_population_by_field_name = True
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
    try:
        load_catalog()
    except Exception as exc:
        logger.warning("failed to load NYC catalog at startup: %s", exc)
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
    else:
        scene.state_json = seed_root_state()
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
        candidate = "Frame the junction—highlight left, right, and downward rails with glowing signage before committing."
    scene["sora_prompt"] = prompt.rstrip() + f"\nAction Beat: {candidate}"
    logger.info("[prompt] appended action beat: %s", candidate)


def validate_sora_prompt_structure(prompt: str) -> Optional[str]:
    required_tokens = [
        "Context (not visible",
        "Location:",
        "Faces:",
        "Movement:",
        "Inventory:",
        "Ecosystem:",
        "Continuity:",
        "Audio:",
        "Photorealism:",
        "Camera:",
        "Prompt:",
        "Action Beat:",
    ]
    missing = [token for token in required_tokens if token not in prompt]
    if missing:
        return f"missing required prompt lines: {', '.join(missing)}"
    return None


def build_scene_response(session: Session, scene: Scene) -> SceneResponse:
    choices = scene.choices if isinstance(scene.choices, list) else []
    choices_short = scene.choices_short if isinstance(scene.choices_short, list) else []
    if not choices_short:
        choices_short = list(choices)
    else:
        normalized_short: List[str] = []
        for idx in range(len(choices)):
            short_value = choices_short[idx] if idx < len(choices_short) else None
            long_value = choices[idx] if idx < len(choices) else None
            candidate = (short_value or "").strip()
            if not candidate and long_value is not None:
                candidate = str(long_value).strip()
            if not candidate:
                candidate = f"Choice {idx + 1}"
            normalized_short.append(candidate)
        choices_short = normalized_short
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
        choices=choices,
        choices_short=choices_short,
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
        stateJson=getattr(scene, "state_json", None),
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
    next_state = planner_result.get("_next_state")
    if not isinstance(next_state, dict):
        next_state = update_state(planner_result.get("_prior_state"), planner_result.get("state_update"))
        planner_result["_next_state"] = next_state
    if planner_result.get("_planner_missing_prompt"):
        _mark_failed(world_id, path, "planner_missing_prompt", planner_result.get("_planner_missing_prompt_reason", ""))
        return

    if cancel_event.is_set():
        _mark_pending(world_id, path)
        return

    ensure_action_beat(planner_result, planner_result.get("_chosen_choice"))

    structure_error = validate_sora_prompt_structure(planner_result["sora_prompt"])
    if structure_error:
        _mark_failed(world_id, path, "prompt_structure_error", structure_error)
        return

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
        scene.choices_short = planner_result.get("choices_short")
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
        scene.state_json = planner_result.get("_next_state")
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
        prior_state = seed_root_state()
        area = get_area(prior_state["location"].get("area_id"))
        result = call_planner(
            api_key=api_key,
            model=PLANNER_MODEL,
            base_prompt=BASE_PROMPT,
            state=prior_state,
            area=area,
            prior_prompts=[],
            prior_state_summaries=[],
            player_choice=None,
            stage_label="initial",
        )
        first_choice = (result.get("choices") or [None])[0]
        result["_chosen_choice"] = first_choice
        result["_state_context"] = []
        result["_prior_state"] = prior_state
        next_state = update_state(prior_state, result.get("state_update"))
        prev_area_id = prior_state["location"].get("area_id")
        new_area_id = next_state["location"].get("area_id")
        if prev_area_id and new_area_id and new_area_id != prev_area_id:
            allowed = set(list_adjacent(prev_area_id))
            if new_area_id not in allowed:
                logger.warning(
                    "[planner] invalid area hop %s -> %s; clamping to previous",
                    prev_area_id,
                    new_area_id,
                )
                next_state["location"].update(prior_state["location"])
        result["_next_state"] = next_state
        if not isinstance(result.get("state_update"), dict) or not result["state_update"]:
            logger.warning("[planner] initial scene missing state_update; using prior defaults")
            result["state_update"] = next_state
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
        prior_prompts: List[str] = []
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
        prior_state = normalize_state(getattr(parent, "state_json", None))
        area = get_area(prior_state["location"].get("area_id"))
        result = call_planner(
            api_key=api_key,
            model=PLANNER_MODEL,
            base_prompt=BASE_PROMPT,
            state=prior_state,
            area=area,
            prior_prompts=prior_prompts,
            prior_state_summaries=state_context,
            player_choice=chosen_choice,
            stage_label="continuation",
        )
        result["_chosen_choice"] = chosen_choice
        result["_state_context"] = state_context
        result["_prior_state"] = prior_state
        next_state = update_state(prior_state, result.get("state_update"))
        prev_area_id = prior_state["location"].get("area_id")
        new_area_id = next_state["location"].get("area_id")
        if prev_area_id and new_area_id and new_area_id != prev_area_id:
            allowed = set(list_adjacent(prev_area_id))
            if new_area_id not in allowed:
                logger.warning(
                    "[planner] invalid area hop %s -> %s; clamping to previous",
                    prev_area_id,
                    new_area_id,
                )
                next_state["location"].update(prior_state["location"])
        result["_next_state"] = next_state
        if not isinstance(result.get("state_update"), dict) or not result["state_update"]:
            logger.warning("[planner] continuation missing state_update; using merged state")
            result["state_update"] = next_state
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
You are the chronicler for the apocalyptic cyberpunk NYC skytram expedition.

Summarise the evolving situation in at most three short bullet points.
- Track which rail line the tram is on, beacon/signal progress, notable sights uncovered, and upcoming junction opportunities.
- Mention district transitions or planned forks (e.g., diverting toward SoHo vs continuing to FiDi).
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
    reference_path: Optional[Path] = None
    cleanup_reference: Optional[Path] = None
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
                reference_path = parent_last_frame
                cleanup_reference = parent_last_frame
            else:
                logger.warning("[continuity] failed to obtain last frame for world=%s path=%s", world_id, path or "root")
    else:
        if ROOT_REFERENCE_IMAGE.exists():
            reference_path = ROOT_REFERENCE_IMAGE
            logger.info("[continuity] using root reference image %s", ROOT_REFERENCE_IMAGE)
        else:
            logger.warning("[continuity] root reference image missing at %s", ROOT_REFERENCE_IMAGE)

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_dir_path = Path(tmp_dir)
            if reference_path:
                logger.info("[continuity] sending input_reference=%s", reference_path)
            else:
                logger.info("[continuity] no input_reference available for world=%s path=%s", world_id, path or "root")
            video_job = sora_create_video(
                api_key=api_key,
                sora_prompt=sora_prompt,
                model=SORA_MODEL,
                size=VIDEO_SIZE,
                seconds=DEFAULT_SECONDS,
                input_reference_path=reference_path,
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
        if cleanup_reference and cleanup_reference.exists():
            cleanup_reference.unlink(missing_ok=True)


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
You are the Scenario Planner for a Sora-powered, street-accurate cyberpunk New York City rail experience.

Workflow:
1. Read the WORLD BASE PROMPT (tone & stakes).
2. Examine the CURRENT STREET STATE JSON (location, movement, inventory, audio, policy).
3. Study the CATALOG AREA CONTEXT bullets.
4. Review PRIOR SORA PROMPTS (continuity) and, when provided, the PLAYER CHOICE.
5. Produce JSON with keys: scenario_display, sora_prompt, choices, choices_short, state_update.

Rules:
- Shots are photorealistic, continuous 8-second scenes. They must begin already in motion, escalate by the 3-second mark, and close on a hook that pushes the next decision.
- Perspective is first-person from the stabilized skytram cockpit. Keep the camera locked forward with gentle head turns—no third-person or external chase shots.
- Photorealism is mandatory: cinematic HDR lighting, physically-based materials, crisp atmospheric depth, zero stylisation or toy-like renderings.
- Movement must remain within 1–3 Manhattan blocks consistent with the rail route. Only switch to an adjacent catalog area when a junction logically branches there.
- Faces of every figure stay obscured (hoods, masks, deep shadow). Content must remain PG-13 and free of copyrighted logos/characters.
- Maintain geography: highlight real intersections, skyline silhouettes, and landmarks as seen from elevated rails.
- Audio stays heart-pounding and continuous; blend tram hum, HUD chimes, and district motif.
- Inventory represents cockpit controls (navigation holomap, signal scanner, stabilizer); show their effects on the ride rather than external gear.
- Choices must revolve around diverging rail paths (left branch to one district, right branch to another, vertical spur descending into infrastructure, etc.).
- End the shot by clearly presenting the available junction: all rails in view with signage/holographic markers that match the three upcoming choices (left/right/vertical or similar).

Sora prompt structure (exact wording & order):
Context (not visible in video, only for AI guidance):
Location: <borough>, <district>/<neighborhood>, nearest <intersection>, heading <heading>, moved <blocks> blocks
Faces: all faces obscured (hoods/masks/shadows) — mandatory
Movement: <tech_in_use> skytram on mag-rails at velocity "fast"; respect 1–3 block traversal budget
Inventory: <item in use> (tram interface/HUD element) and how it affects the ride
Ecosystem: adversaries <...>; creatures <...>; hazards <...>
Continuity: start from prior shot's final frame; keep time-of-day/weather consistent
Audio: <audio motif>, continuous, heart-pounding, no copyrighted music
Photorealism: cinematic HDR, physically-based materials, realistic textures, zero stylisation
Camera: first-person on open-air tram nose, stabilized gimbal, gentle roll only, no collisions or jitter

Prompt: <Concrete 8-second cinematic beat from the open-air tram nose, highlighting skyline vistas, rail forks, ambient life, and a discovery>

Action Beat: <Imperative describing the climax that lands inside the 8-second window>

Choices:
- Exactly three options (≤22 words each), clearly distinct in intent and traversal.
- Each choice must be a rail decision (e.g., divert left to <landmark>, stay on mainline toward <district>, descend into maintenance tunnel near <location>). Reference landmarks or signals that justify the fork.
- choices_short mirrors the order, ≤12 words, punchy imperative.

State update:
- Return `state_update` matching the schema (location, movement, inventory, ecosystem, audio, policy).
- Update nearest_intersection, heading, and blocks_moved (clamp to 1–3). Change area_id only if the chosen rail logically connects there.
- Keep policy.faces_obscured true. Audio intensity stays within 8–10.
- Emphasize exploration cues (beacons, data spikes, cultural remnants) rather than combat.

Output strictly JSON:
{
  "scenario_display": "...",
  "sora_prompt": "...",
  "choices": ["...", "...", "..."],
  "choices_short": ["...", "...", "..."],
  "state_update": {...}
}

Do not wrap output in markdown or explain your reasoning.
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
    last_error: Optional[Exception] = None
    for attempt in range(3):
        try:
            response = requests.post(
                RESPONSES_ENDPOINT,
                headers=headers,
                json=payload,
                timeout=180,
            )
            if response.status_code >= 400:
                raise RuntimeError(
                    f"Responses API error {response.status_code}: {response.text}"
                )
            data = response.json()
            break
        except (requests.exceptions.ReadTimeout, requests.exceptions.ConnectionError) as exc:
            last_error = exc
            logger.warning(
                "responses_create timeout attempt=%s model=%s", attempt + 1, model
            )
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)
        except Exception as exc:
            last_error = exc
            raise
    else:
        if last_error:
            raise last_error
        raise RuntimeError("Responses API returned no data after retries")

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
    choices_short_keys = [
        "choices_short",
        "choicesShort",
        "concise_choices",
        "conciseChoices",
        "short_choices",
        "shortChoices",
    ]
    state_update_keys = [
        "state_update",
        "stateUpdate",
        "state_json",
        "stateJson",
        "state",
    ]

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

    raw_choices_short = _pick(choices_short_keys)
    choices_short: List[str] = []
    if isinstance(raw_choices_short, list):
        choices_short = [str(choice).strip() for choice in raw_choices_short if str(choice).strip()]
    elif isinstance(raw_choices_short, str):
        fragments = re.split(r"[\n|]", raw_choices_short)
        choices_short = [frag.strip(" •-\t").strip() for frag in fragments if frag.strip()]

    if choices:
        while len(choices_short) < len(choices):
            fallback = choices[len(choices_short)]
            choices_short.append(str(fallback).strip())
        if len(choices_short) > len(choices):
            choices_short = choices_short[: len(choices)]
    else:
        while len(choices_short) < 3:
            choices_short.append(f"Choice {len(choices_short) + 1}")
        if len(choices_short) > 3:
            choices_short = choices_short[:3]

    normalized = dict(scene)
    normalized["scenario_display"] = scenario_display
    normalized["sora_prompt"] = sora_prompt
    normalized["choices"] = choices
    normalized["choices_short"] = choices_short
    normalized["_planner_missing_prompt"] = sora_prompt_missing
    normalized["_planner_missing_prompt_reason"] = sora_prompt_missing_reason
    state_update = _pick(state_update_keys)
    if isinstance(state_update, dict):
        normalized["state_update"] = state_update
    else:
        normalized["state_update"] = {}
    return normalized


def _format_list(items: List[str], empty_placeholder: str) -> str:
    if not items:
        return empty_placeholder
    return "\n".join(f"- {item}" for item in items)


def build_planner_user_input(
    *,
    base_prompt: str,
    state: Dict[str, Any],
    area: Optional[Dict[str, Any]],
    prior_prompts: List[str],
    prior_state_summaries: List[str],
    player_choice: Optional[str],
    stage_label: str,
) -> str:
    area_context = format_area_context(area)
    state_json = json.dumps(state, indent=2, ensure_ascii=False)
    prompts_section = (
        "(none yet — opening beat)"
        if not prior_prompts
        else "\n---\n".join(prior_prompts[-3:])
    )
    summaries_section = _format_list(
        prior_state_summaries[-5:], "- (no prior summaries recorded)"
    )
    choice_section = player_choice or "(root scene — no prior choice)"
    guidance_section = PROMPT_GUIDANCE or "(none)"

    return f"""
STAGE: {stage_label}

WORLD BASE PROMPT:
{base_prompt}

CURRENT STREET STATE (JSON):
{state_json}

CATALOG AREA CONTEXT:
{area_context}

PRIOR SORA PROMPTS (most recent last, max 3 shown):
{prompts_section}

PRIOR STATE SNAPSHOTS (human-readable, optional):
{summaries_section}

PLAYER CHOICE TRIGGERING THIS SCENE:
{choice_section}

ADDITIONAL WORLD GUIDANCE:
{guidance_section}

Return JSON with keys: scenario_display, sora_prompt, choices, choices_short, state_update.
""".strip()


def call_planner(
    *,
    api_key: str,
    model: str,
    base_prompt: str,
    state: Dict[str, Any],
    area: Optional[Dict[str, Any]],
    prior_prompts: List[str],
    prior_state_summaries: List[str],
    player_choice: Optional[str],
    stage_label: str,
) -> Dict[str, Any]:
    user_input = build_planner_user_input(
        base_prompt=base_prompt,
        state=state,
        area=area,
        prior_prompts=prior_prompts,
        prior_state_summaries=prior_state_summaries,
        player_choice=player_choice,
        stage_label=stage_label,
    )
    raw = responses_create(
        api_key=api_key,
        model=model,
        instructions=PLANNER_SYSTEM,
        user_input=user_input,
    )
    scene = normalize_scene_payload(extract_first_json(raw))
    scene["_raw_planner_output"] = raw.strip()
    scene["_planner_model"] = model
    scene["_planner_stage"] = stage_label
    return scene


def plan_initial_scene(api_key: str, base_prompt: str, model: str) -> dict:
    seed = seed_root_state()
    area = get_area(seed["location"].get("area_id"))
    return call_planner(
        api_key=api_key,
        model=model,
        base_prompt=base_prompt,
        state=seed,
        area=area,
        prior_prompts=[],
        prior_state_summaries=[],
        player_choice=None,
        stage_label="initial",
    )


def plan_next_scene(
    api_key: str,
    base_prompt: str,
    prior_sora_prompts: List[str],
    chosen_choice: str,
    state_summaries: List[str],
    model: str,
) -> dict:
    seed = seed_root_state()
    area = get_area(seed["location"].get("area_id"))
    return call_planner(
        api_key=api_key,
        model=model,
        base_prompt=base_prompt,
        state=seed,
        area=area,
        prior_prompts=prior_sora_prompts,
        prior_state_summaries=state_summaries,
        player_choice=chosen_choice,
        stage_label="legacy",
    )


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
