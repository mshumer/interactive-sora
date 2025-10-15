import json
import mimetypes
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from uuid import uuid4

import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import imageio_ffmpeg  # type: ignore

    FFMPEG_BIN = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:  # pragma: no cover - optional dependency
    FFMPEG_BIN = None

APP_TITLE = "Sora Choose-Your-Own Adventure API"

VIDEO_DIR = Path("sora_cyoa_videos")
FRAME_DIR = Path("sora_cyoa_frames")
VIDEO_DIR.mkdir(parents=True, exist_ok=True)
FRAME_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_VIDEO_SIZE = "1280x720"
DEFAULT_SECONDS = 8
ALLOWED_SECONDS = [4, 8, 12]

OPENAI_API_BASE = "https://api.openai.com/v1"
SORA_VIDEOS_ENDPOINT = f"{OPENAI_API_BASE}/videos"
RESPONSES_ENDPOINT = f"{OPENAI_API_BASE}/responses"

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

5) Output strictly JSON. No markdown, no commentary, no code fences.
""".strip()


class CreateSessionRequest(BaseModel):
    api_key: str = Field(..., alias="apiKey")
    planner_model: str = Field(..., alias="plannerModel")
    sora_model: str = Field(..., alias="soraModel")
    video_size: str = Field(DEFAULT_VIDEO_SIZE, alias="videoSize")
    base_prompt: str = Field(..., alias="basePrompt")
    max_steps: int = Field(10, alias="maxSteps")

    @validator("max_steps")
    def validate_max_steps(cls, value: int) -> int:
        if not 1 <= value <= 30:
            raise ValueError("maxSteps must be between 1 and 30")
        return value


class ChoiceRequest(BaseModel):
    choice_index: int = Field(..., alias="choiceIndex")

    @validator("choice_index")
    def validate_choice_index(cls, value: int) -> int:
        if value not in (0, 1, 2):
            raise ValueError("choiceIndex must be 0, 1, or 2")
        return value


class StoryStepResponse(BaseModel):
    scene_number: int = Field(..., alias="sceneNumber")
    scenario_display: str = Field(..., alias="scenarioDisplay")
    sora_prompt: str = Field(..., alias="soraPrompt")
    choices: List[str]
    choice_index: Optional[int] = Field(None, alias="choiceIndex")
    video_url: Optional[str] = Field(None, alias="videoUrl")
    poster_url: Optional[str] = Field(None, alias="posterUrl")
    video_id: Optional[str] = Field(None, alias="videoId")
    planner_missing_prompt: bool = Field(False, alias="plannerMissingPrompt")
    planner_missing_prompt_reason: str = Field("", alias="plannerMissingPromptReason")


class SessionResponse(BaseModel):
    session_id: str = Field(..., alias="sessionId")
    story: List[StoryStepResponse]
    step_count: int = Field(..., alias="stepCount")
    max_steps: int = Field(..., alias="maxSteps")
    has_remaining_steps: bool = Field(..., alias="hasRemainingSteps")


@dataclass
class SessionConfig:
    api_key: str
    planner_model: str
    sora_model: str
    video_size: str
    base_prompt: str
    max_steps: int


@dataclass
class SessionState:
    config: SessionConfig
    story: List[Dict[str, Any]] = field(default_factory=list)
    step_count: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)


SESSIONS: Dict[str, SessionState] = {}

app = FastAPI(title=APP_TITLE)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/media/videos", StaticFiles(directory=str(VIDEO_DIR)), name="videos")
app.mount("/media/frames", StaticFiles(directory=str(FRAME_DIR)), name="frames")


def _auth_headers(api_key: str) -> Dict[str, str]:
    if not api_key:
        raise RuntimeError("OpenAI API key is required")
    return {
        "Authorization": f"Bearer {api_key}",
    }


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
    user_input = f"""
TASK: Create the opening scene with three choices.

BASE PROMPT:
{base_prompt}

Shot length: 8 seconds.
Return JSON with keys: scenario_display, sora_prompt, choices (3).
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
    model: str,
) -> dict:
    prior_joined = "\n\n---\n\n".join(prior_sora_prompts)
    user_input = f"""
TASK: Create the next scene with three choices, continuing the story.

BASE PROMPT:
{base_prompt}

PRIOR SORA PROMPTS (in order; each was used to generate an 8s video):
{prior_joined}

PLAYER'S CHOSEN ACTION TO CONTINUE:
{chosen_choice}

Note: The next 8-second shot MUST begin exactly from the final frame of the previous shot,
preserving continuity (subjects, camera position, lighting, motion direction), unless the chosen action implies a change.

Return JSON with keys: scenario_display, sora_prompt, choices (3).
""".strip()
    raw = responses_create(api_key=api_key, model=model, instructions=PLANNER_SYSTEM, user_input=user_input)
    scene = normalize_scene_payload(extract_first_json(raw))
    scene["_raw_planner_output"] = raw.strip()
    scene["_planner_model"] = model
    scene["_planner_stage"] = "continuation"
    return scene


def _guess_mime(path: Path) -> str:
    mime = mimetypes.guess_type(str(path))[0]
    return mime or "application/octet-stream"


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
    if input_reference_path:
        files["input_reference"] = (
            input_reference_path.name,
            open(input_reference_path, "rb"),  # noqa: SIM115 - leave open for requests to stream
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


def sora_poll_until_complete(api_key: str, job: dict) -> dict:
    video = job
    video_id = video["id"]
    while video.get("status") in ("queued", "in_progress"):
        time.sleep(2)
        video = sora_retrieve_video(api_key, video_id)

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


def normalize_seconds(secs: int) -> int:
    return min(ALLOWED_SECONDS, key=lambda value: abs(value - int(secs)))


def generate_scene_video(
    api_key: str,
    sora_prompt: str,
    model: str,
    size: str,
    seconds: int,
    input_reference: Optional[Path],
) -> (str, Path, Path):
    seconds = normalize_seconds(seconds)
    job = sora_create_video(
        api_key=api_key,
        sora_prompt=sora_prompt,
        model=model,
        size=size,
        seconds=seconds,
        input_reference_path=input_reference,
    )

    video = sora_poll_until_complete(api_key, job)
    video_id = video["id"]

    video_path = VIDEO_DIR / f"{video_id}.mp4"
    sora_download_content(api_key, video_id, video_path, variant="video")

    last_frame_path = FRAME_DIR / f"{video_id}_last.jpg"
    extract_last_frame(video_path, last_frame_path)
    return video_id, video_path, last_frame_path


def create_story_item(scene: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "scenario_display": scene["scenario_display"],
        "sora_prompt": scene["sora_prompt"],
        "choices": scene["choices"],
        "choice_index": None,
        "video_id": None,
        "video_path": None,
        "last_frame_path": None,
        "planner_missing_prompt": scene.get("_planner_missing_prompt", False),
        "planner_missing_prompt_reason": scene.get("_planner_missing_prompt_reason", ""),
        "planner_raw_output": scene.get("_raw_planner_output", ""),
        "planner_model": scene.get("_planner_model", ""),
        "planner_stage": scene.get("_planner_stage", ""),
    }


def build_story_payload(story: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    payload: List[Dict[str, Any]] = []
    for idx, step in enumerate(story, start=1):
        video_url = None
        if step.get("video_path"):
            video_url = f"/media/videos/{Path(step['video_path']).name}"

        poster_url = None
        if step.get("last_frame_path"):
            poster_url = f"/media/frames/{Path(step['last_frame_path']).name}"

        payload.append(
            {
                "sceneNumber": idx,
                "scenarioDisplay": step.get("scenario_display", ""),
                "soraPrompt": step.get("sora_prompt", ""),
                "choices": step.get("choices", []),
                "choiceIndex": step.get("choice_index"),
                "videoUrl": video_url,
                "posterUrl": poster_url,
                "videoId": step.get("video_id"),
                "plannerMissingPrompt": step.get("planner_missing_prompt", False),
                "plannerMissingPromptReason": step.get("planner_missing_prompt_reason", ""),
            }
        )
    return payload


def serialize_session(session_id: str, state: SessionState) -> SessionResponse:
    story_payload = build_story_payload(state.story)
    response = SessionResponse(
        sessionId=session_id,
        story=[StoryStepResponse(**item) for item in story_payload],
        stepCount=state.step_count,
        maxSteps=state.config.max_steps,
        hasRemainingSteps=state.step_count < state.config.max_steps,
    )
    return response


def get_session_or_404(session_id: str) -> SessionState:
    state = SESSIONS.get(session_id)
    if not state:
        raise HTTPException(status_code=404, detail="Session not found")
    return state


@app.get("/health")
def healthcheck() -> Dict[str, str]:
    return {"status": "ok"}


@app.post("/api/session", response_model=SessionResponse)
def create_session(payload: CreateSessionRequest) -> SessionResponse:
    config = SessionConfig(
        api_key=payload.api_key.strip(),
        planner_model=payload.planner_model.strip() or "gpt-5",
        sora_model=payload.sora_model.strip() or "sora-2",
        video_size=payload.video_size.strip() or DEFAULT_VIDEO_SIZE,
        base_prompt=payload.base_prompt.strip(),
        max_steps=payload.max_steps,
    )

    if not config.api_key:
        raise HTTPException(status_code=400, detail="API key is required")
    if not config.base_prompt:
        raise HTTPException(status_code=400, detail="Base prompt is required")

    session_id = uuid4().hex
    state = SessionState(config=config)
    SESSIONS[session_id] = state

    try:
        scene = plan_initial_scene(
            api_key=config.api_key,
            base_prompt=config.base_prompt,
            model=config.planner_model,
        )
        story_item = create_story_item(scene)

        if not story_item["planner_missing_prompt"]:
            video_id, video_path, last_frame_path = generate_scene_video(
                api_key=config.api_key,
                sora_prompt=story_item["sora_prompt"],
                model=config.sora_model,
                size=config.video_size,
                seconds=DEFAULT_SECONDS,
                input_reference=None,
            )
            story_item["video_id"] = video_id
            story_item["video_path"] = str(video_path)
            story_item["last_frame_path"] = str(last_frame_path)
        else:
            story_item["planner_missing_prompt"] = True

        state.story.append(story_item)
    except Exception as exc:
        SESSIONS.pop(session_id, None)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return serialize_session(session_id, state)


@app.get("/api/session")
def list_sessions() -> Dict[str, Any]:
    return {
        "sessions": [
            {
                "sessionId": session_id,
                "stepCount": state.step_count,
                "maxSteps": state.config.max_steps,
            }
            for session_id, state in SESSIONS.items()
        ]
    }


@app.get("/api/session/{session_id}", response_model=SessionResponse)
def get_session(session_id: str) -> SessionResponse:
    state = get_session_or_404(session_id)
    return serialize_session(session_id, state)


@app.post("/api/session/{session_id}/choice", response_model=SessionResponse)
def advance_story(session_id: str, payload: ChoiceRequest) -> SessionResponse:
    state = get_session_or_404(session_id)
    with state.lock:
        if not state.story:
            raise HTTPException(status_code=400, detail="Session has not been initialized")

        current = state.story[-1]
        if current.get("choice_index") is not None:
            raise HTTPException(status_code=400, detail="Current scene already has a recorded choice")

        current["choice_index"] = payload.choice_index
        chosen_choice = current["choices"][payload.choice_index]
        state.step_count += 1

        if state.step_count >= state.config.max_steps:
            return serialize_session(session_id, state)

        prior_sora_prompts = [step["sora_prompt"] for step in state.story]

        try:
            next_scene = plan_next_scene(
                api_key=state.config.api_key,
                base_prompt=state.config.base_prompt,
                prior_sora_prompts=prior_sora_prompts,
                chosen_choice=chosen_choice,
                model=state.config.planner_model,
            )
            next_item = create_story_item(next_scene)

            input_reference = None
            if current.get("last_frame_path"):
                input_reference = Path(current["last_frame_path"])

            if not next_item["planner_missing_prompt"]:
                video_id, video_path, last_frame_path = generate_scene_video(
                    api_key=state.config.api_key,
                    sora_prompt=next_item["sora_prompt"],
                    model=state.config.sora_model,
                    size=state.config.video_size,
                    seconds=DEFAULT_SECONDS,
                    input_reference=input_reference,
                )
                next_item["video_id"] = video_id
                next_item["video_path"] = str(video_path)
                next_item["last_frame_path"] = str(last_frame_path)
            else:
                next_item["planner_missing_prompt"] = True

            state.story.append(next_item)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=str(exc)) from exc

    return serialize_session(session_id, state)


if __name__ == "__main__":  # pragma: no cover - convenience for local dev
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
