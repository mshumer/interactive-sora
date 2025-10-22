from __future__ import annotations

import hashlib
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Dict, List, Optional

from argon2 import PasswordHasher, exceptions as argon2_exceptions
from fastapi import APIRouter, Form, HTTPException, Query, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from itsdangerous import BadSignature, SignatureExpired, TimestampSigner
from pydantic import BaseModel
from sqlalchemy import delete, func, or_, select
from sqlalchemy.orm import Session

from database import session_scope
from models import AdminSecret, Scene, SceneMetric, SceneStatus
from storage import build_storage_client


ADMIN_COOKIE_NAME = "admin_session"
SESSION_MAX_AGE_SECONDS = 14 * 24 * 3600
SIGNER_STATIC_SALT = "sora-control-admin-session"

LOGIN_WINDOW_SECONDS = 15 * 60
LOGIN_MAX_FAILURES = 5

_password_hasher = PasswordHasher()
_login_failures: dict[str, list[float]] = {}

templates = Jinja2Templates(directory="templates")
router = APIRouter(prefix="/admin", tags=["admin"])
WORLD_ID = os.environ.get("WORLD_ID", "default")
storage_client = build_storage_client()
logger = logging.getLogger("admin_dashboard")
_request_cancel_callable = None


@dataclass
class AdminAuthState:
    password_hash: str


def _load_admin_secret(session: Session) -> Optional[AdminSecret]:
    stmt = select(AdminSecret).limit(1)
    return session.execute(stmt).scalars().first()


def _ensure_admin_secret() -> AdminAuthState:
    with session_scope() as session:
        secret = _load_admin_secret(session)
        if secret is None or not secret.password_hash:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Admin password not configured. Run tools/set_admin_password.py first.",
            )
        return AdminAuthState(password_hash=secret.password_hash)


def _derive_signer(password_hash: str) -> TimestampSigner:
    material = f"{password_hash}:{SIGNER_STATIC_SALT}".encode("utf-8")
    key = hashlib.sha256(material).hexdigest()
    return TimestampSigner(key)


def create_session_token() -> str:
    state = _ensure_admin_secret()
    signer = _derive_signer(state.password_hash)
    signed = signer.sign("admin").decode("utf-8")
    return signed


def verify_session_token(token: str) -> bool:
    try:
        state = _ensure_admin_secret()
    except HTTPException:
        return False
    signer = _derive_signer(state.password_hash)
    try:
        value = signer.unsign(token, max_age=SESSION_MAX_AGE_SECONDS)
    except (SignatureExpired, BadSignature):
        return False
    return value.decode("utf-8") == "admin"


def verify_password(candidate: str) -> bool:
    state = _ensure_admin_secret()
    try:
        _password_hasher.verify(state.password_hash, candidate)
        return True
    except argon2_exceptions.VerifyMismatchError:
        return False
    except argon2_exceptions.VerificationError:
        return False


def _prune_failures(entries: list[float], now: float) -> list[float]:
    return [stamp for stamp in entries if now - stamp < LOGIN_WINDOW_SECONDS]


def register_login_failure(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _login_failures.get(ip, [])
    bucket = _prune_failures(bucket, now)
    bucket.append(now)
    _login_failures[ip] = bucket


def clear_login_failures(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    _login_failures.pop(ip, None)


def ensure_login_not_rate_limited(request: Request) -> None:
    ip = request.client.host if request.client else "unknown"
    now = time.time()
    bucket = _prune_failures(_login_failures.get(ip, []), now)
    _login_failures[ip] = bucket
    if len(bucket) >= LOGIN_MAX_FAILURES:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many failed login attempts. Try again later.",
        )


def _cookie_secure_flag(request: Request) -> bool:
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    return proto.lower() == "https"


def set_auth_cookie(response: Response, request: Request, token: str) -> None:
    response.set_cookie(
        ADMIN_COOKIE_NAME,
        token,
        max_age=SESSION_MAX_AGE_SECONDS,
        httponly=True,
        secure=_cookie_secure_flag(request),
        samesite="strict",
    )


def clear_auth_cookie(response: Response, request: Request) -> None:
    response.delete_cookie(
        ADMIN_COOKIE_NAME,
        httponly=True,
        secure=_cookie_secure_flag(request),
        samesite="strict",
    )


def is_authenticated(request: Request) -> bool:
    token = request.cookies.get(ADMIN_COOKIE_NAME)
    if not token:
        return False
    return verify_session_token(token)


def require_admin(request: Request) -> None:
    if not is_authenticated(request):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Authentication required")


def _normalize_path(value: str) -> str:
    if value is None:
        return ""
    value = value.strip().strip("/")
    if value == "":
        return ""
    if not re.fullmatch(r"(\d+)(/\d+)*", value):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid path format")
    return value


def _resolve_asset_url(stored_value: Optional[str], variant: str) -> Optional[str]:
    if not stored_value:
        return None
    try:
        return storage_client.resolve_url(stored_value, variant=variant)
    except AttributeError:
        return stored_value


def _scene_to_dict(scene: Scene) -> Dict[str, object]:
    return {
        "worldId": scene.world_id,
        "path": scene.path,
        "depth": scene.depth,
        "status": scene.status.value if scene.status else None,
        "scenarioDisplay": scene.scenario_display,
        "videoUrl": _resolve_asset_url(scene.video_url, "video"),
        "posterUrl": _resolve_asset_url(scene.poster_url, "poster"),
        "contextVideoUrl": _resolve_asset_url(scene.context_video_url, "context"),
        "updatedAt": scene.updated_at,
        "startedAt": scene.started_at,
        "progress": getattr(scene, "progress", None),
        "progressUpdatedAt": getattr(scene, "progress_updated_at", None),
        "choices": scene.choices if isinstance(scene.choices, list) else None,
        "choicesShort": scene.choices_short if isinstance(scene.choices_short, list) else None,
    }


@router.get("", response_class=HTMLResponse)
def admin_home(request: Request) -> HTMLResponse:
    authenticated = is_authenticated(request)
    template = "admin/index.html" if authenticated else "admin/login.html"
    context = {
        "request": request,
        "authenticated": authenticated,
        "login_error": None,
    }
    return templates.TemplateResponse(template, context)


@router.post("/login")
def admin_login(request: Request, password: str = Form(...)) -> Response:
    ensure_login_not_rate_limited(request)
    if not verify_password(password):
        register_login_failure(request)
        context = {
            "request": request,
            "authenticated": False,
            "login_error": "Invalid password.",
        }
        return templates.TemplateResponse(
            "admin/login.html", context, status_code=status.HTTP_401_UNAUTHORIZED
        )

    clear_login_failures(request)
    token = create_session_token()
    response = RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)
    set_auth_cookie(response, request, token)
    return response


@router.post("/logout")
def admin_logout(request: Request) -> Response:
    response = RedirectResponse(url="/admin", status_code=status.HTTP_303_SEE_OTHER)
    clear_auth_cookie(response, request)
    return response


@router.get("/api/scenes")
def list_scenes(
    request: Request,
    prefix: str = Query("", max_length=512),
    limit: int = Query(200, ge=1, le=500),
    status_filter: Optional[str] = Query(None, alias="status"),
):
    require_admin(request)
    normalized_prefix = _normalize_path(prefix)

    filters = [Scene.world_id == WORLD_ID]
    if normalized_prefix:
        filters.append(
            or_(Scene.path == normalized_prefix, Scene.path.like(f"{normalized_prefix}/%"))
        )

    if status_filter:
        try:
            status_value = SceneStatus(status_filter)
        except ValueError:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid status filter")
        filters.append(Scene.status == status_value)

    with session_scope() as session:
        total = session.execute(select(func.count()).where(*filters)).scalar_one()
        stmt = select(Scene).where(*filters).order_by(Scene.path).limit(limit)
        scenes = session.execute(stmt).scalars().all()

    return {
        "total": total,
        "items": [_scene_to_dict(scene) for scene in scenes],
    }


@router.get("/api/scene")
def get_scene_details(request: Request, path: str = Query("", max_length=512)):
    require_admin(request)
    normalized_path = _normalize_path(path)

    with session_scope() as session:
        stmt = select(Scene).where(Scene.world_id == WORLD_ID, Scene.path == normalized_path)
        scene = session.execute(stmt).scalars().first()
        if scene is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Scene not found")

        max_children = max(len(scene.choices or []), 3)
        child_paths = [scene.child_path(idx) for idx in range(max_children)]
        children: Dict[str, SceneStatus] = {}
        if child_paths:
            child_stmt = select(Scene.path, Scene.status).where(
                Scene.world_id == WORLD_ID, Scene.path.in_(child_paths)
            )
            for child_path, child_status in session.execute(child_stmt).all():
                children[child_path] = child_status

        metrics_stmt = (
            select(func.coalesce(func.sum(SceneMetric.storage_bytes), 0))
            .where(SceneMetric.scene_id == scene.id)
        )
        storage_bytes = session.execute(metrics_stmt).scalar_one()

    children_info: List[Dict[str, object]] = []
    for child_path in child_paths:
        status_value = children.get(child_path, SceneStatus.PENDING)
        children_info.append({
            "path": child_path,
            "status": status_value.value if isinstance(status_value, SceneStatus) else status_value,
        })

    return {
        "scene": _scene_to_dict(scene),
        "children": children_info,
        "storageBytes": int(storage_bytes or 0),
    }


class ResetRequest(BaseModel):
    path: str = ""
    inclusive: bool = True


def _cancel_generation(world_id: str, path: str) -> None:
    global _request_cancel_callable
    if _request_cancel_callable is None:
        try:
            from app import request_cancel as cancel_callable  # type: ignore

            _request_cancel_callable = cancel_callable
        except Exception:
            _request_cancel_callable = False  # type: ignore[assignment]
    if callable(_request_cancel_callable):
        try:
            _request_cancel_callable(world_id, path)
        except Exception:  # pragma: no cover - defensive
            logger.debug("Failed to cancel generation for %s/%s", world_id, path, exc_info=True)


@router.post("/api/reset")
def reset_path(request: Request, payload: ResetRequest) -> Dict[str, int]:
    require_admin(request)
    if not payload.inclusive:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Only inclusive resets are supported")

    normalized_path = _normalize_path(payload.path)

    filters = [Scene.world_id == WORLD_ID]
    if normalized_path:
        filters.append(or_(Scene.path == normalized_path, Scene.path.like(f"{normalized_path}/%")))

    with session_scope() as session:
        stmt = select(Scene).where(*filters)
        scenes = session.execute(stmt).scalars().all()
        if not scenes:
            return {"deletedScenes": 0, "deletedBytes": 0}

        scene_ids = [scene.id for scene in scenes]
        target_paths = [scene.path for scene in scenes]
        metrics_stmt = select(func.coalesce(func.sum(SceneMetric.storage_bytes), 0)).where(
            SceneMetric.scene_id.in_(scene_ids)
        )
        total_bytes = session.execute(metrics_stmt).scalar_one()

    for path in target_paths:
        _cancel_generation(WORLD_ID, path)

    key_prefixes = {f"{WORLD_ID}/{path or 'root'}" for path in target_paths}
    for key_prefix in key_prefixes:
        try:
            storage_client.delete(key_prefix)
        except Exception as exc:  # pragma: no cover - best effort cleanup
            logger.warning("Failed to delete assets for prefix %s: %s", key_prefix, exc)

    with session_scope() as session:
        session.execute(delete(SceneMetric).where(SceneMetric.scene_id.in_(scene_ids)))
        session.execute(delete(Scene).where(Scene.id.in_(scene_ids)))

    return {
        "deletedScenes": len(target_paths),
        "deletedBytes": int(total_bytes or 0),
    }
