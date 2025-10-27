from __future__ import annotations

import hashlib
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

from fastapi import Request

try:  # pragma: no cover - optional dependency in some environments
    from mixpanel import Mixpanel  # type: ignore
except Exception:  # pragma: no cover - allow the project to run without Mixpanel installed
    Mixpanel = None  # type: ignore

logger = logging.getLogger("mixpanel")

_token = os.environ.get("MIXPANEL_TOKEN", "").strip()
_client: Optional[Mixpanel] = None
if _token and Mixpanel is not None:
    try:
        _client = Mixpanel(_token)
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("failed to initialise mixpanel client: %s", exc)
        _client = None


def is_enabled() -> bool:
    return _client is not None


def _resolve_ip(request: Request) -> str:
    forwarded_for = request.headers.get("x-forwarded-for") or request.headers.get("X-Forwarded-For")
    if forwarded_for:
        ip = forwarded_for.split(",")[0].strip()
        if ip:
            return ip
    client = request.client
    if client and client.host:
        return client.host
    return "unknown"


def fingerprint_from_ip(ip_address: str) -> str:
    if not ip_address or ip_address == "unknown":
        return "anonymous"
    digest = hashlib.sha256(ip_address.encode("utf-8")).hexdigest()
    return digest


def fingerprint_from_request(request: Request) -> Tuple[str, str]:
    ip_address = _resolve_ip(request)
    fingerprint = fingerprint_from_ip(ip_address)
    return fingerprint, ip_address


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def track_with_request(
    request: Request,
    event_name: str,
    properties: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    distinct_id, ip_address = fingerprint_from_request(request)
    track_with_fingerprint(distinct_id, ip_address, event_name, properties)
    return distinct_id


def track_with_fingerprint(
    distinct_id: Optional[str],
    ip_address: Optional[str],
    event_name: str,
    properties: Optional[Dict[str, Any]] = None,
) -> None:
    if not distinct_id or not _client:
        return
    payload: Dict[str, Any] = {"timestamp": _now_iso()}
    if properties:
        payload.update(properties)
    if ip_address and ip_address != "unknown":
        payload.setdefault("$ip", ip_address)
        payload.setdefault("ip_address", ip_address)
    try:
        _client.track(distinct_id, event_name, payload)
    except Exception as exc:  # pragma: no cover - keep analytics non-blocking
        logger.warning("mixpanel track failed for %s: %s", event_name, exc)


def identify(distinct_id: str, properties: Optional[Dict[str, Any]] = None) -> None:
    if not distinct_id or not _client:
        return
    props = dict(properties or {})
    try:
        _client.people_set(distinct_id, props)
    except Exception:
        # do not raise - identify is best-effort
        pass
