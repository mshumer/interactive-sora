"""Utilities for loading and presenting the NYC catalog."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

CATALOG_PATH = Path("data/nyc_catalog.json")


class CatalogNotFound(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_catalog() -> Dict[str, Any]:
    if not CATALOG_PATH.exists():
        raise CatalogNotFound(f"Catalog file missing: {CATALOG_PATH}")
    with CATALOG_PATH.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    by_id = {area["id"]: area for area in data.get("areas", [])}
    data["areas_by_id"] = by_id
    return data


def get_area(area_id: Optional[str]) -> Optional[Dict[str, Any]]:
    if not area_id:
        return None
    catalog = load_catalog()
    return catalog["areas_by_id"].get(area_id)


def list_adjacent(area_id: Optional[str]) -> List[str]:
    area = get_area(area_id)
    if not area:
        return []
    return list(area.get("adjacent", []))


def format_area_context(area: Optional[Dict[str, Any]]) -> str:
    """Produce a compact bullet list describing the selected catalog area."""

    if not area:
        return "(area unknown)"

    visuals = area.get("visuals", {})
    ecosystem = area.get("ecosystem", {})
    movement = area.get("movement", {})

    bullets = [
        f"- Area ID: {area.get('id')} | {area.get('borough')} › {area.get('district')} › {area.get('neighborhood')}",
        f"- Palette: {', '.join(visuals.get('palette', []))}",
        f"- Lighting: {visuals.get('lighting')}",
        f"- Architecture: {', '.join(visuals.get('architecture', []))}",
        f"- Set dressing: {', '.join(visuals.get('set_dressing', []))}",
        f"- Adversaries: {', '.join(ecosystem.get('adversaries', []))}",
        f"- Hazards: {', '.join(ecosystem.get('hazards', []))}",
        f"- Creatures: {', '.join(ecosystem.get('creatures', []))}",
        f"- Audio motif: {area.get('audio_motif')}",
        f"- Preferred traversal: {', '.join(movement.get('preferred_tech', []))}",
        f"- Traversal calls: {', '.join(movement.get('traversal_calls', []))}",
        f"- Landmarks: {', '.join(area.get('landmarks', []))}",
        f"- Adjacent areas (IDs): {', '.join(area.get('adjacent', []))}",
        f"- Sora tokens: {', '.join(area.get('sora_tokens', []))}",
        f"- Street essence: {area.get('street_essence')}",
    ]
    return "\n".join(bullet for bullet in bullets if bullet and not bullet.endswith(': '))

