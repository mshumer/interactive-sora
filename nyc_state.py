"""NYC street-tracker state helpers.

This module defines the structured scene state that GPT-5 maintains for the
apocalyptic NYC experience. The format is intentionally constrained so prompts
can remain lean while carrying all the kinetic, location, inventory, and audio
context required by the video generator.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

STATE_VERSION = "1.0"

# Root spawn configuration (Times Square, hoverbike immediately available).
ROOT_AREA_ID = "manhattan_midtown_times_square"
ROOT_INTERSECTION = "W 42 St & 7th Ave"
ROOT_HEADING = "east"

# Persistent starter inventory — always available unless explicitly dropped by
# narrative (we keep that simple for now).
STARTER_INVENTORY = [
    "agile_hoverbike",
    "energy_shield",
    "grappling_hook",
    "ar_visor",
]


@dataclass
class StateField:
    key: str
    description: str
    required: bool = True


STATE_FIELDS: List[StateField] = [
    StateField(
        key="location",
        description=(
            "Dict containing: area_id, borough, district, neighborhood, "
            "nearest_intersection, heading, blocks_moved (1-3), time_of_day, weather"
        ),
    ),
    StateField(
        key="movement",
        description="Dict with tech_in_use (one of hoverbike/jetpack/grappling_hook/parkour_gear) and velocity_tier",
    ),
    StateField(
        key="inventory",
        description="Dict with current (list of item slugs) and in_use (single item slug)",
    ),
    StateField(
        key="ecosystem",
        description="Dict summarising adversaries/creatures/hazards active in the scene",
    ),
    StateField(
        key="audio",
        description="Dict with motif (string) and intensity (8-10)",
    ),
    StateField(
        key="policy",
        description="Dict that must include faces_obscured=true to satisfy generation rules",
    ),
]


def blank_state() -> Dict[str, Any]:
    """Return a state dict with empty values but locked structure."""

    return {
        "version": STATE_VERSION,
        "location": {
            "area_id": None,
            "borough": None,
            "district": None,
            "neighborhood": None,
            "nearest_intersection": None,
            "heading": None,
            "blocks_moved": None,
            "time_of_day": None,
            "weather": None,
        },
        "movement": {
            "tech_in_use": None,
            "velocity_tier": None,
        },
        "inventory": {
            "current": list(STARTER_INVENTORY),
            "in_use": None,
        },
        "ecosystem": {
            "adversaries": [],
            "creatures": [],
            "hazards": [],
        },
        "audio": {
            "motif": None,
            "intensity": 8,
        },
        "policy": {
            "faces_obscured": True,
        },
    }


def seed_root_state() -> Dict[str, Any]:
    """Return the initial state used when the player first spawns."""

    state = blank_state()
    state["location"].update(
        {
            "area_id": ROOT_AREA_ID,
            "borough": "Manhattan",
            "district": "Midtown West",
            "neighborhood": "Times Square",
            "nearest_intersection": ROOT_INTERSECTION,
            "heading": ROOT_HEADING,
            "blocks_moved": 0,
            "time_of_day": "perpetual neon midnight",
            "weather": "light rain and aerosol mist",
        }
    )
    state["movement"].update({"tech_in_use": "agile_hoverbike", "velocity_tier": "fast"})
    state["inventory"]["in_use"] = "agile_hoverbike"
    state["ecosystem"].update(
        {
            "adversaries": ["hijacked signage drones", "riot-control mechs"],
            "creatures": [],
            "hazards": ["electrical surges", "falling billboard panels"],
        }
    )
    state["audio"].update(
        {
            "motif": "Stuttering synth bass with siren-tail glissandos and pounding percussion",
            "intensity": 9,
        }
    )
    return state


def clamp_blocks(blocks: Optional[int]) -> int:
    """Clamp the 1-3 block movement budget."""

    if not isinstance(blocks, int):
        return 1
    return max(1, min(3, blocks))


def normalize_state(raw: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Ensure any provided state adheres to the required schema."""

    state = blank_state()
    if not isinstance(raw, dict):
        return state

    for field in STATE_FIELDS:
        value = raw.get(field.key)
        if isinstance(value, dict):
            state[field.key].update(value)

    # Ensure defaults/coercions.
    state["version"] = raw.get("version", STATE_VERSION)

    location = state["location"]
    location["blocks_moved"] = clamp_blocks(location.get("blocks_moved"))
    heading = location.get("heading")
    if isinstance(heading, str):
        location["heading"] = heading.lower()

    inventory = state["inventory"]
    current = inventory.get("current")
    if not isinstance(current, list):
        current = list(STARTER_INVENTORY)
    inventory["current"] = sorted({*STARTER_INVENTORY, *(item for item in current if isinstance(item, str))})
    in_use = inventory.get("in_use")
    if in_use not in inventory["current"]:
        inventory["in_use"] = inventory["current"][0]

    policy = state["policy"]
    policy["faces_obscured"] = True

    audio = state["audio"]
    intensity = audio.get("intensity")
    if not isinstance(intensity, int):
        intensity = 9
    audio["intensity"] = max(8, min(10, intensity))

    movement = state["movement"]
    tech = movement.get("tech_in_use")
    if tech not in {"agile_hoverbike", "jetpack", "grappling_hook", "parkour_gear"}:
        movement["tech_in_use"] = "agile_hoverbike"
    movement["velocity_tier"] = "fast"

    return state


def update_state(prev: Optional[Dict[str, Any]], update: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Merge a planner-provided state update onto the previous state."""

    base = normalize_state(prev)
    if not isinstance(update, dict):
        return base

    for key in ("location", "movement", "ecosystem", "audio", "policy"):
        incoming = update.get(key)
        if isinstance(incoming, dict):
            for sub_key, value in incoming.items():
                if value is None:
                    continue
                base[key][sub_key] = value

    inventory_update = update.get("inventory")
    if isinstance(inventory_update, dict):
        current = inventory_update.get("current")
        if isinstance(current, list):
            merged = set(base["inventory"]["current"])
            merged.update(item for item in current if isinstance(item, str))
            merged.update(STARTER_INVENTORY)
            base["inventory"]["current"] = sorted(merged)
        in_use = inventory_update.get("in_use")
        if isinstance(in_use, str):
            base["inventory"]["in_use"] = in_use

    if isinstance(update.get("version"), str):
        base["version"] = update["version"]

    return normalize_state(base)
