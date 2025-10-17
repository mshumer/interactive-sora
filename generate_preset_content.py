"""Utility script to pre-generate cinematic starter trees for the three default presets.

Usage:
    OPENAI_API_KEY=sk-... python generate_preset_content.py

The script will create a directory `prebaked_content/<preset_slug>/` containing:
    - mp4 videos for depth-0 to depth-2 scenes (13 clips per preset)
    - JPG poster frames for each clip
    - A manifest.json describing the branching tree and relative asset paths

It reuses the helper functions from app.py to stay consistent with runtime logic.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

from app import DEFAULT_SECONDS, generate_scene_video, plan_initial_scene, plan_next_scene  # type: ignore

try:
    from tqdm import tqdm  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    tqdm = None

OUTPUT_ROOT = Path("prebaked_content")
OUTPUT_ROOT.mkdir(exist_ok=True)


@dataclass
class SceneNode:
    path: List[int]
    trigger_choice: Optional[str]
    scenario_display: str
    sora_prompt: str
    choices: List[str]
    choices_short: List[str]
    video_relpath: Optional[str]
    poster_relpath: Optional[str]
    children: List["SceneNode"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, object]:
        return {
            "path": self.path,
            "triggerChoice": self.trigger_choice,
            "scenarioDisplay": self.scenario_display,
            "soraPrompt": self.sora_prompt,
            "choices": self.choices,
            "choicesShort": self.choices_short,
            "video": self.video_relpath,
            "poster": self.poster_relpath,
            "children": [child.to_dict() for child in self.children],
        }


PRESETS: Dict[str, str] = {
    "neon_midnight_walk": (
        "Photorealistic rainy midnight street in a near-future Tokyo district. "
        "Camera glides beside a lone figure walking past noodle stalls, holographic billboards, "
        "flickering neon reflections in puddles, steam rising from manholes, and curious onlookers in reflective raincoats. "
        "Emphasize cinematic lighting, wet asphalt textures, and the sense that anything could emerge from the crowd."
    ),
    "verdant_quest": (
        "Photorealistic enchanted forest adventure at golden hour. "
        "Follow an explorer in weathered travel gear trekking through towering moss-covered trees, shafts of light cutting through mist, "
        "ancient stone ruins hidden under vines, and distant drumbeats hinting at hidden civilizations. "
        "The air feels alive with curiosity and imminent discovery."
    ),
    "house_of_echoes": (
        "Photorealistic claustrophobic horror inside a decaying Victorian mansion. "
        "The protagonist moves room to room; each doorway reveals a new terror: portraits whose eyes bleed shadows, a nursery of toys that whisper, "
        "a dining hall table set for spirits. Lighting is minimal, with handheld flashlight beams and erratic power surges casting unsettling moving silhouettes."
    ),
}

MAX_DEPTH = 2  # root (0) + first layer (1) + second layer (2)


def slug_path(path: List[int]) -> str:
    if not path:
        return "scene_root"
    return "scene_" + "_".join(str(idx) for idx in path)


def copy_assets(video_path: Path, frame_path: Path, target_dir: Path, slug: str) -> (str, str):
    target_dir.mkdir(parents=True, exist_ok=True)
    target_video = target_dir / f"{slug}.mp4"
    target_poster = target_dir / f"{slug}.jpg"
    shutil.copy2(video_path, target_video)
    shutil.copy2(frame_path, target_poster)
    video_rel = str(target_video.relative_to(OUTPUT_ROOT))
    poster_rel = str(target_poster.relative_to(OUTPUT_ROOT))
    return video_rel, poster_rel


@dataclass
class ProgressHandle:
    total: int
    desc: str
    position: int = 0
    use_tqdm: bool = tqdm is not None

    def __post_init__(self) -> None:
        self.use_tqdm = self.use_tqdm and (tqdm is not None)
        if self.use_tqdm:
            self._bar = tqdm(total=self.total, desc=self.desc, position=self.position, leave=(self.position == 0))
        else:
            self.current = 0

    def advance(self, detail: str = "") -> None:
        if self.use_tqdm:
            if detail:
                self._bar.set_postfix_str(detail, refresh=False)
            self._bar.update(1)
        else:
            self.current += 1
            pct = self.current / self.total if self.total else 1.0
            bar_len = 28
            filled = int(bar_len * pct)
            bar = "#" * filled + "-" * (bar_len - filled)
            msg = f"{self.desc}: [{bar}] {pct*100:5.1f}% ({self.current}/{self.total})"
            if detail:
                msg += f" {detail}"
            print(msg)

    def close(self) -> None:
        if self.use_tqdm:
            self._bar.close()


def build_tree(
    api_key: str,
    planner_model: str,
    sora_model: str,
    video_size: str,
    preset_tracker: ProgressHandle,
    overall_tracker: ProgressHandle,
    preset_slug: str,
    base_prompt: str,
    path: List[int],
    prior_prompts: List[str],
    trigger_choice: Optional[str],
    parent_last_frame: Optional[Path],
    depth: int,
) -> SceneNode:
    if depth == 0:
        scene = plan_initial_scene(api_key=api_key, base_prompt=base_prompt, model=planner_model)
    else:
        scene = plan_next_scene(
            api_key=api_key,
            base_prompt=base_prompt,
            prior_sora_prompts=prior_prompts,
            chosen_choice=trigger_choice or "",
            state_summaries=[],
            model=planner_model,
        )

    node_slug = slug_path(path)

    video_relpath: Optional[str] = None
    poster_relpath: Optional[str] = None
    last_frame_path: Optional[Path] = None

    if not scene.get("_planner_missing_prompt"):
        video_id, video_path, frame_path = generate_scene_video(
            api_key=api_key,
            sora_prompt=scene["sora_prompt"],
            model=sora_model,
            size=video_size,
            seconds=DEFAULT_SECONDS,
            input_reference=parent_last_frame,
        )
        # copy into preset directory
        preset_dir = OUTPUT_ROOT / preset_slug
        video_relpath, poster_relpath = copy_assets(video_path, frame_path, preset_dir, node_slug)
        last_frame_path = frame_path
    else:
        print(f"  ⚠️ planner missing prompt at slug={node_slug}; skipping video generation")

    node = SceneNode(
        path=list(path),
        trigger_choice=trigger_choice,
        scenario_display=scene["scenario_display"],
        sora_prompt=scene["sora_prompt"],
        choices=scene["choices"],
        choices_short=scene["choices_short"],
        video_relpath=video_relpath,
        poster_relpath=poster_relpath,
    )

    if depth < MAX_DEPTH:
        updated_prompts = prior_prompts + [scene["sora_prompt"]]
        for idx, choice_text in enumerate(scene["choices"]):
            child_path = path + [idx]
            child = build_tree(
                api_key=api_key,
                planner_model=planner_model,
                sora_model=sora_model,
                video_size=video_size,
                preset_tracker=preset_tracker,
                overall_tracker=overall_tracker,
                preset_slug=preset_slug,
                base_prompt=base_prompt,
                path=child_path,
                prior_prompts=updated_prompts,
                trigger_choice=choice_text,
                parent_last_frame=Path(last_frame_path) if last_frame_path else None,
                depth=depth + 1,
            )
            node.children.append(child)

    detail = node_slug if trigger_choice is None else f"{node_slug} <- {trigger_choice[:24]}"
    preset_tracker.advance(detail)
    overall_detail = f"{preset_slug}:{detail}" if overall_tracker.use_tqdm else ""
    overall_tracker.advance(overall_detail)

    return node


def main() -> None:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        print("ERROR: OPENAI_API_KEY must be set in the environment.")
        sys.exit(1)

    planner_model = os.environ.get("PLANNER_MODEL", "gpt-5")
    sora_model = os.environ.get("SORA_MODEL", "sora-2")
    video_size = os.environ.get("VIDEO_SIZE", "1280x720")

    nodes_per_preset = sum(3 ** depth for depth in range(MAX_DEPTH + 1))
    overall_tracker = ProgressHandle(total=nodes_per_preset * len(PRESETS), desc="Overall", position=0)

    for idx, (preset_slug, base_prompt) in enumerate(PRESETS.items()):
        print(f"\n=== Generating preset '{preset_slug}' ===")
        preset_dir = OUTPUT_ROOT / preset_slug
        preset_dir.mkdir(parents=True, exist_ok=True)

        preset_tracker = ProgressHandle(total=nodes_per_preset, desc=preset_slug, position=idx + 1)

        tree = build_tree(
            api_key=api_key,
            planner_model=planner_model,
            sora_model=sora_model,
            video_size=video_size,
            preset_tracker=preset_tracker,
            overall_tracker=overall_tracker,
            preset_slug=preset_slug,
            base_prompt=base_prompt,
            path=[],
            prior_prompts=[],
            trigger_choice=None,
            parent_last_frame=None,
            depth=0,
        )

        preset_tracker.close()

        manifest = {
            "preset": preset_slug,
            "basePrompt": base_prompt,
            "tree": tree.to_dict(),
        }
        manifest_path = preset_dir / "manifest.json"
        with manifest_path.open("w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Manifest written to {manifest_path}")

    overall_tracker.close()
    print("\nAll presets generated. Assets live in", OUTPUT_ROOT)


if __name__ == "__main__":
    main()
