# Project Plan: Apocalyptic Cyberpunk NYC — Sora Experience

**Progress Tracker:** 🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢🟢 (100% complete)

Note: As execution progresses, I will tick the checkboxes below and update the emoji tracker (⚪→🟡→🟢) to reflect real status and % complete.

## Overview
We will build a fast‑paced, heart‑pounding, interactive video experience set in an apocalyptic, cyberpunk New York City that maps to real NYC geography in spirit and street‑level continuity. The system uses GPT‑5 to maintain a location‑aware “street tracker,” a compact NYC Catalog that defines each area’s visual/ecosystem identity, and tightly‑specified prompts for Sora 2 so every 8‑second shot is kinetic, consistent, and policy‑safe (faces obscured; continuous, intense audio).

## Goals
- Start in Midtown (Times Square) with immediate high‑speed traversal (no walking).
- Maintain near‑exact street continuity via a GPT‑5 “street tracker” (intersection/heading/blocks moved) without external map data (GPT-5, when prompted clearly, knows the geography of NYC well).
- Use a small, frozen NYC Catalog (few dozen areas) to stamp clear district identities (visuals, inhabitants/adversaries, hazards, audio motifs, traversal affordances).
- Enforce Sora prompt structure that guarantees speed, faces‑obscured, audio intensity, and NYC specificity every scene.
- Keep implementation lean: one Planner call per scene (state update + prompt + choices), minimal DB change, no frontend churn.

## Task List

- [x] 1) Confirm Plan And Lock Assumptions
  - Context: Confirm Midtown Times Square spawn; immediate hoverbike; GPT‑only generative mapping; catalog size (~30–40 areas); simple persistent inventory.
  - Goals: Freeze scope to avoid rework; agree on catalog breadth and storage location.
  - Deliverables: Short confirmation note in repo (comment in this doc) that plan is approved.
  - Dependencies: None.

- [x] 2) Generate NYC Catalog (Few Dozen Areas) With GPT-5
  - Context: One‑shot bootstrap; no runtime APIs; file remains static.
  - Goals: Produce `data/nyc_catalog.json` with borough/district/neighborhood entries containing visuals, ecosystem (adversaries/inhabitants/hazards), audio motifs, traversal affordances, animals/creatures, Sora tokens, and “street essence” notes.
  - Subtasks:
    - Draft strict JSON schema and system prompt for GPT‑5.
    - Run generation locally; manual quick review; freeze file.
  - Deliverables: `data/nyc_catalog.json` (checked in).
  - Dependencies: Step 1.

- [x] 3) Add Structured State Column
  - Context: Minimal DB change; store per‑scene machine state.
  - Goals: Add `scenes.state_json` (JSON/text) to persist the street tracker, inventory, audio motif, ecosystem context.
  - Subtasks: Migration command file; model mapping (non‑breaking).
  - Deliverables: Schema migration commands (separate file); updated model; no other API changes yet.
  - Dependencies: Step 1.

- [x] 4) Define Street Tracker State Schema
  - Context: Single source of truth for location/movement each shot.
  - Goals: Finalize `state_json` shape:
    - `location`: `borough`, `district`, `neighborhood`, `nearest_intersection`, `heading`, `blocks_moved` (1–3), `time_of_day`, `weather`.
    - `movement`: `tech_in_use` (e.g., `agile_hoverbike`/`jetpack`/`grapple`/`parkour`), `velocity_tier` = `fast`.
    - `inventory`: `current` array, `in_use`.
    - `ecosystem`: `adversaries`, `creatures`, `hazards`.
    - `audio`: `motif`, `intensity` (8–10).
    - `policy`: `faces_obscured` = true.
  - Deliverables: Inline developer doc (this file) + constants.
  - Dependencies: Step 3.

- [x] 5) Update Planner System To NYC Mode (Single Call Flow)
  - Context: Keep one GPT‑5 call per scene.
  - Goals: Prepend “NYC WORLD RULES” and require output keys: `scenario_display`, `sora_prompt`, `choices` (3), `choices_short` (3), and `state_update` (the next `state_json`).
  - Subtasks:
    - Inject prior `state_json` (or seed) + catalog slice for current area.
    - Enforce movement budget (1–3 blocks) and realistic heading changes.
  - Deliverables: Revised Planner system prompt; integration in backend planning function.
  - Dependencies: Steps 2–4.

- [x] 6) Enforce Sora Prompt Template (Speed/Policy/Audio)
  - Context: Guarantee kinetic pacing and policy safety.
  - Goals: Ensure `sora_prompt` follows strict structure:
    - Context (AI‑only): location/heading/blocks moved; faces obscured; movement tech; inventory in use; ecosystem; continuity; audio motif (continuous, intense).
    - Prompt: one continuous 8‑second shot with concrete camera actions and NYC visual anchors.
    - Action Beat: imperative, must fire within 8s.
  - Subtasks: Keep `ensure_action_beat()`; add velocity guard if Planner under‑specifies motion.
  - Deliverables: Updated planner rules + minor guard code.
  - Dependencies: Step 5.

- [x] 7) Spawn & Movement Setup
  - Context: Start in Times Square with fast traversal.
  - Goals: Seed root `state_json` with `nearest_intersection = "W 42 St & 7th Ave"`, `tech_in_use = agile_hoverbike`, `blocks_moved = 0`, `faces_obscured = true`, `audio.intensity = 9`.
  - Deliverables: Root scene seed logic.
  - Dependencies: Steps 3–6.

- [x] 8) Minimal Inventory Model
  - Context: Keep it simple and always useful.
  - Goals: Persistent starting items: `agile_hoverbike`, `energy_shield`, `grappling_hook`, `ar_visor` (extendable); Planner must pick one `in_use` each shot and reflect it in the Prompt.
  - Deliverables: Inventory seed in initial `state_json`; Planner hints to prefer variety.
  - Dependencies: Steps 3–5, 7.

- [x] 9) Persist And Expose State (Backend Only)
  - Context: Frontend remains unchanged.
  - Goals: Save Planner’s `state_update` to `scenes.state_json` after successful Sora rendering; extend API response with `stateJson` (non‑breaking alias).
  - Deliverables: Storage + response shaping (no UI work).
  - Dependencies: Steps 3–5.

- [x] 10) Continuity & Location Stability Pass
  - Context: Ensure travel feels plausible; no teleports.
  - Goals: Validate that heading/blocks moved are consistent, and area transitions happen at reasonable boundaries. Confirm last‑frame reference + street tracker yield smooth visuals.
  - Deliverables: Adjust small heuristics in prompts if needed.
  - Dependencies: Steps 5–9.

- [x] 11) Catalog And Prompt Tuning
  - Context: Ensure distinct identities per area (SoHo/Harlem/FiDi etc.).
  - Goals: Tighten visuals/ecosystem/audio motifs and traversal verbs so choices feel fresh and kinetic.
  - Deliverables: Minor catalog edits; prompt tweaks.
  - Dependencies: Steps 2, 5–10.

- [x] 12) Documentation & Handoff
  - Context: Keep maintenance trivial.
  - Goals: Add short notes on catalog ownership, how state evolves, and how to expand areas later if desired.
  - Deliverables: README updates or a small DEVNOTES section; this PLAN.md status updated.
  - Dependencies: Steps 2–11.

- [x] 13) Reinforce Photoreal Prompting
  - Context: Fidelity drifts toward toy-like renderings after turns.
  - Goals: Update planner instructions/default guidance to mandate photorealistic, cinematic output with HDR lighting and material fidelity; add validation ensuring prompts include these cues.
  - Deliverables: Adjusted planner system prompt/guidance; extended prompt validator.
  - Dependencies: Steps 5–6.

- [x] 14) Stabilize Camera & Avoid Collisions
  - Context: Rider collisions and camera oscillations reduce visual quality.
  - Goals: Instruct planner to maintain a steady trailing camera, plan obstacle-free paths, and ensure prompts explicitly forbid collisions; add guards verifying camera notes.
  - Deliverables: Updated planner rules and prompt checks to enforce steady third-person framing and collision avoidance.
  - Dependencies: Step 13.

- [x] 15) Skytram Exploration Mode
  - Context: Bikes/grapples cause instability; need a first-person rail experience with fork choices.
  - Goals: Reframe traversal as stabilized first-person mag-tram riding, simplify hazards, and focus choices on diverging rail paths that showcase districts.
  - Deliverables: Updated base prompt/guidance, planner instructions, state schema defaults, and choice language to support branchable rail routes with exploratory beats.
  - Dependencies: Steps 5–14.

- [ ] 15) Transition to First-Person Skytram Exploration
  - Context: High-speed bikes and grapples remain unstable; need traversal that plays to model strengths.
  - Goals: Shift traversal to a stabilized first-person mag-rail tram focused on exploration, low obstacles, and skyline vistas; update prompts, guidance, and state defaults.
  - Deliverables: Revised base prompt, planner system/guidance, state inventory, and documentation reflecting the skytram exploration format.
  - Dependencies: Steps 5–14.

## Dependencies Summary
- Step 2 depends on Step 1.
- Step 3 depends on Step 1.
- Step 4 depends on Step 3.
- Step 5 depends on Steps 2–4.
- Step 6 depends on Step 5.
- Step 7 depends on Steps 3–6.
- Step 8 depends on Steps 3–5 and 7.
- Step 9 depends on Steps 3–5.
- Steps 10–11 depend on prior integration (5–9).
- Step 12 depends on 2–11.

## Notes
- Tests are not required; we will validate through short manual playthroughs and prompt inspection.
- We are not using external map data; GPT‑5 is the generative mapping layer. The street tracker requires explicit intersection strings and movement budgets to maintain plausibility.
- Faces are always obscured; audio is continuous and intense in every scene.

## Update Protocol
- As we execute, I will:
  1) Mark checkboxes for completed steps.
  2) Change the emoji tracker at the top (⚪→🟡→🟢) to mirror aggregate progress.
  3) Append brief change notes under the relevant step.

### Step 1 Notes
- 2025-10-20: Plan approved; confirmed Times Square spawn, agile hoverbike start, GPT-only mapping, ~30–40 area catalog, 1–3 block traversal budget, simple persistent inventory.

### Step 2 Notes
- 2025-10-20: Created `data/nyc_catalog.json` (41 richly described areas across all boroughs) with visuals, ecosystems, audio motifs, movement calls, and adjacency lists.

### Step 3 Notes
- 2025-10-20: Added `state_json` column/ORM mapping/migration guard and exposed `stateJson` in API response for upcoming street tracker state.

### Step 4 Notes
- 2025-10-20: Created `nyc_state.py` with structured schema helpers, root spawn defaults (Times Square, hoverbike), normalization utilities, and merge logic.

### Step 5 Notes
- 2025-10-20: Rewired planner pipeline — new NYC-focused system prompt, added catalog + state context inputs, built structured planner caller, merged state updates, stored `state_json`, enforced prompt structure/state alignment, and updated base prompt/guidance.

### Step 6 Notes
- 2025-10-20: Added prompt gatekeeping — validating required Context/Prompt/Action Beat lines, checking state-aligned mentions (intersection, heading, gear, audio, blocks, velocity, faces), and failing generation if the structure slips.

### Step 7 Notes
- 2025-10-20: Seeded default Times Square state (`state_json`) upon root scene creation with hoverbike in use, blocks_moved=0, neon-rain ambience, and ensured planner merges next-state updates on top of it.

### Step 8 Notes
- 2025-10-20: Locked starter inventory (`agile_hoverbike`, `energy_shield`, `grappling_hook`, `ar_visor`), normalized merges to keep items persistent, defaulted `in_use`, and added prompt/state guards ensuring gear appears explicitly in every scene.

### Step 9 Notes
- 2025-10-20: Stored planner `state_update` results in `scenes.state_json`, seeded root rows, surfaced `stateJson` via API, and added structure validation before video generation.

### Step 10 Notes
- 2025-10-20: Enforced catalog adjacency for state transitions, clamped invalid hops, and validated prompts against location/gear/audio/velocity details so visuals stay geographically coherent.

### Step 11 Notes
- 2025-10-20: Polished catalog data (audio intensity floor = 8, richer guidance), tightened default prompt guidance to emphasise gear usage and district identity, ensuring Sora prompts stay vivid and consistent across boroughs.

### Step 12 Notes
- 2025-10-20: Refreshed README with the cyberpunk NYC narrative, catalog/state helper references, and updated frontend behaviour to describe the street tracker + inventory-driven prompts.

### Step 13 Notes
- 2025-10-20: Amplified photoreal requirements in planner rules, context block, and guidance; validator now blocks prompts lacking explicit photoreal language.

### Step 14 Notes
- 2025-10-20: Codified steady trailing camera + collision-free lanes, added guardrails to ensure prompts mention stability/avoidance, and updated guidance to keep the rider clear of obstacles.

### Step 15 Notes
- 2025-10-20: Converted traversal to first-person skytram exploration with branching rail choices, refreshed base prompt/guidance, planner rules, state defaults, and documentation to emphasize photoreal sightseeing over combat.
