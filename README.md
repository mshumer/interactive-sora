# Veo Control – Shared World Edition

[![Follow on X](https://img.shields.io/twitter/follow/mattshumer_?style=social)](https://x.com/mattshumer_)

[Be the first to know when I publish new AI builds + demos!](https://tally.so/r/w2M17p)

**One canonical choose-your-own adventure world, expanded by the community.**

The shared canon is now a portal-hopping multiverse: the Courier chases chronoglyph shards through remixed takes on famous game worlds (neon Vice City vibes, rune-soaked gothic battlefields, clockwork fantasy cities) to seal the Cataclysm Rift. Every 8-second beat delivers a high-energy action moment and sets up the next choice. When a branch already exists its video plays instantly; if not, explorers can contribute their own Gemini key (used for both Veo renders and planning) to mint the clip for everyone else.

---

## Quick Start

```bash
# In the project root
./start.sh
```

The script installs backend/frontend deps, spins up FastAPI on `http://localhost:8000`, and serves the React UI on `http://localhost:5173`.

The world boots with a placeholder base prompt. First-time explorers are asked for a Gemini API key (used for Veo renders and Gemini 2.5 Pro planning) the first time they select an ungenerated branch. Keys stay in `localStorage` and never leave the browser.

---

## Repo Layout

- `app.py` – FastAPI backend with shared-world persistence, R2 uploads, and the Veo/Planner orchestration.
- `frontend/` – React client with the immersive player UI.
- `start.sh` – Convenience script that bootstraps everything.
- `generate_preset_content.py` – Optional tool for pre-rendering demo trees per preset.

---

## Environment

| Variable | Default | Description |
| --- | --- | --- |
| `WORLD_ID` | `default` | Namespace for this shared world. |
| `WORLD_BASE_PROMPT` | multiverse chase narrative | Cinematic seed prompt used for the very first scene. Override to reskin the world. |
| `PLANNER_MODEL` | `gemini-2.5-pro` | Planner model passed to Gemini text generation. |
| `VEO_MODEL` | `veo-3.1-generate-preview` | Model name forwarded to the Gemini `/models/*:predictLongRunning` endpoint. |
| `GEMINI_API_BASE` | `https://generativelanguage.googleapis.com/v1beta` | Override to point at a different Gemini endpoint if needed. |
| `VIDEO_SIZE` | `1980x1080` | Render resolution for all clips (also used to infer Veo aspect ratio). |
| `DATABASE_URL` | `sqlite:///./veo_world.db` | SQLAlchemy connection string. Supply your Railway/Supabase URL in production. |
| `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` | — | Cloudflare R2 credentials. If unset, assets fall back to local disk (`storage/`). |
| `R2_PUBLIC_BASE_URL` | — | Optional CDN base (e.g. `https://media.example.com`). |
| `R2_SIGNED_URLS` | `1` | Emit signed download URLs for R2 assets. Set to `0` to return raw public links. |
| `R2_SIGNED_URL_TTL` | `3600` | Lifetime (seconds) for each signed asset URL. |
| `SCENE_TIMEOUT_SECONDS` | `900` | Cancel and recycle claims that sit in `queued` longer than 15 minutes. |
| `WATCHDOG_INTERVAL_SECONDS` | `60` | How often the timeout watchdog scans for stale jobs. |
| `CONTRIBUTOR_SALT` | `veo-shared-world` | Salt used when hashing contributor metadata. |
| `WORLD_PROMPT_GUIDANCE` | — | Optional extra flavor/examples injected into every planner call. |
| `STATE_SUMMARY_MODEL` | `gemini-2.5-pro` | Model used to summarise each scene’s evolving world state (set to blank to disable). |

For local hacking you can skip the R2 vars—videos will be copied into `storage/` automatically.

Run the backend once to create the tables:

```bash
uvicorn app:app --reload
```

On first load the root scene is `pending`. Launch the UI at `http://localhost:5173`, click the highlighted branch, and drop in your key to mint the opening clip.

---

## Admin Dashboard

Operate the shared world safely from the built-in admin panel:

1. Set or update the admin password (stored as an Argon2 hash):

   ```bash
   source .venv/bin/activate  # if you use the bundled virtualenv
   python tools/set_admin_password.py
   ```

2. Visit `http://localhost:8000/admin` and sign in with that password.

The dashboard lets you:

- Filter by path prefix to inspect generated scenes.
- Preview videos inline (including continuity clips).
- View child status, storage footprint, and quick metadata.
- Reset any branch (or the entire world) in one click—this cancels in-flight generations, deletes associated media (local storage or R2), removes metrics, and prunes the scene tree so it can be regenerated cleanly.

**Canvas explorer.** The default view is a pan/zoom canvas showing the branch tree with live thumbnails. Click a node to load its clip in the inspector, double-click to collapse/expand a subtree, drag to pan, and use the toolbar or mouse wheel to zoom. The inspector’s controls let you play clips, open the larger preview modal, or recycle the selected branch instantly. A fallback List view remains available from the Canvas/List toggle.

Sessions are stored in a signed, HttpOnly cookie (auto-expiring after 14 days or immediately on password rotation). Login attempts are rate limited, and the admin page is locked behind SameSite=Strict cookies and hardened response headers.

---

## API Surface

| Endpoint | Purpose |
| --- | --- |
| `GET /worlds/{worldId}` | World metadata (base prompt, fixed models). |
| `GET /worlds/{worldId}/scenes?path=...` | Fetch a scene and child status. Creates placeholder rows on demand. |
| `POST /worlds/{worldId}/scenes` | Claim or generate a branch using the caller’s API key. Returns `ready`, `queued`, or `failed`. |
| `POST /worlds/{worldId}/scenes/{path}/retry` | Convenience alias for retrying failed branches. |
| `GET /worlds/{worldId}/metrics` | Aggregate telemetry: branch counts, queued/failed totals, storage bytes, and success rate. |

All writes are serialized per `worldId + path`, so only the first explorer to claim a branch spends credits. Everyone else waits for the cached asset.

---

## Frontend Behaviour

- Config screen removed—players jump straight into the world.
- Choices with cached clips are highlighted, signalling instant playback.
- Selecting an unexplored branch prompts for a key (with cancel option to pick another path).
- Keys persist in `localStorage` under `veo_shared_world_video_api_key` (Veo) and `veo_shared_world_planner_api_key` (planner).
- Storyboard/timeline reflects the canonical branch status in real time.
- Active generations surface live progress so explorers can see how close a branch is to finishing.
- Behind the scenes, each scene stores a state summary (generated with Gemini 2.5 Pro) so future branches carry forward the evolving world context.
- Planner prompts now emphasise a full 8-second action beat (setup → escalation → outcome) so every clip lands a decisive moment before offering new choices.
- The Nexus Gate opening beat presents three mysterious portals; hit **Restart** anytime to return there and choose a different world with instant playback of already-generated branches.

---

## Veo 3.1 Continuity

- Each freshly generated beat is stitched onto the entire path-to-date using Veo 3.1 scene extension, so every new prompt sees the full video context instead of a single poster frame.
- The backend stores both the delivered 8-second clip and the aggregated continuity file for future generations.
- Veo's preview tier only accepts ~20 seconds of Veo-processed footage per extension; if you exceed it the service will ask you to restart earlier in the branch.

---

## Telemetry & Operations

- `SceneMetric` rows capture storage usage for each completed render.
- The timeout watchdog resets branches stuck in `queued` for >10 minutes and cancels the underlying generation thread.
- `GET /worlds/{id}/metrics` powers lightweight dashboards for branch count, queued backlog, and success rate.
- Logs surface generation start/finish/failure events, making Railway alerts straightforward.

---

Build shared, cinematic adventures—once a branch exists, the whole world inherits it.
