# Sora Control – Shared World Edition

[![Follow on X](https://img.shields.io/twitter/follow/mattshumer_?style=social)](https://x.com/mattshumer_)

[Be the first to know when I publish new AI builds + demos!](https://tally.so/r/w2M17p)

**One canonical choose-your-own adventure world, expanded by the community.**

The shared canon is now a portal-hopping multiverse: the Courier chases chronoglyph shards through remixed takes on famous game worlds (neon Vice City vibes, rune-soaked gothic battlefields, clockwork fantasy cities) to seal the Cataclysm Rift. Every 8-second beat delivers a high-energy action moment and sets up the next choice. When a branch already exists its video plays instantly; if not, explorers can contribute their own OpenAI API key to mint the clip for everyone else.

---

## Quick Start

```bash
# In the project root
./start.sh
```

The script installs backend/frontend deps, spins up FastAPI on `http://localhost:8000`, and serves the React UI on `http://localhost:5173`.

The world boots with a placeholder base prompt. First-time explorers will be asked for an OpenAI key only when they select an ungenerated branch. Keys stay in `localStorage` and never leave the browser.

---

## Repo Layout

- `app.py` – FastAPI backend with shared-world persistence, R2 uploads, and the Sora/Planner orchestration.
- `frontend/` – React client with the immersive player UI.
- `start.sh` – Convenience script that bootstraps everything.
- `generate_preset_content.py` – Optional tool for pre-rendering demo trees per preset.

---

## Environment

| Variable | Default | Description |
| --- | --- | --- |
| `WORLD_ID` | `default` | Namespace for this shared world. |
| `WORLD_BASE_PROMPT` | multiverse chase narrative | Cinematic seed prompt used for the very first scene. Override to reskin the world. |
| `PLANNER_MODEL` | `gpt-5` | Planner model passed to the Responses API. |
| `SORA_MODEL` | `sora-2` | Model name forwarded to the Sora `/videos` endpoint. |
| `VIDEO_SIZE` | `1280x720` | Render resolution for all clips. |
| `DATABASE_URL` | `sqlite:///./sora_world.db` | SQLAlchemy connection string. Supply your Railway/Supabase URL in production. |
| `R2_ACCOUNT_ID`, `R2_ACCESS_KEY_ID`, `R2_SECRET_ACCESS_KEY`, `R2_BUCKET_NAME` | — | Cloudflare R2 credentials. If unset, assets fall back to local disk (`storage/`). |
| `R2_PUBLIC_BASE_URL` | — | Optional CDN base (e.g. `https://media.example.com`). |
| `R2_SIGNED_URLS` | `1` | Emit signed download URLs for R2 assets. Set to `0` to return raw public links. |
| `R2_SIGNED_URL_TTL` | `3600` | Lifetime (seconds) for each signed asset URL. |
| `SCENE_TIMEOUT_SECONDS` | `900` | Cancel and recycle claims that sit in `queued` longer than 15 minutes. |
| `WATCHDOG_INTERVAL_SECONDS` | `60` | How often the timeout watchdog scans for stale jobs. |
| `CONTRIBUTOR_SALT` | `sora-shared-world` | Salt used when hashing contributor metadata. |
| `WORLD_PROMPT_GUIDANCE` | — | Optional extra flavor/examples injected into every planner call. |
| `STATE_SUMMARY_MODEL` | `gpt-5-mini` | Model used to summarise each scene’s evolving world state (set to blank to disable). |

For local hacking you can skip the R2 vars—videos will be copied into `storage/` automatically.

Run the backend once to create the tables:

```bash
uvicorn app:app --reload
```

On first load the root scene is `pending`. Launch the UI at `http://localhost:5173`, click the highlighted branch, and drop in your key to mint the opening clip.

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
- Keys persist in `localStorage` under `sora_shared_world_api_key`.
- Storyboard/timeline reflects the canonical branch status in real time.
- Active generations surface live progress so explorers can see how close a branch is to finishing.
- Behind the scenes, each scene stores a state summary (generated with `gpt-5-mini`) so future branches carry forward the evolving world context.
- Planner prompts now emphasise a full 8-second action beat (setup → escalation → outcome) so every clip lands a decisive moment before offering new choices.
- The Nexus Gate opening beat presents three mysterious portals; hit **Restart** anytime to return there and choose a different world with instant playback of already-generated branches.

---

## Telemetry & Operations

- `SceneMetric` rows capture storage usage for each completed render.
- The timeout watchdog resets branches stuck in `queued` for >10 minutes and cancels the underlying generation thread.
- `GET /worlds/{id}/metrics` powers lightweight dashboards for branch count, queued backlog, and success rate.
- Logs surface generation start/finish/failure events, making Railway alerts straightforward.

---

Build shared, cinematic adventures—once a branch exists, the whole world inherits it.
