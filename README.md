# Interactive Sora

[![Follow on X](https://img.shields.io/twitter/follow/mattshumer_?style=social)](https://x.com/mattshumer_)

[Be the first to know when I publish new AI builds + demos!](https://tally.so/r/w2M17p)

**Immersive choose-your-own adventure experiences powered by OpenAI Sora 2.**

Interactive Sora is an exploration of how real-time video generation can power interactive worlds. It's built as a choose-your-own-adventure system... every choice spins up a fresh Sora 2 sequence—turning player choices into responsive, explorable environments that morph with the player’s intent.

---

## Quick Start

```bash
# In the project root
./start.sh
```

The script will:

1. Create (or reuse) `.venv` and install backend dependencies.
2. Install the frontend packages with `npm install`/`npm ci`.
3. Launch FastAPI on `http://localhost:8000` and the Vite dev server on `http://localhost:5173`.

Open your browser at `http://localhost:5173` and drop in an OpenAI API key to play.

> **Tip:** The key is stored locally via `localStorage` so you don’t need to re-enter it while iterating.

---

## Repo Layout

- `app.py` – FastAPI backend that orchestrates planner calls, Sora jobs, and media delivery.
- `frontend/` – React client with the immersive player UI.
- `start.sh` – Convenience script that bootstraps everything.
- `generate_preset_content.py` – Optional tool for pre-rendering demo trees per preset.

---

Build rich, cinematic branching adventures with Interactive Sora.
