import { useEffect, useMemo, useState } from "react";
import "../styles/config.css";

const PRESETS = [
  {
    label: "Neon Midnight Walk",
    basePrompt:
      "Photorealistic rainy midnight street in a near-future Tokyo district. Camera glides beside a lone figure walking past noodle stalls, holographic billboards, flickering neon reflections in puddles, steam rising from manholes, and curious onlookers in reflective raincoats. Emphasize cinematic lighting, wet asphalt textures, and the sense that anything could emerge from the crowd.",
  },
  {
    label: "Verdant Quest",
    basePrompt:
      "Photorealistic enchanted forest adventure at golden hour. Follow an explorer in weathered travel gear trekking through towering moss-covered trees, shafts of light cutting through mist, ancient stone ruins hidden under vines, and distant drumbeats hinting at hidden civilizations. The air feels alive with curiosity and imminent discovery.",
  },
  {
    label: "House of Echoes",
    basePrompt:
      "Photorealistic claustrophobic horror inside a decaying Victorian mansion. The protagonist moves room to room; each doorway reveals a new terror: portraits whose eyes bleed shadows, a nursery of toys that whisper, a dining hall table set for spirits. Lighting is minimal, with handheld flashlight beams and erratic power surges casting unsettling moving silhouettes.",
  },
];

const PLANNER_KEY_STORAGE = "veo_shared_world_planner_api_key";
const VIDEO_KEY_STORAGE = "veo_shared_world_video_api_key";
const GEMINI_KEY_STORAGE = "veo_shared_world_gemini_api_key";

const ConfigScreen = ({ onSubmit, isSubmitting, error, apiBaseUrl }) => {
  const [form, setForm] = useState({
    apiKey: "",
    plannerModel: "gemini-2.5-pro",
    veoModel: "veo-3.1-generate-preview",
    videoSize: "1280x720",
    basePrompt:
      "A cozy fantasy village at dusk, with glowing lanterns, narrow cobblestone streets, and a mysterious whisper about an ancient forest relic.",
  });

  useEffect(() => {
    if (typeof window === "undefined") return;
    const storedGemini =
      window.localStorage.getItem(GEMINI_KEY_STORAGE) ||
      window.localStorage.getItem("sora_cyoa_api_key") ||
      "";
    const storedPlanner = window.localStorage.getItem(PLANNER_KEY_STORAGE) || "";
    const storedVideo = window.localStorage.getItem(VIDEO_KEY_STORAGE) || "";
    const effectiveKey = storedGemini || storedVideo || storedPlanner;
    if (effectiveKey) {
      setForm((prev) => ({
        ...prev,
        apiKey: effectiveKey,
      }));
    }
  }, []);

  const maskedApiKey = useMemo(() => {
    if (!form.apiKey) return "";
    return form.apiKey.replace(/.(?=.{4})/g, "·");
  }, [form.apiKey]);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
    if (typeof window === "undefined") return;
    if (name === "apiKey") {
      window.localStorage.setItem(GEMINI_KEY_STORAGE, value);
      window.localStorage.setItem(PLANNER_KEY_STORAGE, value);
      window.localStorage.setItem(VIDEO_KEY_STORAGE, value);
    }
  };

  const handleSubmit = (event) => {
    event.preventDefault();
    onSubmit(form);
  };

  return (
    <div className="config-shell">
      <div className="config-backdrop" />
      <div className="config-inner">
        <section className="config-hero">
          <p className="config-tag">Veo Control</p>
          <h1>
            Dial in your <span>story engine</span>
          </h1>
          <p className="config-subtitle">
            First, prime the director. Tune the models, set the vibe, and let the cinematic universe know who holds the reins.
          </p>

          <div className="config-presets">
            {PRESETS.map((preset) => (
              <button
                key={preset.label}
                type="button"
                className="config-preset"
                disabled={isSubmitting}
                onClick={() => setForm((prev) => ({ ...prev, basePrompt: preset.basePrompt }))}
              >
                <span>{preset.label}</span>
                <small>Inject prompt</small>
              </button>
            ))}
          </div>

          <div className="config-meta">
            <span>Backend:</span>
            <strong>{apiBaseUrl || "http://localhost:8000"}</strong>
          </div>
        </section>

        <section className="config-panel">
          <div className="panel-glass">
            <header>
              <h2>Launch Configuration</h2>
            </header>

            <form onSubmit={handleSubmit}>
              <label className="field">
                <span>Gemini API key</span>
                <div className="masked">
                  <input
                    name="apiKey"
                    type="password"
                    placeholder="AIza..."
                    value={form.apiKey}
                    onChange={handleChange}
                    required
                    autoComplete="off"
                    spellCheck={false}
                  />
                  <span className="mask-preview">{maskedApiKey}</span>
                </div>
              </label>

              <div className="field-grid">
                <label className="field">
                  <span>Planner model</span>
                  <input
                    name="plannerModel"
                    value={form.plannerModel}
                    onChange={handleChange}
                    placeholder="gemini-2.5-pro"
                    required
                  />
                </label>
                <label className="field">
                  <span>Veo model</span>
                  <select name="veoModel" value={form.veoModel} onChange={handleChange}>
                    <option value="veo-3.1-generate-preview">veo-3.1-generate-preview</option>
                    <option value="veo-3.1-fast-preview">veo-3.1-fast-preview</option>
                  </select>
                </label>
                <label className="field">
                  <span>Video size</span>
                  <select name="videoSize" value={form.videoSize} onChange={handleChange}>
                    <option value="1280x720">1280 × 720</option>
                    <option value="1920x1080">1920 × 1080</option>
                    <option value="720x1280">720 × 1280</option>
                  </select>
                </label>
              </div>

              <label className="field">
                <span>World / tone prompt</span>
                <textarea
                  name="basePrompt"
                  value={form.basePrompt}
                  onChange={handleChange}
                  rows={6}
                  required
                />
              </label>

              {error && <div className="error-banner">{error}</div>}

              <button type="submit" className="launch" disabled={isSubmitting}>
                {isSubmitting ? "Generating opening scene…" : "Launch cinematic adventure"}
              </button>
            </form>
          </div>
        </section>
      </div>
    </div>
  );
};

export default ConfigScreen;
