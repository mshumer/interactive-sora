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

const ConfigScreen = ({ onSubmit, isSubmitting, error, apiBaseUrl }) => {
  const [form, setForm] = useState({
    apiKey: "",
    plannerModel: "gpt-5",
    soraModel: "sora-2",
    videoSize: "1280x720",
    basePrompt:
      "A cozy fantasy village at dusk, with glowing lanterns, narrow cobblestone streets, and a mysterious whisper about an ancient forest relic.",
  });

  useEffect(() => {
    if (typeof window === "undefined") return;
    const storedKey = window.localStorage.getItem("sora_cyoa_api_key");
    if (storedKey) {
      setForm((prev) => ({ ...prev, apiKey: storedKey }));
    }
  }, []);

  const maskedKey = useMemo(() => {
    if (!form.apiKey) return "";
    return form.apiKey.replace(/.(?=.{4})/g, "·");
  }, [form.apiKey]);

  const handleChange = (event) => {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
    if (name === "apiKey" && typeof window !== "undefined") {
      window.localStorage.setItem("sora_cyoa_api_key", value);
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
          <p className="config-tag">Sora Control</p>
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
                <span>OpenAI API key</span>
                <div className="masked">
                  <input
                    name="apiKey"
                    type="password"
                    placeholder="sk-..."
                    value={form.apiKey}
                    onChange={handleChange}
                    required
                    autoComplete="off"
                    spellCheck={false}
                  />
                  <span className="mask-preview">{maskedKey}</span>
                </div>
              </label>

              <div className="field-grid">
                <label className="field">
                  <span>Planner model</span>
                  <input
                    name="plannerModel"
                    value={form.plannerModel}
                    onChange={handleChange}
                    placeholder="gpt-5"
                    required
                  />
                </label>
                <label className="field">
                  <span>Sora model</span>
                  <select name="soraModel" value={form.soraModel} onChange={handleChange}>
                    <option value="sora-2">sora-2</option>
                    <option value="sora-2-pro">sora-2-pro</option>
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
