import { useEffect, useMemo, useRef, useState } from "react";
import Timeline from "./Timeline.jsx";
import "../styles/experience.css";

const ExperienceScreen = ({
  context,
  onMakeChoice,
  isGenerating,
  error,
  apiBaseUrl,
  onRestart,
  onPromptForKey,
  showKeyModal,
  onKeySubmit,
  onKeyCancel,
}) => {
  const { story, worldInfo, apiKey } = context;
  const [activeIndex, setActiveIndex] = useState(Math.max(story.length - 1, 0));
  const [hasVideoEnded, setHasVideoEnded] = useState(false);
  const [showStoryboard, setShowStoryboard] = useState(false);
  const [isVideoLoading, setIsVideoLoading] = useState(false);
  const [isMuted] = useState(true);
  const videoRef = useRef(null);

  useEffect(() => {
    if (!story.length) return;
    const latest = story.length - 1;
    setActiveIndex(latest);
    setShowStoryboard(false);
    setHasVideoEnded(false);
  }, [story]);

  const activeScene = story[activeIndex] || null;

  useEffect(() => {
    if (!activeScene) return;
    setIsVideoLoading(Boolean(activeScene.videoUrl));
    setHasVideoEnded(!activeScene.videoUrl);
    if (activeScene.videoUrl && videoRef.current) {
      const playPromise = videoRef.current.play();
      if (playPromise?.catch) {
        playPromise.catch(() => setHasVideoEnded(true));
      }
    }
  }, [activeScene]);

  const videoSrc = useMemo(() => {
    if (!activeScene?.videoUrl) return null;
    if (/^https?:/i.test(activeScene.videoUrl)) return activeScene.videoUrl;
    return `${apiBaseUrl}${activeScene.videoUrl}`;
  }, [activeScene?.videoUrl, apiBaseUrl]);

  const posterSrc = useMemo(() => {
    if (!activeScene?.posterUrl) return null;
    if (/^https?:/i.test(activeScene.posterUrl)) return activeScene.posterUrl;
    return `${apiBaseUrl}${activeScene.posterUrl}`;
  }, [activeScene?.posterUrl, apiBaseUrl]);

  const choices = activeScene?.choices || [];
  const choicesStatus = activeScene?.choicesStatus || [];
  const isQueued = activeScene?.status === "queued";
  const progress = typeof activeScene?.progress === "number" ? activeScene.progress : null;

  const canChoose = Boolean(activeScene && choices.length && !isGenerating);

  const handleChoice = (index) => {
    if (!canChoose) return;
    onMakeChoice(index);
  };

  const handleVideoEnd = () => {
    setHasVideoEnded(true);
  };

  const handleReplay = () => {
    if (!videoRef.current) return;
    setHasVideoEnded(false);
    videoRef.current.currentTime = 0;
    videoRef.current.play().catch(() => setHasVideoEnded(true));
  };

  const handleLoadedData = () => {
    setIsVideoLoading(false);
  };

  const showLoader = isGenerating || isQueued || (Boolean(videoSrc) && isVideoLoading);
  const allowReplay = Boolean(videoSrc && hasVideoEnded);

  if (!activeScene) {
    return (
      <div className="immersive-shell">
        <div className="immersive-stage empty">
          <p>Loading shared world…</p>
        </div>
      </div>
    );
  }

  const renderStatusLabel = () => {
    switch (activeScene.status) {
      case "ready":
        return "Scene ready";
      case "queued":
        return typeof progress === "number" ? `Generating… ${progress}%` : "Another explorer is generating this scene";
      case "failed":
        return activeScene.failureDetail || "Generation failed";
      default:
        return "Scene not yet generated";
    }
  };

  return (
    <div className="immersive-shell">
      <div className="immersive-stage">
        {videoSrc ? (
          <video
            key={videoSrc}
            ref={videoRef}
            className="immersive-video"
            src={videoSrc}
            poster={posterSrc || undefined}
            autoPlay
            playsInline
            controls={false}
            muted={isMuted}
            onEnded={handleVideoEnd}
            onPlay={() => setHasVideoEnded(false)}
            onLoadedData={handleLoadedData}
          />
        ) : (
          <div className="video-placeholder">
            <p>{renderStatusLabel()}</p>
            {activeScene.status !== "ready" && (
              <button
                type="button"
                className="cta"
                onClick={() => onPromptForKey(activeScene.path || "")}
              >
                Generate this scene
              </button>
            )}
          </div>
        )}
        <div className="video-mask" />

        <div className="scene-pill">Scene {(activeIndex + 1).toString().padStart(2, "0")}</div>

        <div className="floating-controls">
          <button type="button" className="control-button" onClick={() => setShowStoryboard(true)}>
            Storyboard
          </button>
          <button type="button" className="control-button" onClick={() => onPromptForKey(undefined)}>
            {apiKey ? "Update API Key" : "Add API Key"}
          </button>
          <button type="button" className="control-button" onClick={onRestart}>
            Restart
          </button>
        </div>

        {allowReplay && videoSrc && (
          <button type="button" className="replay-float" onClick={handleReplay}>
            Replay Scene
          </button>
        )}

        <div className="choice-drawer visible">
          <div className="drawer-inner">
            <div className="drawer-header">
              <span className="status awaiting">Choose the next branch</span>
            </div>
            <div className="choice-grid">
              {choices.length ? (
                choices.map((choice, index) => {
                  const status = choicesStatus[index] || "pending";
                  const highlight = status === "ready";
                  const showStatusBadge = status === "queued" || status === "failed";
                  return (
                    <button
                      key={`${choice}-${index}`}
                      type="button"
                      className={`choice-pod ${highlight ? "ready" : ""} ${status}`}
                      disabled={isGenerating}
                      onClick={() => handleChoice(index)}
                    >
                      <span className="choice-number">{index + 1}</span>
                      <span className="choice-copy">{choice}</span>
                      {showStatusBadge && <ChoiceStatusBadge status={status} />}
                    </button>
                  );
                })
              ) : (
                <p className="choice-placeholder">No choices yet — generate this scene to continue.</p>
              )}
            </div>
          </div>
        </div>

        <Timeline
          story={story}
          activeIndex={activeIndex}
          onSelect={(index) => setActiveIndex(index)}
          apiBaseUrl={apiBaseUrl}
          isOpen={showStoryboard}
          onClose={() => setShowStoryboard(false)}
        />

        {showLoader && (
          <div className="loading-veil">
            <div className="loading-core" />
            <p>
              {isQueued
                ? typeof progress === "number"
                  ? `Generating… ${progress}%`
                  : "Generating scene…"
                : isGenerating
                ? "Forging the next sequence…"
                : "Preparing scene…"}
            </p>
            {typeof progress === "number" && <ProgressBar progress={progress} />}
          </div>
        )}

        {error && <div className="error-pill">{error}</div>}
      </div>

      {showKeyModal && (
        <ApiKeyModal
          initialValue={apiKey}
          onSubmit={onKeySubmit}
          onCancel={onKeyCancel}
        />
      )}
    </div>
  );
};

const ChoiceStatusBadge = ({ status }) => {
  if (status === "queued") {
    return <span className="choice-status queued">Generating</span>;
  }
  if (status === "failed") {
    return <span className="choice-status failed">Needs retry</span>;
  }
  return null;
};

const ProgressBar = ({ progress }) => {
  const value = Math.min(100, Math.max(0, progress));
  return (
    <div className="progress-bar">
      <div className="progress-bar-track">
        <div className="progress-bar-fill" style={{ width: `${value}%` }} />
      </div>
      <span className="progress-label">{value}%</span>
    </div>
  );
};

const ApiKeyModal = ({ initialValue, onSubmit, onCancel }) => {
  const [value, setValue] = useState(initialValue || "");
  const [revealed, setRevealed] = useState(false);

  const handleSubmit = (event) => {
    event.preventDefault();
    if (!value.trim()) return;
    onSubmit(value.trim());
  };

  return (
    <div className="key-modal-backdrop">
      <div className="key-modal">
        <h3>Provide your OpenAI API key</h3>
        <p>
          We only use this key if the next branch has never been generated. It never leaves your
          browser storage.
        </p>
        <form onSubmit={handleSubmit}>
          <label>
            <span>API Key</span>
            <div className="key-input-row">
              <input
                type={revealed ? "text" : "password"}
                value={value}
                onChange={(event) => setValue(event.target.value)}
                placeholder="sk-..."
                autoFocus
              />
              <button
                type="button"
                className="reveal-toggle"
                onClick={() => setRevealed((prev) => !prev)}
              >
                {revealed ? "Hide" : "Show"}
              </button>
            </div>
          </label>
          <div className="key-actions">
            <button type="button" className="ghost" onClick={onCancel}>
              Cancel
            </button>
            <button type="submit" className="primary" disabled={!value.trim()}>
              Save Key
            </button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ExperienceScreen;
