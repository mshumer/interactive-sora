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
  const { story, worldInfo, apiKey, prefetchedVideos } = context;
  const [activeIndex, setActiveIndex] = useState(Math.max(story.length - 1, 0));
  const [hasVideoEnded, setHasVideoEnded] = useState(false);
  const [showStoryboard, setShowStoryboard] = useState(false);
  const [isVideoLoading, setIsVideoLoading] = useState(false);
  const [hasEnteredExperience, setHasEnteredExperience] = useState(false);
  const [choicesRevealActive, setChoicesRevealActive] = useState(false);
  const [showPoster, setShowPoster] = useState(true);
  const videoRef = useRef(null);
  const lastSceneIdRef = useRef(null);

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

    const hasVideo = Boolean(activeScene.videoUrl);
    const sceneIdentity = activeScene.path || activeScene.videoUrl || "__scene__";
    const isNewScene = lastSceneIdRef.current !== sceneIdentity;
    lastSceneIdRef.current = sceneIdentity;

    setIsVideoLoading(hasVideo);
    setHasVideoEnded(!hasVideo);
    setChoicesRevealActive(!hasVideo);
    setShowPoster(Boolean(activeScene.posterUrl));

    if (!hasVideo) return;

    const videoElement = videoRef.current;
    if (!videoElement) return;

    if (videoElement.readyState >= 2) {
      setIsVideoLoading(false);
    }

    if (isNewScene) {
      videoElement.currentTime = 0;
    }

    const preloadedPool = prefetchedVideos?.current;
    const preloadedVideo = preloadedPool?.get(activeScene.path);
    if (preloadedVideo && preloadedVideo.readyState >= 2) {
      const source = preloadedVideo.currentSrc || preloadedVideo.src;
      if (source && videoElement.src !== source) {
        videoElement.pause();
        videoElement.removeAttribute("src");
        videoElement.load();
        videoElement.src = source;
      }
      try {
        videoElement.currentTime = 0;
      } catch (error) {
        // ignore seek issues on some browsers with streaming sources.
      }
      preloadedPool?.delete(activeScene.path);
    }

    if (!hasEnteredExperience) {
      return;
    }

    let cancelled = false;

    const tryPlay = async () => {
      try {
        await videoElement.play();
      } catch (error) {
        if (cancelled) return;
        // Leave the scene ready for manual playback if something unexpected happens.
      }
    };

    tryPlay();

    return () => {
      cancelled = true;
    };
  }, [activeScene, hasEnteredExperience, prefetchedVideos]);

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
    setChoicesRevealActive(true);
  };

  const handleVideoPlay = () => {
    setHasVideoEnded(false);
    setIsVideoLoading(false);
    setShowPoster(false);
  };

  const handleReplay = () => {
    if (!videoRef.current) return;
    setHasVideoEnded(false);
    setChoicesRevealActive(false);
    setShowPoster(Boolean(activeScene?.posterUrl));
    videoRef.current.currentTime = 0;
    videoRef.current.play().catch((error) => {
      setHasVideoEnded(true);
    });
  };

  const handleLoadedData = () => {
    setIsVideoLoading(false);
    const videoElement = videoRef.current;
    if (!videoElement) return;
    const duration = Number.isFinite(videoElement.duration) ? videoElement.duration : null;
    if (duration !== null && duration <= 1) {
      setChoicesRevealActive(true);
    }
  };

  const handleTimelineSelect = (index) => {
    setActiveIndex(index);
  };

  const handleExperienceStart = () => {
    const videoElement = videoRef.current;
    const attemptPlayback = () => {
      if (!videoRef.current) return;
      const playPromise = videoRef.current.play();
      if (playPromise?.catch) {
        playPromise.catch(() => {
          setHasEnteredExperience(false);
        });
      }
    };

    setHasEnteredExperience(true);
    setHasVideoEnded(false);
    setChoicesRevealActive(false);
    setShowPoster(Boolean(activeScene?.posterUrl));
    setIsVideoLoading((prev) => {
      if (!videoElement) return prev;
      return videoElement.readyState < 2;
    });
    if (!videoElement) return;
    videoElement.currentTime = 0;
    if (videoElement.readyState >= 2) {
      attemptPlayback();
      return;
    }

    const handleCanPlay = () => {
      videoElement.removeEventListener("canplay", handleCanPlay);
      attemptPlayback();
    };

    videoElement.addEventListener("canplay", handleCanPlay);
  };

  const handleTimeUpdate = () => {
    if (choicesRevealActive) return;
    const videoElement = videoRef.current;
    if (!videoElement) return;
    const duration = Number.isFinite(videoElement.duration) ? videoElement.duration : null;
    if (!duration || duration <= 0) return;
    const remaining = duration - videoElement.currentTime;
    if (remaining <= 1) {
      setChoicesRevealActive(true);
    }
  };

  const showLoader = isGenerating || isQueued || (Boolean(videoSrc) && isVideoLoading);
  const allowReplay = Boolean(videoSrc && hasVideoEnded);
  const shouldShowChoiceDrawer = hasEnteredExperience && choicesRevealActive;
  const worldTitle = worldInfo?.title || worldInfo?.name || "Choose Your Odyssey";
  const worldDescription =
    worldInfo?.tagline ||
    worldInfo?.description ||
    "Step into a cinematic universe that rewrites itself each time you make a choice.";

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
            ref={videoRef}
            className="immersive-video"
            src={videoSrc}
            poster={showPoster ? posterSrc || undefined : undefined}
            autoPlay
            playsInline
            preload="auto"
            controls={false}
            onEnded={handleVideoEnd}
            onPlay={handleVideoPlay}
            onLoadedData={handleLoadedData}
            onTimeUpdate={handleTimeUpdate}
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
          <button
            type="button"
            className="control-button"
            onClick={() => {
              setHasEnteredExperience(false);
              setChoicesRevealActive(false);
              setHasVideoEnded(false);
              onRestart();
            }}
          >
            Restart
          </button>
        </div>

        {allowReplay && videoSrc && (
          <button type="button" className="replay-float" onClick={handleReplay}>
            Replay Scene
          </button>
        )}

        {!hasEnteredExperience && (
          <div className="start-screen">
            <div className="start-screen__backdrop" />
            <div className="start-screen__content">
              <span className="start-screen__eyebrow">Shared World Prelude</span>
              <h1 className="start-screen__headline">{worldTitle}</h1>
              <p className="start-screen__body">{worldDescription}</p>
              <div className="start-screen__chips">
                <span className="start-chip">Dynamic Sora Scenes</span>
                <span className="start-chip">Branching Story Paths</span>
                <span className="start-chip">Your Decisions Matter</span>
              </div>
              <button type="button" className="start-screen__cta" onClick={handleExperienceStart}>
                Begin Experience
              </button>
              <p className="start-screen__hint">Sound on · Best viewed fullscreen</p>
            </div>
          </div>
        )}

        <div className={`choice-drawer ${shouldShowChoiceDrawer ? "visible" : ""}`}>
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
          onSelect={handleTimelineSelect}
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
