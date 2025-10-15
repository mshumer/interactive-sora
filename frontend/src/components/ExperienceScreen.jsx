import { useEffect, useMemo, useRef, useState } from "react";
import Timeline from "./Timeline.jsx";
import "../styles/experience.css";

const ExperienceScreen = ({ context, onMakeChoice, isGenerating, error, apiBaseUrl, onRestart }) => {
  const { story, hasRemainingSteps } = context;
  const [activeIndex, setActiveIndex] = useState(Math.max(story.length - 1, 0));
  const [hasVideoEnded, setHasVideoEnded] = useState(false);
  const [showStoryboard, setShowStoryboard] = useState(false);
  const [isVideoLoading, setIsVideoLoading] = useState(false);
  const videoRef = useRef(null);

  useEffect(() => {
    if (story.length > 0) {
      const latestIndex = story.length - 1;
      setActiveIndex(latestIndex);
      const nextScene = story[latestIndex];
      setHasVideoEnded(!nextScene?.videoUrl);
      setShowStoryboard(false);
    }
  }, [story]);

  useEffect(() => {
    const scene = story[activeIndex];
    const isLatest = activeIndex === story.length - 1;
    const alreadyLocked = Boolean(scene?.choiceIndex !== null);
    setHasVideoEnded(!scene?.videoUrl || alreadyLocked || !isLatest);
    setIsVideoLoading(Boolean(scene?.videoUrl));
  }, [activeIndex, story]);

  const activeScene = story[activeIndex] || null;

  const videoSrc = useMemo(() => {
    if (!activeScene?.videoUrl) return null;
    return `${apiBaseUrl}${activeScene.videoUrl}`;
  }, [activeScene?.videoUrl, apiBaseUrl]);

  const posterSrc = useMemo(() => {
    if (!activeScene?.posterUrl) return null;
    return `${apiBaseUrl}${activeScene.posterUrl}`;
  }, [activeScene?.posterUrl, apiBaseUrl]);

  const isCurrentScene = activeIndex === story.length - 1;
  const choiceLocked = activeScene?.choiceIndex !== null;
  const canRevealChoices = hasVideoEnded || !videoSrc;
  const canChoose = Boolean(
    isCurrentScene && !choiceLocked && hasRemainingSteps && !isGenerating && canRevealChoices
  );

  const handleChoice = (index) => {
    if (!canChoose) return;
    onMakeChoice(index);
  };

  const handleVideoEnd = () => {
    setHasVideoEnded(true);
  };

  const handleReplay = () => {
    setHasVideoEnded(false);
    if (videoRef.current) {
      videoRef.current.currentTime = 0;
      videoRef.current.play().catch(() => setHasVideoEnded(true));
    }
  };

  const handleLoadedData = () => {
    setIsVideoLoading(false);
  };

  const awaitingScene = isGenerating || (!videoSrc && !activeScene.plannerMissingPrompt);
  const showLoader = awaitingScene || (Boolean(videoSrc) && isVideoLoading);
  const allowReplay = Boolean(videoSrc && canRevealChoices);

  if (!activeScene) {
    return null;
  }

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
            onEnded={handleVideoEnd}
            onPlay={() => setHasVideoEnded(false)}
            onLoadedData={handleLoadedData}
          />
        ) : (
          <div className="video-placeholder">
            <p>{activeScene.plannerMissingPrompt ? "Prompt missing — no video available." : "Video generating…"}</p>
          </div>
        )}
        <div className="video-mask" />

        <div className="scene-pill">Scene {activeScene.sceneNumber.toString().padStart(2, "0")}</div>

        <div className="floating-controls">
          <button type="button" className="control-button" onClick={() => setShowStoryboard(true)}>
            Storyboard
          </button>
          <button type="button" className="control-button" onClick={onRestart}>
            Restart
          </button>
        </div>

        {allowReplay && (
          <button type="button" className="replay-float" onClick={handleReplay}>
            Replay Scene
          </button>
        )}

        <div className={`choice-drawer ${canRevealChoices ? "visible" : ""}`}>
          <div className="drawer-inner">
            <div className="drawer-header">
              <span className={choiceLocked ? "status resolved" : canChoose ? "status awaiting" : "status locked"}>
                {choiceLocked ? "Choice locked" : canChoose ? "Select your path" : "Awaiting cinematic moment"}
              </span>
            </div>
            <div className="choice-grid">
              {activeScene.choices.map((choice, index) => {
                const isSelected = activeScene.choiceIndex === index;
                return (
                  <button
                    key={choice}
                    type="button"
                    className={`choice-pod ${isSelected ? "selected" : ""}`}
                    disabled={!canChoose && !isSelected}
                    onClick={() => handleChoice(index)}
                  >
                    <span className="choice-number">{index + 1}</span>
                    <span className="choice-copy">{choice}</span>
                  </button>
                );
              })}
            </div>
          </div>
        </div>

        <Timeline
          story={story}
          activeIndex={activeIndex}
          onSelect={(index) => {
            setActiveIndex(index);
          }}
          apiBaseUrl={apiBaseUrl}
          isOpen={showStoryboard}
          onClose={() => setShowStoryboard(false)}
        />

        {showLoader && (
          <div className="loading-veil">
            <div className="loading-core" />
            <p>{isGenerating ? "Synthesizing the next sequence…" : "Preparing scene…"}</p>
          </div>
        )}

        {error && <div className="error-pill">{error}</div>}
      </div>
    </div>
  );
};

export default ExperienceScreen;
