import "../styles/timeline.css";

const Timeline = ({ story, activeIndex, onSelect, apiBaseUrl, isOpen = false, onClose = () => {} }) => {
  return (
    <aside className={`storyboard ${isOpen ? "open" : ""}`} aria-hidden={!isOpen}>
      <div className="storyboard-backdrop" onClick={onClose} />
      <div className="storyboard-body">
        <header className="storyboard-header">
          <div>
            <p className="storyboard-label">Storyboard</p>
            <h3>Travel through your previous shots</h3>
          </div>
          <button type="button" className="storyboard-close" onClick={onClose} aria-label="Close storyboard">
            Close
          </button>
        </header>
        <div className="timeline-track">
          {story.map((scene, index) => {
            const poster = scene.posterUrl ? `${apiBaseUrl}${scene.posterUrl}` : null;
            const isActive = index === activeIndex;
            const hasDecision = scene.choiceIndex !== null;
            const chosen = hasDecision ? scene.choices[scene.choiceIndex] : null;

            return (
              <button
                key={scene.sceneNumber}
                type="button"
                className={`timeline-card ${isActive ? "active" : ""}`}
                onClick={() => {
                  onSelect(index);
                  onClose();
                }}
              >
                <div className="timeline-thumb">
                  {poster ? <img src={poster} alt={`Scene ${scene.sceneNumber}`} /> : <div className="thumb-fallback" />}
                </div>
                <div className="timeline-info">
                  <header>
                    <span className="timeline-number">Scene {scene.sceneNumber}</span>
                    <span className={`timeline-status ${hasDecision ? "locked" : "open"}`}>
                      {hasDecision ? "Resolved" : "Awaiting"}
                    </span>
                  </header>
                  <p className="timeline-description">{scene.scenarioDisplay}</p>
                  {chosen && <p className="timeline-choice">Chosen: {chosen}</p>}
                </div>
              </button>
            );
          })}
        </div>
      </div>
    </aside>
  );
};

export default Timeline;
