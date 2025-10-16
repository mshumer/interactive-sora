import "../styles/timeline.css";

const Timeline = ({ story, activeIndex, onSelect, apiBaseUrl, isOpen = false, onClose = () => {} }) => (
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
          const poster = scene.posterUrl
            ? /^https?:/i.test(scene.posterUrl)
              ? scene.posterUrl
              : `${apiBaseUrl}${scene.posterUrl}`
            : null;
          const isActive = index === activeIndex;
          const status = scene.status || "pending";
          const chosen = scene.triggerChoice || null;

          return (
            <button
              key={scene.path || index}
              type="button"
              className={`timeline-card ${isActive ? "active" : ""}`}
              onClick={() => {
                onSelect(index);
                onClose();
              }}
            >
              <div className="timeline-thumb">
                {poster ? <img src={poster} alt={`Scene ${index + 1}`} /> : <div className="thumb-fallback" />}
              </div>
              <div className="timeline-info">
                <header>
                  <span className="timeline-number">Scene {index + 1}</span>
                  <span className={`timeline-status ${status}`}>{statusLabel(status)}</span>
                </header>
                <p className="timeline-description">{scene.scenarioDisplay || "Awaiting generation."}</p>
                {chosen && <p className="timeline-choice">Branch via: {chosen}</p>}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  </aside>
);

const statusLabel = (status) => {
  switch (status) {
    case "ready":
      return "Ready";
    case "queued":
      return "Rendering";
    case "failed":
      return "Needs retry";
    default:
      return "Unexplored";
  }
};

export default Timeline;
