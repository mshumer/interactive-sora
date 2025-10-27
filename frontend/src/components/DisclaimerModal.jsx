import "../styles/experience.css";

const DisclaimerModal = ({ open, onAccept }) => {
  if (!open) return null;

  return (
    <div className="disclaimer-backdrop" role="dialog" aria-modal="true">
      <div className="disclaimer-modal">
        <h2>Heads up!</h2>
        <p>
          Keep in mind this is an open-source project, using publicly available models. It's an insanely early demo of
          what's to come, and is FAR from perfect today. Don't expect magic. Go in with an open mind. It's cool!
        </p>
        <p className="disclaimer-legal">
          Disclaimer: This project is a demonstration of experimental technology and provided "as-is" without warranties of
          any kind. By using this site, you acknowledge and agree that the creator is not liable for any damages, losses, or
          liabilities arising from your use of the platform or reliance on generated content.
        </p>
        <button type="button" className="disclaimer-cta" onClick={onAccept}>
          I Understand
        </button>
      </div>
    </div>
  );
};

export default DisclaimerModal;
