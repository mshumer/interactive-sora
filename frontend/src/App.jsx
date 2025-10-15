import { useMemo, useState } from "react";
import axios from "axios";
import ConfigScreen from "./components/ConfigScreen.jsx";
import ExperienceScreen from "./components/ExperienceScreen.jsx";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";

const api = axios.create({
  baseURL: API_BASE_URL || undefined,
});

const App = () => {
  const [phase, setPhase] = useState("config");
  const [sessionId, setSessionId] = useState(null);
  const [story, setStory] = useState([]);
  const [stepCount, setStepCount] = useState(0);
  const [maxSteps, setMaxSteps] = useState(10);
  const [hasRemainingSteps, setHasRemainingSteps] = useState(true);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isGenerating, setIsGenerating] = useState(false);
  const [globalError, setGlobalError] = useState(null);
  const [configSnapshot, setConfigSnapshot] = useState(null);

  const handleConfigSubmit = async (config) => {
    setIsSubmitting(true);
    setGlobalError(null);

    try {
      const payload = {
        apiKey: config.apiKey,
        plannerModel: config.plannerModel,
        soraModel: config.soraModel,
        videoSize: config.videoSize,
        basePrompt: config.basePrompt,
      };
      const { data } = await api.post("/api/session", payload);
      setSessionId(data.sessionId);
      setStory(data.story);
      setStepCount(data.stepCount);
      setMaxSteps(data.maxSteps);
      setHasRemainingSteps(data.hasRemainingSteps);
      setConfigSnapshot({
        ...config,
      });
      setPhase("experience");
    } catch (error) {
      const message = error.response?.data?.detail || error.message || "Failed to start session.";
      setGlobalError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleChoice = async (choiceIndex) => {
    if (!sessionId) return;
    setIsGenerating(true);
    setGlobalError(null);

    try {
      const { data } = await api.post(`/api/session/${sessionId}/choice`, { choiceIndex });
      setStory(data.story);
      setStepCount(data.stepCount);
      setMaxSteps(data.maxSteps);
      setHasRemainingSteps(data.hasRemainingSteps);
    } catch (error) {
      const message = error.response?.data?.detail || error.message || "Failed to advance story.";
      setGlobalError(message);
    } finally {
      setIsGenerating(false);
    }
  };

  const context = useMemo(
    () => ({
      sessionId,
      configSnapshot,
      story,
      stepCount,
      maxSteps,
      hasRemainingSteps,
    }),
    [sessionId, configSnapshot, story, stepCount, maxSteps, hasRemainingSteps]
  );

  return phase === "config" ? (
    <ConfigScreen
      onSubmit={handleConfigSubmit}
      isSubmitting={isSubmitting}
      error={globalError}
      apiBaseUrl={API_BASE_URL}
    />
  ) : (
    <ExperienceScreen
      context={context}
      onMakeChoice={handleChoice}
      isGenerating={isGenerating}
      error={globalError}
      apiBaseUrl={API_BASE_URL}
      onRestart={() => {
        setPhase("config");
        setSessionId(null);
        setStory([]);
        setStepCount(0);
        setMaxSteps(10);
        setHasRemainingSteps(true);
        setConfigSnapshot(null);
        setGlobalError(null);
      }}
    />
  );
};

export default App;
