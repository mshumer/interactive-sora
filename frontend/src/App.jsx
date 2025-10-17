import { useEffect, useMemo, useState, useCallback, useRef } from "react";
import axios from "axios";
import ExperienceScreen from "./components/ExperienceScreen.jsx";

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || "";
const WORLD_ID = import.meta.env.VITE_WORLD_ID || "default";
const API_KEY_STORAGE_KEY = "sora_shared_world_api_key";

const api = axios.create({
  baseURL: API_BASE_URL || undefined,
});

const delay = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const buildAssetUrl = (value) => {
  if (!value) return null;
  if (/^https?:/i.test(value)) return value;
  return `${API_BASE_URL}${value}`;
};

const App = () => {
  const [worldInfo, setWorldInfo] = useState(null);
  const [story, setStory] = useState([]);
  const [activePath, setActivePath] = useState("");
  const [apiKey, setApiKey] = useState("");
  const [isGenerating, setIsGenerating] = useState(false);
  const [isPolling, setIsPolling] = useState(false);
  const [globalError, setGlobalError] = useState(null);
  const [showKeyModal, setShowKeyModal] = useState(false);
  const [pendingChoice, setPendingChoice] = useState(null);
  const [prefetchedScenes, setPrefetchedScenes] = useState({});
  const prefetchedAssetUrls = useRef(new Set());
  const inFlightPrefetch = useRef(new Set());

  const prefetchSceneAssets = useCallback((scene) => {
    if (!scene || typeof window === "undefined") return;

    const record = prefetchedAssetUrls.current;

    const primeVideo = (rawUrl) => {
      const url = buildAssetUrl(rawUrl);
      if (!url || record.has(url)) return;
      const link = document.createElement("link");
      link.rel = "preload";
      link.as = "video";
      link.href = url;
      link.crossOrigin = "anonymous";
      document.head.appendChild(link);
      record.add(url);
    };

    const primeImage = (rawUrl) => {
      const url = buildAssetUrl(rawUrl);
      if (!url || record.has(url)) return;
      const image = new Image();
      image.src = url;
      record.add(url);
    };

    primeVideo(scene.videoUrl);
    primeImage(scene.posterUrl);
  }, []);

  useEffect(() => {
    const stored = window.localStorage.getItem(API_KEY_STORAGE_KEY);
    if (stored) {
      setApiKey(stored);
    }
  }, []);

  useEffect(() => {
    const bootstrap = async () => {
      try {
        const [worldRes, rootRes] = await Promise.all([
          api.get(`/worlds/${WORLD_ID}`),
          api.get(`/worlds/${WORLD_ID}/scenes`, { params: { path: "" } }),
        ]);
        setWorldInfo(worldRes.data);
        setStory([rootRes.data]);
        setActivePath(rootRes.data.path || "");
      } catch (error) {
        const message = error.response?.data?.detail || error.message || "Failed to load world.";
        setGlobalError(message);
      }
    };
    bootstrap();
  }, []);

  const updateStoryWithScene = useCallback((scene) => {
    setStory((prev) => {
      const next = prev.filter((step) => step.depth < scene.depth);
      next.push(scene);
      return next;
    });
    setActivePath(scene.path);
  }, []);

  const fetchScene = useCallback(async (path) => {
    const { data } = await api.get(`/worlds/${WORLD_ID}/scenes`, { params: { path } });
    return data;
  }, []);

  const pollForScene = useCallback(async (path, onUpdate) => {
    setIsPolling(true);
    try {
      for (let attempt = 0; attempt < 120; attempt += 1) {
        const scene = await fetchScene(path);
        onUpdate?.(scene);
        if (scene.status === "ready") {
          return scene;
        }
        if (scene.status === "failed") {
          const detail = scene.failureDetail || scene.failureCode || "Generation failed.";
          throw new Error(detail);
        }
        await delay(3000);
      }
      throw new Error("Timed out waiting for scene generation.");
    } finally {
      setIsPolling(false);
    }
  }, [fetchScene]);

  const ensureSceneReady = useCallback(
    async (path, key) => {
      setIsGenerating(true);
      setGlobalError(null);
      try {
        const { data: kickoff } = await api.post(`/worlds/${WORLD_ID}/scenes`, {
          path,
          apiKey: key,
        });
        if (kickoff.status === "ready") {
          updateStoryWithScene(kickoff);
          return;
        }
        if (kickoff.status === "failed") {
          const detail = kickoff.failureDetail || kickoff.failureCode || "Generation failed.";
          throw new Error(detail);
        }
        updateStoryWithScene(kickoff);
        const finished = await pollForScene(path, updateStoryWithScene);
        updateStoryWithScene(finished);
      } finally {
        setIsGenerating(false);
      }
    },
    [pollForScene, updateStoryWithScene]
  );

  const handleChoice = useCallback(
    async (choiceIndex) => {
      if (!story.length) return;
      const activeScene = story[story.length - 1];
      setGlobalError(null);
      const status = activeScene.choicesStatus[choiceIndex] || "pending";
      const childPath = activeScene.childrenPaths[choiceIndex] || buildChildPath(activeScene.path, choiceIndex);

      if (status === "ready") {
        const cached = prefetchedScenes[childPath];
        if (cached?.status === "ready") {
          updateStoryWithScene(cached);
          setPrefetchedScenes((prev) => {
            const next = { ...prev };
            delete next[childPath];
            return next;
          });
          prefetchSceneAssets(cached);
          return;
        }
        try {
          const nextScene = await fetchScene(childPath);
          if (nextScene?.status === "ready") {
            prefetchSceneAssets(nextScene);
            setPrefetchedScenes((prev) => {
              const next = { ...prev };
              delete next[childPath];
              return next;
            });
          }
          updateStoryWithScene(nextScene);
        } catch (error) {
          const message = error.response?.data?.detail || error.message || "Failed to load scene.";
          setGlobalError(message);
        }
        return;
      }

      if (!apiKey) {
        setPendingChoice({ index: choiceIndex, path: childPath });
        setShowKeyModal(true);
        return;
      }

      try {
        await ensureSceneReady(childPath, apiKey);
      } catch (error) {
        const message = error.response?.data?.detail || error.message || error.toString();
        setGlobalError(message);
      }
    },
    [apiKey, ensureSceneReady, fetchScene, story, updateStoryWithScene, prefetchedScenes, prefetchSceneAssets]
  );

  const handleApiKeySubmit = useCallback(
    async (key) => {
      const trimmed = key.trim();
      setApiKey(trimmed);
      window.localStorage.setItem(API_KEY_STORAGE_KEY, trimmed);
      setShowKeyModal(false);
      if (pendingChoice) {
        try {
          await ensureSceneReady(pendingChoice.path, trimmed);
        } catch (error) {
          const message = error.response?.data?.detail || error.message || error.toString();
          setGlobalError(message);
        } finally {
          setPendingChoice(null);
        }
      }
    },
    [ensureSceneReady, pendingChoice]
  );

  const handleKeyCancel = useCallback(() => {
    setPendingChoice(null);
    setShowKeyModal(false);
  }, []);

  const handlePromptForKey = useCallback(
    (path) => {
      if (typeof path === "string") {
        setPendingChoice({ index: null, path });
      }
      setShowKeyModal(true);
    },
    []
  );

  useEffect(() => {
    const activeScene = story[story.length - 1];
    if (!activeScene) return;

    const statuses = activeScene.choicesStatus || [];
    const children = activeScene.childrenPaths || [];

    statuses.forEach((status, index) => {
      if (status !== "ready") return;
      const childPath = children[index] || buildChildPath(activeScene.path, index);
      if (!childPath || prefetchedScenes[childPath] || inFlightPrefetch.current.has(childPath)) return;

      inFlightPrefetch.current.add(childPath);
      fetchScene(childPath)
        .then((scene) => {
          if (!scene || scene.status !== "ready") return;
          setPrefetchedScenes((prev) => {
            if (prev[childPath]) return prev;
            return { ...prev, [childPath]: scene };
          });
          prefetchSceneAssets(scene);
        })
        .catch(() => {
          // Swallow prefetch errors; user interaction will retry.
        })
        .finally(() => {
          inFlightPrefetch.current.delete(childPath);
        });
    });
  }, [story, fetchScene, prefetchedScenes, prefetchSceneAssets]);

  useEffect(() => {
    const latest = story[story.length - 1];
    if (!latest) return undefined;
    if (latest.status !== "queued") return undefined;
    if (isGenerating || isPolling) return undefined;

    let cancelled = false;
    const watch = async () => {
      try {
        const finished = await pollForScene(latest.path || "", updateStoryWithScene);
        if (!cancelled && finished) {
          updateStoryWithScene(finished);
        }
      } catch (error) {
        if (!cancelled) {
          const message = error?.response?.data?.detail || error?.message || "Failed to load scene.";
          setGlobalError(message);
        }
      }
    };

    watch();
    return () => {
      cancelled = true;
    };
  }, [story, isGenerating, isPolling, pollForScene, updateStoryWithScene]);

  const context = useMemo(
    () => ({
      worldInfo,
      story,
      activePath,
      apiKey,
      isPolling,
    }),
    [worldInfo, story, activePath, apiKey, isPolling]
  );

  return (
    <ExperienceScreen
      context={context}
      onMakeChoice={handleChoice}
      isGenerating={isGenerating || isPolling}
      error={globalError}
      apiBaseUrl={API_BASE_URL}
      onRestart={async () => {
        try {
          prefetchedAssetUrls.current.clear();
          inFlightPrefetch.current.clear();
          setPrefetchedScenes({});
          const rootScene = await fetchScene("");
          setStory([rootScene]);
          setActivePath(rootScene.path || "");
          setGlobalError(null);
        } catch (error) {
          const message = error.response?.data?.detail || error.message || "Failed to reset.";
          setGlobalError(message);
        }
      }}
      onPromptForKey={handlePromptForKey}
      showKeyModal={showKeyModal}
      onKeySubmit={handleApiKeySubmit}
      onKeyCancel={handleKeyCancel}
    />
  );
};

const buildChildPath = (path, index) => {
  if (!path) return String(index);
  return `${path}/${index}`;
};

export default App;
