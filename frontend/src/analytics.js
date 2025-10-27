import mixpanel from "mixpanel-browser";
import { API_BASE_URL, MIXPANEL_TOKEN, WORLD_ID } from "./config.js";

let initialised = false;

const resolveEndpoint = (path) => {
  if (!API_BASE_URL) return path;
  const trimmed = API_BASE_URL.replace(/\/$/, "");
  return `${trimmed}${path}`;
};

const fetchIdentity = async () => {
  try {
    const response = await fetch(resolveEndpoint("/analytics/identity"), {
      credentials: "omit",
    });
    if (!response.ok) {
      return null;
    }
    const payload = await response.json();
    return typeof payload?.distinctId === "string" ? payload.distinctId : null;
  } catch (error) {
    return null;
  }
};

export const initAnalytics = async () => {
  if (initialised || !MIXPANEL_TOKEN) {
    return;
  }
  initialised = true;
  mixpanel.init(MIXPANEL_TOKEN, {
    track_pageview: false,
    persistence: "localStorage",
  });
  mixpanel.register({ world_id: WORLD_ID });
  const distinctId = await fetchIdentity();
  if (distinctId) {
    mixpanel.identify(distinctId);
  }
};

export const trackEvent = (eventName, properties = {}) => {
  if (!MIXPANEL_TOKEN) {
    return;
  }
  mixpanel.track(eventName, properties);
};
