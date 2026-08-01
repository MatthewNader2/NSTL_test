// src/hooks/useApi.js

const ALLOWED_ORIGINS_PATTERN = /^https?:\/\/(localhost|127\.0\.0\.1|10\.0\.2\.2)(:\d+)?$/;

function safeGetStorage(key) {
  try {
    if (typeof window !== "undefined" && "localStorage" in window && window.localStorage) {
      return window.localStorage.getItem(key);
    }
  } catch (e) {}
  return null;
}

function safeSetStorage(key, val) {
  try {
    if (typeof window !== "undefined" && "localStorage" in window && window.localStorage) {
      window.localStorage.setItem(key, val);
    }
  } catch (e) {}
}

function safeRemoveStorage(key) {
  try {
    if (typeof window !== "undefined" && "localStorage" in window && window.localStorage) {
      window.localStorage.removeItem(key);
    }
  } catch (e) {}
}

export function getApiBase() {
  const saved = safeGetStorage("nstl_api_base");
  if (saved && ALLOWED_ORIGINS_PATTERN.test(saved)) return saved;
  if (saved && !ALLOWED_ORIGINS_PATTERN.test(saved)) {
    console.warn(`[NSTL Security] Rejected untrusted API base URL from localStorage: "${saved}". Falling back to default.`);
    safeRemoveStorage("nstl_api_base");
  }
  if (/Android/i.test(navigator.userAgent)) {
    return "http://10.0.2.2:58102";
  }
  if (typeof window !== "undefined" && window.location && window.location.origin && window.location.origin.startsWith("http")) {
    return window.location.origin;
  }
  return "http://127.0.0.1:58102";
}

export function setApiBase(url) {
  let cleanedUrl = url.trim();
  if (cleanedUrl && !cleanedUrl.startsWith("http://") && !cleanedUrl.startsWith("https://")) {
    cleanedUrl = "http://" + cleanedUrl;
  }
  if (!ALLOWED_ORIGINS_PATTERN.test(cleanedUrl)) {
    console.warn(`[NSTL Security] Refused to save non-local API base: "${cleanedUrl}"`);
    return;
  }
  safeSetStorage("nstl_api_base", cleanedUrl);
}

/**
 * Checks fetch response and throws on HTTP errors, avoiding silent json() parse failures.
 */
async function checkedJson(res) {
  if (!res.ok) {
    const errorText = await res.text().catch(() => "Unknown error");
    throw new Error(`API error ${res.status}: ${errorText}`);
  }
  return res.json();
}

export async function fetchStatus() {
  const base = getApiBase();
  return fetch(`${base}/api/status`);
}

export async function fetchHealth() {
  const base = getApiBase();
  return fetch(`${base}/api/health`);
}

export async function fetchCells() {
  const base = getApiBase();
  const res = await fetch(`${base}/api/cells`);
  const data = await checkedJson(res);
  return data.cells || [];
}

export async function runPrompt(prompt) {
  const base = getApiBase();
  const res = await fetch(`${base}/api/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  return checkedJson(res);
}

export async function initializeEngine(profile, embedder_model, llm_model, embedder_device, llm_device, trees_storage) {
  const base = getApiBase();
  const res = await fetch(`${base}/api/initialize`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ profile, embedder_model, llm_model, embedder_device, llm_device, trees_storage }),
  });
  return checkedJson(res);
}

export async function toggleBenchmark() {
  const base = getApiBase();
  const res = await fetch(`${base}/api/benchmark/toggle`, { method: "POST" });
  return checkedJson(res);
}

export async function getBenchmarkStatus() {
  const base = getApiBase();
  const res = await fetch(`${base}/api/benchmark/status`);
  return checkedJson(res);
}

export async function fetchAvailableModels() {
  const base = getApiBase();
  const res = await fetch(`${base}/api/models`);
  return checkedJson(res);
}
