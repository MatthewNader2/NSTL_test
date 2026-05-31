// src/hooks/useApi.js

export function getApiBase() {
  const saved = localStorage.getItem("nstl_api_base");
  if (saved) return saved;
  if (/Android/i.test(navigator.userAgent)) {
    return "http://10.0.2.2:8000";
  }
  return "http://127.0.0.1:8000";
}

export function setApiBase(url) {
  let cleanedUrl = url.trim();
  if (cleanedUrl && !cleanedUrl.startsWith("http://") && !cleanedUrl.startsWith("https://")) {
    cleanedUrl = "http://" + cleanedUrl;
  }
  localStorage.setItem("nstl_api_base", cleanedUrl);
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
  return res.json();
}

export async function runPrompt(prompt) {
  const base = getApiBase();
  const res = await fetch(`${base}/api/run`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt }),
  });
  return res.json();
}
