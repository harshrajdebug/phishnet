// PhishNet service worker — automatic, always-on phishing guard.
//
// Lifecycle per main-frame navigation:
//   1. onBeforeNavigate  -> instant URL-only scan (≈5 ms). If it already looks
//      phishing, block before the page paints.
//   2. onCompleted       -> capture a screenshot + pull the visible page text
//      from the content script, then run the FULL multimodal scan and re-decide.
// The extension applies its own AGGRESSIVE threshold on top of the server verdict,
// so borderline pages are blocked rather than waved through. It fails open only
// when the backend is unreachable (never trap the user offline), and it never
// scans the reputation-trusted big brands twice.

const API = "http://127.0.0.1:8000";
const DEFAULTS = { enabled: true, aggressive: true, aggroThreshold: 0.35 };

const verdicts = {};      // tabId -> last result
const pageText = {};      // tabId -> visible text captured by content script

function cfg() {
  return new Promise((res) =>
    chrome.storage.local.get(DEFAULTS, (v) => res({ ...DEFAULTS, ...v })));
}

async function post(path, body) {
  try {
    const r = await fetch(API + path, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    return r.ok ? await r.json() : null;
  } catch (e) {
    return { _offline: true };
  }
}

async function health() {
  try {
    const r = await fetch(API + "/health", { cache: "no-store" });
    return r.ok;
  } catch (e) { return false; }
}

// A page is "dangerous" under the aggressive policy if the model calls it
// phishing OR its phishing probability clears the (lower) aggressive bar.
function dangerous(result, conf) {
  if (!result || result._offline) return false;
  if (result.verdict === "phishing") return true;
  if (conf.aggressive && result.phishing_probability >= conf.aggroThreshold) return true;
  return false;
}

function badge(tabId, result, conf) {
  let text = "…", color = "#9aa0a6";
  if (result && result._offline) { text = "⚠"; color = "#9aa0a6"; }
  else if (result) {
    if (dangerous(result, conf)) { text = "!"; color = "#d93025"; }
    else if (result.phishing_probability > 0.15) { text = "?"; color = "#f9ab00"; }
    else { text = "✓"; color = "#1e8e3e"; }
  }
  chrome.action.setBadgeText({ tabId, text });
  chrome.action.setBadgeBackgroundColor({ tabId, color });
}

async function decide(tabId, result) {
  const conf = await cfg();
  verdicts[tabId] = result;
  badge(tabId, result, conf);
  if (dangerous(result, conf)) {
    chrome.tabs.sendMessage(tabId, { type: "PHISHNET_BLOCK", result, conf }).catch(() => {});
  } else {
    chrome.tabs.sendMessage(tabId, { type: "PHISHNET_CLEAR" }).catch(() => {});
  }
}

// ---- Stage 1: instant URL scan, before paint ----
chrome.webNavigation.onBeforeNavigate.addListener(async ({ tabId, url, frameId }) => {
  if (frameId !== 0 || !/^https?:/.test(url)) return;
  const conf = await cfg();
  if (!conf.enabled) return;
  const result = await post("/scan", { url, explain: true });
  await decide(tabId, result);
});

// ---- Stage 2: full multimodal scan after the page loads ----
chrome.webNavigation.onCompleted.addListener(async ({ tabId, url, frameId }) => {
  if (frameId !== 0 || !/^https?:/.test(url)) return;
  const conf = await cfg();
  if (!conf.enabled) return;

  // Screenshot (visual modality). Only for the active tab; captureVisibleTab
  // cannot grab a background tab.
  let shot = null;
  try {
    const tab = await chrome.tabs.get(tabId);
    if (tab.active) {
      shot = await chrome.tabs.captureVisibleTab(tab.windowId, { format: "png" });
    }
  } catch (e) { /* ignore */ }

  const text = pageText[tabId] || null;
  const result = await post("/scan", {
    url, text, screenshot_b64: shot, explain: true,
  });
  if (result && !result._offline) await decide(tabId, result);
});

// ---- messages from content script / popup ----
chrome.runtime.onMessage.addListener((msg, sender, reply) => {
  const tabId = sender.tab ? sender.tab.id : msg.tabId;
  if (msg.type === "PHISHNET_PAGE_TEXT") {
    pageText[tabId] = msg.text;
  } else if (msg.type === "PHISHNET_GET") {
    health().then((up) => reply({ result: verdicts[msg.tabId] || null, up }));
    return true;
  } else if (msg.type === "PHISHNET_RESCAN") {
    (async () => {
      const result = await post("/scan", { url: msg.url, text: pageText[msg.tabId] || null, explain: true });
      await decide(msg.tabId, result);
      reply({ result });
    })();
    return true;
  }
});

chrome.tabs.onRemoved.addListener((tabId) => {
  delete verdicts[tabId]; delete pageText[tabId];
});
