// Popup: shows the current tab's verdict, gate weights, reasons, and the
// always-on / aggressive controls.

const DEFAULTS = { enabled: true, aggressive: true, aggroThreshold: 0.35 };

function render(result, up, conf) {
  const app = document.getElementById("app");
  document.getElementById("status").className = "dot " + (up ? "up" : "down");

  if (!up) {
    app.innerHTML = `<div class="off">⚠ Local server offline.<br>Start it with
      <code>bash scripts/serve.sh</code> so PhishNet can scan pages.</div>` + controls(conf);
    wire(conf); return;
  }
  if (!result) { app.innerHTML = "<div class='sub'>No scan yet for this tab. Reload the page.</div>" + controls(conf); wire(conf); return; }

  const p = result.phishing_probability, phish = result.verdict === "phishing" ||
    (conf.aggressive && p >= conf.aggroThreshold);
  const cls = phish ? "phish" : (p > 0.15 ? "warn" : "legit");
  const label = phish ? "⚠ Phishing" : (p > 0.15 ? "Suspicious" : "✓ Looks safe");
  const g = result.modality_gates;
  const gates = g ? `<div class="gates">
    ${["url","visual","text"].map(k=>`<div class="g"><div class="k">${k}</div>
    <div class="v">${Math.round((g[k]||0)*100)}%</div></div>`).join("")}</div>` : "";
  const reasons = (result.explanation && result.explanation.summary || [])
    .slice(0,4).map(r=>`<li>${r}</li>`).join("");

  app.innerHTML = `
    <div class="verdict ${cls}">${label}</div>
    <div class="sub">${(p*100).toFixed(1)}% · ${result.decided_by||"model"} · ${result.url||""}</div>
    ${gates}
    <ul>${reasons||"<li>No dominant signal.</li>"}</ul>
    <button class="re" id="rescan">Re-scan this page</button>
    ${controls(conf)}`;
  wire(conf);
}

function controls(conf) {
  return `
    <div class="row"><span>Always-on scanning</span>
      <div class="toggle ${conf.enabled?"on":""}" id="t-enabled"><div class="knob"></div></div></div>
    <div class="row"><span>Aggressive mode</span>
      <div class="toggle ${conf.aggressive?"on":""}" id="t-aggro"><div class="knob"></div></div></div>`;
}

function wire(conf) {
  const re = document.getElementById("rescan");
  if (re) re.onclick = async () => {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    chrome.runtime.sendMessage({ type: "PHISHNET_RESCAN", tabId: tab.id, url: tab.url }, () => load());
  };
  const te = document.getElementById("t-enabled");
  if (te) te.onclick = () => chrome.storage.local.set({ enabled: !conf.enabled }, load);
  const ta = document.getElementById("t-aggro");
  if (ta) ta.onclick = () => chrome.storage.local.set({ aggressive: !conf.aggressive }, load);
}

async function load() {
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const conf = await new Promise(r => chrome.storage.local.get(DEFAULTS, v => r({ ...DEFAULTS, ...v })));
  chrome.runtime.sendMessage({ type: "PHISHNET_GET", tabId: tab.id }, (resp) => {
    render(resp ? resp.result : null, resp ? resp.up : false, conf);
  });
}
load();
