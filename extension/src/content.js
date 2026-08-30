// PhishNet content script — extracts page text and renders the block interstitial.
//
// Runs at document_start on every page. As soon as the body exists it sends the
// visible text to the service worker (the text modality), and it listens for a
// BLOCK message to throw up a full-screen, hard-to-dismiss warning. The block is
// the aggressive part: rather than a dismissible banner, phishing pages are
// covered entirely and the user must consciously choose to proceed.

const OVERLAY_ID = "phishnet-blocker";

function sendText() {
  try {
    const t = (document.body ? document.body.innerText : "").slice(0, 4000).trim();
    chrome.runtime.sendMessage({ type: "PHISHNET_PAGE_TEXT", text: t });
  } catch (e) {}
}
if (document.body) sendText();
else document.addEventListener("DOMContentLoaded", sendText, { once: true });

function removeOverlay() {
  const el = document.getElementById(OVERLAY_ID);
  if (el) el.remove();
  document.documentElement.style.overflow = "";
}

function showOverlay(result) {
  if (document.getElementById(OVERLAY_ID)) return;
  const pct = Math.round((result.phishing_probability || 0) * 100);
  const reasons = (result.explanation && result.explanation.summary || [])
    .slice(0, 5).map((r) => `<li>${escapeHtml(r)}</li>`).join("");
  const brand = result.explanation && result.explanation.brand_match;
  const brandLine = brand
    ? `<p class="pn-brand">This page mimics <b>${escapeHtml(brand.brand)}</b>
       (${Math.round(brand.similarity * 100)}% visual match) but is not served from
       <code>${escapeHtml(brand.legit_domain)}</code>.</p>` : "";

  const host = escapeHtml(location.host);
  const el = document.createElement("div");
  el.id = OVERLAY_ID;
  el.innerHTML = `
    <div class="pn-card">
      <div class="pn-shield">🛡️</div>
      <div class="pn-tag">PhishNet blocked this page</div>
      <h1>This site looks like phishing</h1>
      <p class="pn-host">${host}</p>
      <div class="pn-meter"><span style="width:${Math.max(pct,4)}%"></span></div>
      <p class="pn-prob">${pct}% phishing confidence · flagged by ${escapeHtml(result.decided_by || "model")}</p>
      ${brandLine}
      <ul class="pn-why">${reasons || "<li>Multiple suspicious signals.</li>"}</ul>
      <div class="pn-actions">
        <button id="pn-back">Get me out of here</button>
        <button id="pn-proceed" class="pn-ghost">Ignore &amp; continue (unsafe)</button>
      </div>
      <p class="pn-foot">Do not enter passwords, card numbers, or codes on this page.</p>
    </div>`;
  const css = document.createElement("style");
  css.textContent = STYLE;
  el.appendChild(css);
  document.documentElement.appendChild(el);
  document.documentElement.style.overflow = "hidden";

  document.getElementById("pn-back").onclick = () => {
    if (history.length > 1) history.back();
    else location.href = "about:blank";
  };
  document.getElementById("pn-proceed").onclick = () => removeOverlay();
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

chrome.runtime.onMessage.addListener((msg) => {
  if (msg.type === "PHISHNET_BLOCK") showOverlay(msg.result);
  else if (msg.type === "PHISHNET_CLEAR") removeOverlay();
});

const STYLE = `
#${OVERLAY_ID}{position:fixed;inset:0;z-index:2147483647;
  background:rgba(10,14,20,.86);backdrop-filter:blur(6px);
  display:flex;align-items:center;justify-content:center;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
#${OVERLAY_ID} .pn-card{background:#fff;color:#16212b;max-width:460px;width:90%;
  border-radius:18px;padding:34px 30px;text-align:center;
  box-shadow:0 30px 80px rgba(0,0,0,.5);border-top:6px solid #d93025}
#${OVERLAY_ID} .pn-shield{font-size:44px}
#${OVERLAY_ID} .pn-tag{font:600 12px/1 ui-monospace,monospace;letter-spacing:.15em;
  text-transform:uppercase;color:#d93025;margin:8px 0 14px}
#${OVERLAY_ID} h1{font-size:26px;margin:0 0 6px;letter-spacing:-.02em}
#${OVERLAY_ID} .pn-host{font:600 14px ui-monospace,monospace;color:#5a6b78;word-break:break-all;margin:0 0 16px}
#${OVERLAY_ID} .pn-meter{height:8px;background:#f0e0dd;border-radius:6px;overflow:hidden;margin:0 0 6px}
#${OVERLAY_ID} .pn-meter>span{display:block;height:100%;background:#d93025}
#${OVERLAY_ID} .pn-prob{font-size:13px;color:#5a6b78;margin:0 0 14px}
#${OVERLAY_ID} .pn-brand{font-size:14px;background:#f7e7e2;border-radius:10px;padding:10px 12px;margin:0 0 14px}
#${OVERLAY_ID} .pn-why{text-align:left;margin:0 0 20px;padding-left:20px;font-size:14.5px;line-height:1.6}
#${OVERLAY_ID} .pn-actions{display:flex;flex-direction:column;gap:10px}
#${OVERLAY_ID} button{border:0;border-radius:10px;padding:13px;font-size:15px;font-weight:600;cursor:pointer}
#${OVERLAY_ID} #pn-back{background:#d93025;color:#fff}
#${OVERLAY_ID} .pn-ghost{background:transparent;color:#8a97a2;font-weight:500;font-size:13px;text-decoration:underline}
#${OVERLAY_ID} .pn-foot{font-size:12px;color:#8a97a2;margin:14px 0 0}
@media (prefers-color-scheme:dark){
  #${OVERLAY_ID} .pn-card{background:#141e28;color:#e7eef4}
  #${OVERLAY_ID} .pn-host,#${OVERLAY_ID} .pn-prob,#${OVERLAY_ID} .pn-foot{color:#9db0bd}
  #${OVERLAY_ID} .pn-brand{background:#2b1512}
  #${OVERLAY_ID} .pn-meter{background:#3a1f1b}
}`;
