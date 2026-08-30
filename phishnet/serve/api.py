"""FastAPI backend for the browser extension.

Endpoints are deliberately thin: the extension sends what it cheaply has (the
URL, optionally the visible message text and a screenshot), and gets back a
verdict plus an explanation it can render inline. The model is loaded once at
startup, not per request.
"""
from __future__ import annotations

import base64
import tempfile
import time
from collections import deque
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from phishnet.serve.inference import PhishNetEngine

app = FastAPI(title="PhishNet API", version="1.0",
              description="Multimodal, explainable phishing detection")

# In-memory activity log: every scan the extension (or anything) runs is recorded
# here so the dashboard can show live, per-URL activity -- not just that a request
# happened, but what was scanned and how it was judged. Bounded so it never grows
# without limit; nothing is written to disk.
ACTIVITY: deque = deque(maxlen=1000)
STATS = {"total": 0, "phishing": 0, "legitimate": 0, "since": time.time()}


def _record(url: str, result: dict) -> None:
    STATS["total"] += 1
    if result.get("verdict") == "phishing":
        STATS["phishing"] += 1
    else:
        STATS["legitimate"] += 1
    m = result.get("modalities_used", {})
    ACTIVITY.appendleft({
        "t": time.time(),
        "url": url,
        "verdict": result.get("verdict"),
        "p": result.get("phishing_probability"),
        "decided_by": result.get("decided_by"),
        "modalities": [k for k, v in m.items() if v],
        "gates": result.get("modality_gates"),
        "reasons": (result.get("explanation") or {}).get("summary", [])[:3],
        "latency_ms": result.get("latency_ms"),
    })

# The extension runs on arbitrary origins; scope this down to the extension ID
# in production. Kept permissive here for local evaluation only.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"],
                   allow_headers=["*"])

_engine: PhishNetEngine | None = None


def engine() -> PhishNetEngine:
    global _engine
    if _engine is None:
        _engine = PhishNetEngine()
    return _engine


class ScanRequest(BaseModel):
    url: str
    text: str | None = None
    screenshot_b64: str | None = None      # data-URL body, PNG
    explain: bool = True


class ScanResponse(BaseModel):
    url: str
    phishing_probability: float
    verdict: str
    threshold: float
    decided_by: str | None = None
    url_only_probability: float | None = None
    modalities_used: dict
    modality_gates: dict | None = None
    latency_ms: float
    explanation: dict | None = None


@app.on_event("startup")
def _warm():
    e = engine()
    # A warm forward pass so the first real request does not pay compile/JIT cost.
    e.predict("https://example.com/warmup")
    print(f"PhishNet ready (weights: {e.loaded_from or 'RANDOM INIT'}, "
          f"device: {e.device}, threshold: {e.threshold:.4f})")


@app.get("/", response_class=HTMLResponse)
def home():
    """A self-contained test page: type a URL, get an explained verdict."""
    page = Path(__file__).parent / "static" / "index.html"
    return page.read_text()


@app.get("/health")
def health():
    e = engine()
    return {"status": "ok", "device": e.device, "weights": e.loaded_from,
            "threshold": e.threshold,
            "branches": {"url": True,
                         "text": e.model.text_encoder is not None,
                         "visual": e.model.visual_encoder is not None}}


@app.post("/scan", response_model=ScanResponse)
def scan(req: ScanRequest):
    e = engine()
    img_path = None
    tmp = None
    if req.screenshot_b64:
        try:
            data = req.screenshot_b64.split(",", 1)[-1]
            raw = base64.b64decode(data)
            tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp.write(raw)
            tmp.close()
            img_path = tmp.name
        except Exception:
            img_path = None
    try:
        result = e.predict(req.url, text=req.text, image_path=img_path,
                           explain=req.explain)
    finally:
        if tmp:
            Path(tmp.name).unlink(missing_ok=True)
    _record(req.url, result)
    return result


@app.get("/activity")
def activity(limit: int = 200):
    """Recent scans plus running totals, for the live dashboard."""
    up = round(time.time() - STATS["since"])
    return {"stats": {**STATS, "uptime_s": up},
            "events": list(ACTIVITY)[:limit]}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard():
    """A live view of everything the extension is scanning."""
    return (Path(__file__).parent / "static" / "dashboard.html").read_text()


class TrustReq(BaseModel):
    domain: str


@app.get("/trusted")
def trusted_list():
    return {"domains": sorted(engine().user_allowlist)}


@app.post("/trusted")
def trusted_add(req: TrustReq):
    return {"domains": sorted(engine().add_trusted(req.domain))}


@app.delete("/trusted")
def trusted_remove(req: TrustReq):
    return {"domains": sorted(engine().remove_trusted(req.domain))}


def main():
    import uvicorn
    uvicorn.run("phishnet.serve.api:app", host="127.0.0.1", port=8000, reload=False)


if __name__ == "__main__":
    main()
