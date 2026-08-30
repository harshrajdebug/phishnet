#!/usr/bin/env bash
# End-to-end reproduction of every PhishNet result, in dependency order.
# Each stage writes JSON to results/ and is safe to re-run (idempotent caches).
set -euo pipefail
cd "$(dirname "$0")/.."
PY="${PY:-.venv/bin/python}"
export PYTHONPATH=.

echo "==> [1/6] acquire real corpora (OpenPhish, Phishing.Database, Tranco, Nazario, SpamAssassin)"
$PY -m phishnet.data.acquire

echo "==> [2/6] harvest benign deep links (fixes the path/length artefact)"
$PY -m phishnet.data.harvest_benign "${BENIGN_DOMAINS:-9000}"

echo "==> [3/6] render visual corpus (brand anchors + look-alike clones)"
$PY -m phishnet.data.render

echo "==> [4/6] train URL branch + classical baselines"
$PY -m phishnet.train.train_url

echo "==> [5a/6] train text branch + TF-IDF baseline"
$PY -m phishnet.train.train_text
echo "==> [5b/6] train visual Siamese branch + build brand index"
$PY -m phishnet.train.train_visual
echo "==> [5c/6] train fusion + run cross-attention vs concat ablation"
$PY -m phishnet.train.train_fusion

echo "==> [5d/6] modality ablation + per-config deployment thresholds"
$PY -m phishnet.eval.ablation

echo "==> [6/6] consolidate benchmark tables + figures"
$PY -m phishnet.eval.benchmark
$PY -m phishnet.eval.make_figures

echo "done. See results/ and paper/figures/."
