#!/bin/bash
# Resilient training driver: runs text -> visual -> fusion sequentially, fully
# detached from the parent so it survives a Claude Code app restart. Each stage
# writes its own results JSON; a stage is skipped if its results already exist.
cd /Users/harsh/SEPIACOBRA/phishnet
export TRANSFORMERS_VERBOSITY=error TOKENIZERS_PARALLELISM=false PYTHONPATH=.
PY=.venv/bin/python
LOG=results/driver.log
echo "=== driver start $(date) ===" > $LOG

run () { # stage name, module, results-file
  local name=$1 mod=$2 res=$3
  if [ -f "$res" ]; then echo "[$name] already done ($res), skip" >> $LOG; return 0; fi
  echo "[$name] START $(date)" >> $LOG
  $PY -m "$mod" >> $LOG 2>&1
  local rc=$?
  echo "[$name] END rc=$rc $(date)" >> $LOG
  [ -f "$res" ] && echo "[$name] wrote $res" >> $LOG || echo "[$name] MISSING $res" >> $LOG
  return $rc
}

run text   phishnet.train.train_text   results/text_results.json
run visual phishnet.train.train_visual results/visual_results.json
run fusion phishnet.train.train_fusion results/fusion_results.json
echo "=== driver done $(date) ===" >> $LOG
touch results/driver.done
