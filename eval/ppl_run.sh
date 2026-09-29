#!/bin/zsh
# P9 eval — standalone PPL runner (mirrors engine_daemon.sh env composition,
# minus the daemon attach; writes the GPU lock like the wrapper does).
# usage: ppl_run.sh moe|dense
set -u
OPS=~/tinygrad-metal/engine0/ops
WHICH=${1:?usage: ppl_run.sh moe|dense}
if [ "$WHICH" = "moe" ]; then
  MODEL_ENV=$OPS/env.canonical.d/qwen3.6-35b-a3b-egpu.env
  SCRIPT=~/tinygrad-metal/eval/ppl_moe.py
else
  MODEL_ENV=$OPS/env.canonical.d/qwen3.8-27b-egpu.env
  SCRIPT=~/tinygrad-metal/eval/ppl_dense.py
fi
LOCK=/tmp/nv_usb4.lock
set -a
source $OPS/env.common
source $MODEL_ENV
set +a
export M1A_SERVE=0
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"
export DOCKER_HOST="${DOCKER_HOST:-unix://~/.colima/default/docker.sock}"
print -r -- "$$ $(date +%s) p9_ppl_$WHICH" > $LOCK
trap 'rm -f $LOCK' EXIT INT TERM
cd ~/tinygrad-metal/engine0
exec ~/tg311/bin/python -u $SCRIPT
