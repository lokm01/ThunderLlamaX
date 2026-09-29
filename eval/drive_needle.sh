#!/bin/zsh
# P9 eval driver for the needle test (driver-seat side; survives rig resets).
TAG=$1
RIG="${TLX_RIG:?set TLX_RIG=user@engine-host (the rig SSH target)}"
while true; do
  out=$(ssh -o BatchMode=yes -o ConnectTimeout=8 $RIG "
    n=\$(grep -c '\"trial\":' ~/tinygrad-metal/eval/results/needle_${TAG}.jsonl 2>/dev/null || echo 0)
    alive=\$(ps aux | grep needle_eval | grep -v grep | wc -l | tr -d ' ')
    health=\$(curl -s --max-time 3 http://127.0.0.1:8080/health 2>/dev/null | head -c 40)
    echo \"\$n|\$alive|\$health\"" 2>/dev/null)
  if [ -z "$out" ]; then echo "$(date +%H:%M:%S) rig unreachable"; sleep 45; continue; fi
  n=${out%%|*}; rest=${out#*|}; alive=${rest%%|*}; health=${rest#*|}
  echo "$(date +%H:%M:%S) trials=$n alive=$alive health=$health"
  if [ "$n" -ge 10 ]; then echo "COMPLETE"; break; fi
  if [ "$alive" = "0" ] && echo "$health" | grep -q '"ok"'; then
    echo "$(date +%H:%M:%S) relaunching needle harness"
    ssh -o BatchMode=yes -o ConnectTimeout=8 $RIG "cd ~/tinygrad-metal/eval && nohup python3 needle_eval.py --tag ${TAG} --trials 10 >> logs_needle_${TAG}.log 2>&1 &" 2>/dev/null
    sleep 30
  fi
  sleep 40
done
