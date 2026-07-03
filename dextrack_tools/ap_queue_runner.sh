#!/bin/bash
# Apple-ablation queue runner. Watches the 5 first-wave trainings by their unique
# run-name; when one exits (done or crash), cleans that run's intermediate ckpts,
# checks disk, and launches the next queued variant on the freed GPU. Survives
# session exit (start via setsid). Log: /tmp/wuji_ap_queue_runner.log
set -u
cd /data/home/liangheng/DexTrack/wuji-mjlab || exit 1
LOG=/tmp/wuji_ap_queue_runner.log
CKROOT=logs/rsl_rl/wuji_tracking
say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

pidof_run() {  # task-suffix -> python train PID (unique run-name), or empty
  pgrep -f "run-name Tracking_AppleMulti_CGSmooth_Contact_${1}_Env8000" 2>/dev/null | head -1
}
clean_ckpts() {  # task-suffix: keep newest model_*.pt, delete rest
  local d=$(ls -dt ${CKROOT}/*_${1}_Env8000 2>/dev/null | head -1)
  [ -z "$d" ] && { say "clean: no dir for $1"; return; }
  local keep=$(ls -t "$d"/model_*.pt 2>/dev/null | head -1 | xargs -r basename)
  [ -z "$keep" ] && return
  find "$d" -maxdepth 1 -name 'model_*.pt' ! -name "$keep" -delete 2>/dev/null
  say "cleaned $d, kept $keep"
}
launch() {  # task-suffix gpu
  local task=$1 gpu=$2
  local avail=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
  if [ "${avail:-0}" -lt 200 ]; then say "DISK ${avail}G<200 skip $task"; return 1; fi
  say "launch $task on gpu$gpu (disk ${avail}G)"
  CUDA_VISIBLE_DEVICES=$gpu setsid pixi run train \
    --task WujiHand_Tracking_AppleMulti_CGSmooth_Contact_$task \
    --env.scene.num-envs 8000 \
    --agent.run-name Tracking_AppleMulti_CGSmooth_Contact_${task}_Env8000 \
    > /tmp/wuji_ap_$(echo "$task" | tr 'A-Z' 'a-z').log 2>&1 < /dev/null &
}

# Queue of remaining variants (in order).
QUEUE=( RSI ET Noise Friction )
QI=0

say "=== queue runner start (pid $$) ==="
sleep 150   # let first wave finish warp-compile + enter iters

# slot: task-suffix -> gpu ; resolve live PIDs
declare -A GPU=( [MassCur]=5 [R123]=6 [R1]=0 [R2]=1 [R3]=7 )
declare -A PID
for t in "${!GPU[@]}"; do PID[$t]=$(pidof_run "$t"); say "slot $t gpu${GPU[$t]} pid=${PID[$t]:-NONE}"; done

while true; do
  for t in "${!PID[@]}"; do
    p=${PID[$t]}; [ -z "$p" ] && continue
    if ! kill -0 "$p" 2>/dev/null; then
      g=${GPU[$t]}; say "$t (gpu$g pid$p) EXITED"
      clean_ckpts "$t"
      unset 'PID[$t]'
      if [ $QI -lt ${#QUEUE[@]} ]; then
        nt=${QUEUE[$QI]}; QI=$((QI+1))
        launch "$nt" "$g" && { sleep 100; GPU[$nt]=$g; PID[$nt]=$(pidof_run "$nt"); say "queued $nt gpu$g pid=${PID[$nt]:-NONE}"; }
      fi
    fi
  done
  live=0; for p in "${PID[@]}"; do [ -n "$p" ] && kill -0 "$p" 2>/dev/null && live=$((live+1)); done
  [ "$live" -eq 0 ] && [ $QI -ge ${#QUEUE[@]} ] && { say "all done -> exit"; break; }
  sleep 120
done
