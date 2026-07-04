#!/usr/bin/env bash
# Queue runner v2 — coordinated GPU usage: ONLY schedule on my allowed pool,
# NEVER touch 4-7 (reserved for the other user per coordination).
# Runs each queued apple-ablation variant when a pool GPU frees up.
set -u
cd /data/home/liangheng/DexTrack/wuji-mjlab || exit 1
LOG=/tmp/wuji_ap_queue2.log
CKROOT=logs/rsl_rl/wuji_tracking

# --- MY allowed GPU pool (coordination: I keep 0-3; 4-7 are theirs). GPU4 holds
# TOPO now but is NOT in the pool here — this runner won't preempt it. Pool = the
# cards I may launch queued runs on once they're free. GPU0/2 are root's; usable
# pool for queued work = 1 and 3, but 3 runs distill. So effectively runs wait for
# 1 (R2) to finish. Keep it simple: whitelist, and only launch on a pool GPU that
# is actually idle (no compute process, <2GB used).
# Pool = my 0-3 zone. gpu_idle allows root-shared cards but refuses to stack on any
# colleague's process, so if lingxiao (or anyone) grabs a 0-3 card it's skipped.
# GPU4=TOPO apple, GPU5-7 given to colleagues — all outside the pool.
POOL=( 0 1 2 3 )

# --- Queue: the 3 I just stopped for coordination + the 4 originally pending.
# Order: never-run first, then R3, then the other ran-and-stopped ones. MassCur is
# already running (skipped via "already running, advance" if it ever appears).
QUEUE=( RSI ET Noise Friction R3 R123 R1 R2 )
QI=0

say() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }

gpu_idle() {  # launchable if >=30GB free, no MY proc (avoid double-launch), and no
  # colleague proc (only root sharing tolerated — never stack on another user).
  local g=$1 free mine foreign
  free=$(nvidia-smi -i "$g" --query-gpu=memory.free --format=csv,noheader 2>/dev/null | tr -dc '0-9')
  mine=0; foreign=0
  for p in $(nvidia-smi -i "$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -dc '0-9\n'); do
    u=$(ps -o user= -p "$p" 2>/dev/null)
    [ "$u" = liangheng ] && mine=$((mine+1))
    [ "$u" != liangheng ] && [ "$u" != root ] && [ -n "$u" ] && foreign=$((foreign+1))
  done
  [ "${free:-0}" -ge 30000 ] && [ "$mine" -eq 0 ] && [ "$foreign" -eq 0 ]
}

clean_ckpts() {  # task-suffix: keep newest model_*.pt of the newest matching run dir
  local d; d=$(ls -dt ${CKROOT}/*_${1}_Env8000 2>/dev/null | head -1)
  [ -z "$d" ] && return
  local keep; keep=$(ls -t "$d"/model_*.pt 2>/dev/null | head -1 | xargs -r basename)
  [ -z "$keep" ] && return
  find "$d" -maxdepth 1 -name 'model_*.pt' ! -name "$keep" -delete 2>/dev/null
  say "cleaned $d kept $keep"
}

running() {  # task-suffix -> python PID or empty
  pgrep -f "run-name Tracking_AppleMulti_CGSmooth_Contact_${1}_Env8000" 2>/dev/null | head -1
}

launch() {  # task-suffix gpu
  local task=$1 gpu=$2 avail
  avail=$(df --output=avail -BG /data | tail -1 | tr -dc '0-9')
  if [ "${avail:-0}" -lt 200 ]; then say "DISK ${avail}G<200 skip $task"; return 1; fi
  say "launch $task on gpu$gpu (disk ${avail}G)"
  CUDA_VISIBLE_DEVICES=$gpu setsid pixi run train \
    --task WujiHand_Tracking_AppleMulti_CGSmooth_Contact_$task \
    --env.scene.num-envs 8000 \
    --agent.run-name Tracking_AppleMulti_CGSmooth_Contact_${task}_Env8000 \
    > /tmp/wuji_ap_$(echo "$task" | tr 'A-Z' 'a-z').log 2>&1 < /dev/null &
  return 0
}

say "=== queue2 start (pid $$) pool=[${POOL[*]}] queue=[${QUEUE[*]}] ==="
while [ $QI -lt ${#QUEUE[@]} ]; do
  # already running? skip ahead (in case of restart)
  t=${QUEUE[$QI]}
  if [ -n "$(running "$t")" ]; then say "$t already running, advance"; QI=$((QI+1)); continue; fi
  # find an idle pool GPU
  placed=0
  for g in "${POOL[@]}"; do
    if gpu_idle "$g"; then
      clean_ckpts "$t"
      if launch "$t" "$g"; then
        sleep 120  # warp compile
        say "started $t on gpu$g pid=$(running "$t" || echo NONE)"
        QI=$((QI+1)); placed=1
      fi
      break
    fi
  done
  [ "$placed" -eq 0 ] && sleep 180  # no idle pool gpu, wait
done
say "=== queue2 done: all ${#QUEUE[@]} launched -> exit ==="
