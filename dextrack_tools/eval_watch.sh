#!/usr/bin/env bash
# Watch until an eval milestone hits (or a run crashes), then exit so the caller
# is re-invoked to alert the user. Poll every 5 min.
#   MassCur (apple ablation) -> eval point ~5000 iter
#   Gen3obj FPOS/TOPO arms   -> A/B eval point, both >=2500 iter
set -u
iter() { grep -E "Learning iteration" "$1" 2>/dev/null | tail -1 | grep -oE "[0-9]+" | head -1; }
alive() { pgrep -f "$1" >/dev/null 2>&1; }

MC_LOG=/tmp/wuji_ap_masscur.log
FP_LOG=/tmp/wuji_gen3obj_fpos.log
TP_LOG=/tmp/wuji_gen3obj_topo.log

while true; do
  mc=$(iter "$MC_LOG"); fp=$(iter "$FP_LOG"); tp=$(iter "$TP_LOG")
  # crash checks (a run that had iters but whose process died)
  alive "MassCur_Env8000"            || { echo "CRASH MassCur (last iter ${mc:-?})"; exit 0; }
  alive "Gen3obj_coef0_FPOS_Env8000" || { echo "CRASH Gen3obj-FPOS (last iter ${fp:-?})"; exit 0; }
  alive "Gen3obj_coef0_TOPO_Env8000" || { echo "CRASH Gen3obj-TOPO (last iter ${tp:-?})"; exit 0; }
  # milestones
  if [ "${mc:-0}" -ge 5000 ]; then echo "READY MassCur iter=$mc (apple max_z eval)"; exit 0; fi
  if [ "${fp:-0}" -ge 2500 ] && [ "${tp:-0}" -ge 2500 ]; then
    echo "READY Gen3obj A/B fpos=$fp topo=$tp (3obj max_z eval)"; exit 0; fi
  sleep 300
done
