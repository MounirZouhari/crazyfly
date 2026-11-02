#!/bin/bash
# Every Batch 2 analysis, on the results copied home by cluster/pull_results.sh.
# From the repository root:
#
#   bash code/src/tdmpc2_training/analyze_all.sh
#
# Batches and tests without results yet are skipped, so this can be run while
# jobs are still going. Each analysis writes its own summary.md and figures:
#   <batch>/analysis/             analyze_batch2.py (final tests, disturbances)
#   outputs/batch2/*_summary/     snapshots, model error, planner variants,
#                                 horizon, smoothing, precision
#   outputs/batch2/flight_quality/ time to stabilise, command jitter, worst flights
#   outputs/batch2/start_outcomes.png
set -u
A=code/src/tdmpc2_training
O=outputs/batch2
PY=.venv/bin/python
export JAX_PLATFORMS=cpu   # no GPU needed
R1=$O/random_start_1M C1=$O/fixed_start_1M D=$O/random_start_small_1M H2=$O/randomized_small_1M
SH=$O/sharp_small_1M
B1=outputs/batch1_standard_1M B1S=outputs/batch1_small_narval

# with FOLDER BATCH...: the batches in which at least one seed has FOLDER/summary.csv
with() { local folder=$1; shift; for b in "$@"; do ls $b/seed_*/$folder/summary.csv > /dev/null 2>&1 && echo $b; done; }
run() { echo; echo "=== ${*#$A/}"; $PY "$@" 2>&1 | grep -v -E "^WARNING|absl|Failed to import|RuntimeWarning|var = "; }

FINISHED=$(with evaluation $R1 $C1 $D $H2 $SH)
has() { echo "$FINISHED" | grep -qx "$1"; }
others() { for b in "$@"; do has $b && echo $b; done; }

has $R1 && run $A/analyze_batch2.py $R1 --compare $(others $C1 $D $H2) $B1
has $C1 && run $A/analyze_batch2.py $C1 --compare $B1
has $D && run $A/analyze_batch2.py $D --compare $(others $H2 $SH) $B1S
has $H2 && has $D && run $A/analyze_batch2.py $H2 --compare $D
has $SH && has $D && run $A/analyze_batch2.py $SH --compare $D

SNAP=$(with snapshot_evaluation $R1 $C1 $D $H2 $SH)
[ -n "$SNAP" ] && run $A/analyze_snapshots.py $SNAP --out $O/snapshot_summary
SNAP=$(with snapshot_evaluation_policy_only $R1 $C1 $D $H2 $SH)
[ -n "$SNAP" ] && run $A/analyze_snapshots.py $SNAP --folder snapshot_evaluation_policy_only \
  --out $O/snapshot_summary_policy_only
ME=$(with model_error $R1 $C1 $D $H2)
[ -n "$ME" ] && run $A/analyze_model_error.py $ME $B1 --out $O/model_error_summary
PV=$(with planner_variants $R1 $C1 $D $H2)
[ -n "$PV" ] && run $A/analyze_planner_variants.py $PV $B1 --out $O/planner_variants_summary
HZ=$(with horizon_test $R1 $C1 $D $H2)
[ -n "$HZ" ] && run $A/analyze_horizon.py $HZ --out $O/horizon_summary
SM=$(with smoothing_test $R1 $C1 $D $H2)
[ -n "$SM" ] && run $A/analyze_smoothing.py $SM --out $O/smoothing_summary
[ -n "$FINISHED" ] && run $A/analyze_precision.py $FINISHED $B1 --out $O/precision_summary
[ -n "$FINISHED" ] && run $A/analyze_flight_quality.py $FINISHED $B1 --out $O/flight_quality
[ -n "$FINISHED" ] && run $A/analyze_start_outcomes.py $FINISHED $B1 --out $O/start_outcomes.png
echo
echo "Finished batches: $(echo $FINISHED | tr '\n' ' ')"
