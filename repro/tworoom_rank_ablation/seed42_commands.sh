#!/usr/bin/env bash
# Concrete command template for the fixed seed42 TwoRoom rank-only ablation.
# These commands are intentionally explicit so another machine can copy them
# directly without reconstructing run names by hand.

set -euo pipefail

cd /data/rxm210041/le-wm-custom

##############################
# pct05
##############################

bash scripts/run_tworoom_rank_ablation.sh warmup pct05 0 10
bash scripts/run_tworoom_rank_ablation.sh fit-pca pct05 tworoom_rank_seed42_pct05_warmup
bash scripts/run_tworoom_rank_ablation.sh make-random pct05

bash scripts/run_tworoom_rank_ablation.sh branch pct05 full 0 tworoom_rank_seed42_pct05_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct05 pca-r4 1 tworoom_rank_seed42_pct05_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct05 pca-r8 2 tworoom_rank_seed42_pct05_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct05 pca-r16 3 tworoom_rank_seed42_pct05_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct05 pca-r32 0 tworoom_rank_seed42_pct05_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct05 random-r16 1 tworoom_rank_seed42_pct05_warmup 20

##############################
# pct10
##############################

bash scripts/run_tworoom_rank_ablation.sh warmup pct10 0 10
bash scripts/run_tworoom_rank_ablation.sh fit-pca pct10 tworoom_rank_seed42_pct10_warmup
bash scripts/run_tworoom_rank_ablation.sh make-random pct10

bash scripts/run_tworoom_rank_ablation.sh branch pct10 full 0 tworoom_rank_seed42_pct10_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct10 pca-r4 1 tworoom_rank_seed42_pct10_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct10 pca-r8 2 tworoom_rank_seed42_pct10_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct10 pca-r16 3 tworoom_rank_seed42_pct10_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct10 pca-r32 0 tworoom_rank_seed42_pct10_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct10 random-r16 1 tworoom_rank_seed42_pct10_warmup 20

##############################
# pct15
##############################

bash scripts/run_tworoom_rank_ablation.sh warmup pct15 0 10
bash scripts/run_tworoom_rank_ablation.sh fit-pca pct15 tworoom_rank_seed42_pct15_warmup
bash scripts/run_tworoom_rank_ablation.sh make-random pct15

bash scripts/run_tworoom_rank_ablation.sh branch pct15 full 0 tworoom_rank_seed42_pct15_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct15 pca-r4 1 tworoom_rank_seed42_pct15_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct15 pca-r8 2 tworoom_rank_seed42_pct15_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct15 pca-r16 3 tworoom_rank_seed42_pct15_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct15 pca-r32 0 tworoom_rank_seed42_pct15_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct15 random-r16 1 tworoom_rank_seed42_pct15_warmup 20

##############################
# pct20
##############################

bash scripts/run_tworoom_rank_ablation.sh warmup pct20 0 10
bash scripts/run_tworoom_rank_ablation.sh fit-pca pct20 tworoom_rank_seed42_pct20_warmup
bash scripts/run_tworoom_rank_ablation.sh make-random pct20

bash scripts/run_tworoom_rank_ablation.sh branch pct20 full 0 tworoom_rank_seed42_pct20_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct20 pca-r4 1 tworoom_rank_seed42_pct20_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct20 pca-r8 2 tworoom_rank_seed42_pct20_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct20 pca-r16 3 tworoom_rank_seed42_pct20_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct20 pca-r32 0 tworoom_rank_seed42_pct20_warmup 20
bash scripts/run_tworoom_rank_ablation.sh branch pct20 random-r16 1 tworoom_rank_seed42_pct20_warmup 20
