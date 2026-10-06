#!/usr/bin/env bash
set -euo pipefail
cd /home/oms/robotics_simulation/IsaacLab_RS

eval_dir="$PWD/logs/ant_eval_unseen_video_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$eval_dir"
names=(01_baseline 02_dr_original 03_dr_light 04_finetune_original 05_finetune_light)
checkpoints=(
  "logs/rsl_rl/ant/2026-09-17_13-21-32_ant_baseline/model_999.pt"
  "logs/rsl_rl/ant/2026-10-06_02-01-31_scratch_rough_blocks_original/model_999.pt"
  "logs/rsl_rl/ant/2026-10-06_02-21-55_scratch_rough_blocks_light/model_999.pt"
  "logs/rsl_rl/ant/2026-10-05_21-47-43_baseline_ft_rough_blocks_original/model_1998.pt"
  "logs/rsl_rl/ant/2026-10-05_23-50-19_baseline_ft_rough_blocks_light/model_1998.pt"
)

for i in "${!checkpoints[@]}"; do
  for terrain in grid waves; do
    run_dir="$eval_dir/${names[$i]}/$terrain"
    mkdir -p "$run_dir"
    cp "${checkpoints[$i]}" "$run_dir/model.pt"
    if [[ "$terrain" == grid ]]; then
      terrain_args=(
        --task Isaac-Ant-DR-Eval-Blocks-v0
        env.scene.terrain.terrain_generator.seed=2403
        env.scene.terrain.terrain_generator.sub_terrains.blocks.grid_width=0.7
        'env.scene.terrain.terrain_generator.sub_terrains.blocks.grid_height_range=[0.06,0.06]'
      )
    else
      terrain_args=(
        --task Isaac-Ant-DR-Eval-v0
        env.scene.terrain.terrain_generator.seed=2404
      )
    fi
    echo "평가 및 녹화: ${names[$i]} / ${terrain}"
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
      --seed 24 --num_envs 100 \
      --checkpoint "$run_dir/model.pt" \
      --video --video_length 960 --headless \
      "${terrain_args[@]}" \
      2>&1 | tee "$run_dir/evaluation.log"
  done
done

echo "평가·영상 저장 위치: $eval_dir"
rg '\[RESULT\]' "$eval_dir" -g evaluation.log
