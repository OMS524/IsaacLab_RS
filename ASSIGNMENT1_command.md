# 실습 과제 1 — 제출용 평가 명령어

대상: `ASSIGNMENT1.md`의 M1~M5 모델.  
평가 환경: **사각 격자형 / 불규칙 잔요철**.

## 1. 실행 준비

과제용 사용자 정의 환경이 포함된 `IsaacLab_RS`와 학습 체크포인트가 준비된 상태에서 실행한다. 아래 명령은 저장소 루트를 기준으로 한다. 다른 PC에서는 `cd` 경로를 해당 저장소 위치로 바꾼다.

```bash
conda activate lerobot-arena
cd ~/IsaacLab_RS
```

## 2. 모델과 체크포인트

| 모델 | 학습 방법 | 체크포인트: 저장소 루트 기준 |
|---|---|---|
| M1 | 평지 베이스 | `logs/rsl_rl/ant/2026-09-17_13-21-32_ant_baseline/model_999.pt` |
| M2 | 원본 보상 · 처음부터 DR | `logs/rsl_rl/ant/2026-10-06_02-01-31_scratch_rough_blocks_original/model_999.pt` |
| M3 | 추가 보상 · 처음부터 DR | `logs/rsl_rl/ant/2026-10-06_02-21-55_scratch_rough_blocks_light/model_999.pt` |
| M4 | 원본 보상 · DR 파인튜닝 | `logs/rsl_rl/ant/2026-10-05_21-47-43_baseline_ft_rough_blocks_original/model_1998.pt` |
| M5 | 추가 보상 · DR 파인튜닝 | `logs/rsl_rl/ant/2026-10-05_23-50-19_baseline_ft_rough_blocks_light/model_1998.pt` |

위 경로의 `.pt` 파일과 사용자 정의 환경 코드가 평가할 PC에도 있어야 한다. 체크포인트를 다른 위치에 배치한 경우 각 명령의 `--checkpoint` 경로만 실제 위치에 맞게 변경한다.

## 3. 사각 격자형 평가

- task: `Isaac-Ant-DR-Eval-Blocks-v0`
- 칸 너비 **0.7 m**, 높이 설정 **±6 cm**, 지형 생성 seed **2403**.
- 학습 때의 칸 너비 0.6 m와 다른 규격·배치로 평가한다.
- 아래 명령에서 평가할 모델의 블록을 선택해 실행한다. 다섯 모델을 비교하려면 모두 실행한다.

### M1. 평지 베이스

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-Eval-Blocks-v0 \
  --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant/2026-09-17_13-21-32_ant_baseline/model_999.pt \
  env.scene.terrain.terrain_generator.seed=2403 \
  env.scene.terrain.terrain_generator.sub_terrains.blocks.grid_width=0.7 \
  'env.scene.terrain.terrain_generator.sub_terrains.blocks.grid_height_range=[0.06,0.06]'
```

### M2. 원본 보상 · 처음부터 DR

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-Eval-Blocks-v0 \
  --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant/2026-10-06_02-01-31_scratch_rough_blocks_original/model_999.pt \
  env.scene.terrain.terrain_generator.seed=2403 \
  env.scene.terrain.terrain_generator.sub_terrains.blocks.grid_width=0.7 \
  'env.scene.terrain.terrain_generator.sub_terrains.blocks.grid_height_range=[0.06,0.06]'
```

### M3. 추가 보상 · 처음부터 DR

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-Eval-Blocks-v0 \
  --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant/2026-10-06_02-21-55_scratch_rough_blocks_light/model_999.pt \
  env.scene.terrain.terrain_generator.seed=2403 \
  env.scene.terrain.terrain_generator.sub_terrains.blocks.grid_width=0.7 \
  'env.scene.terrain.terrain_generator.sub_terrains.blocks.grid_height_range=[0.06,0.06]'
```

### M4. 원본 보상 · DR 파인튜닝

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-Eval-Blocks-v0 \
  --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant/2026-10-05_21-47-43_baseline_ft_rough_blocks_original/model_1998.pt \
  env.scene.terrain.terrain_generator.seed=2403 \
  env.scene.terrain.terrain_generator.sub_terrains.blocks.grid_width=0.7 \
  'env.scene.terrain.terrain_generator.sub_terrains.blocks.grid_height_range=[0.06,0.06]'
```

### M5. 추가 보상 · DR 파인튜닝

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-Eval-Blocks-v0 \
  --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant/2026-10-05_23-50-19_baseline_ft_rough_blocks_light/model_1998.pt \
  env.scene.terrain.terrain_generator.seed=2403 \
  env.scene.terrain.terrain_generator.sub_terrains.blocks.grid_width=0.7 \
  'env.scene.terrain.terrain_generator.sub_terrains.blocks.grid_height_range=[0.06,0.06]'
```

## 4. 불규칙 잔요철 평가

- task: `Isaac-Ant-DR-v0` — 원본 보상 7개를 사용하는 환경이다.
- 평지·경사·역경사 생성 비율을 0, 잔요철 비율을 1로 지정해 **잔요철 100%**로 평가한다.
- 높이 잡음 설정 **−2.5~+2.5 cm**, 높이 선택 간격 **0.5 cm**, 공간 샘플링 간격 **0.3 m**, 지형 생성 seed **2405**.
- 학습 때의 공간 샘플링 간격 0.4 m와 다른 설정·배치로 평가한다.
- 사각 격자와 잔요철이라는 종류 자체는 DR 학습에 포함됐다. 두 평가는 **학습하지 않은 규격·배치에 대한 일반화 평가**다.

### M1. 평지 베이스

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-v0 \
  --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant/2026-09-17_13-21-32_ant_baseline/model_999.pt \
  env.scene.terrain.terrain_generator.seed=2405 \
  env.scene.terrain.terrain_generator.sub_terrains.flat.proportion=0.0 \
  env.scene.terrain.terrain_generator.sub_terrains.slope.proportion=0.0 \
  env.scene.terrain.terrain_generator.sub_terrains.inverted_slope.proportion=0.0 \
  env.scene.terrain.terrain_generator.sub_terrains.rough.proportion=1.0 \
  'env.scene.terrain.terrain_generator.sub_terrains.rough.noise_range=[-0.025,0.025]' \
  env.scene.terrain.terrain_generator.sub_terrains.rough.noise_step=0.005 \
  env.scene.terrain.terrain_generator.sub_terrains.rough.downsampled_scale=0.3
```

### M2. 원본 보상 · 처음부터 DR

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-v0 \
  --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant/2026-10-06_02-01-31_scratch_rough_blocks_original/model_999.pt \
  env.scene.terrain.terrain_generator.seed=2405 \
  env.scene.terrain.terrain_generator.sub_terrains.flat.proportion=0.0 \
  env.scene.terrain.terrain_generator.sub_terrains.slope.proportion=0.0 \
  env.scene.terrain.terrain_generator.sub_terrains.inverted_slope.proportion=0.0 \
  env.scene.terrain.terrain_generator.sub_terrains.rough.proportion=1.0 \
  'env.scene.terrain.terrain_generator.sub_terrains.rough.noise_range=[-0.025,0.025]' \
  env.scene.terrain.terrain_generator.sub_terrains.rough.noise_step=0.005 \
  env.scene.terrain.terrain_generator.sub_terrains.rough.downsampled_scale=0.3
```

### M3. 추가 보상 · 처음부터 DR

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-v0 \
  --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant/2026-10-06_02-21-55_scratch_rough_blocks_light/model_999.pt \
  env.scene.terrain.terrain_generator.seed=2405 \
  env.scene.terrain.terrain_generator.sub_terrains.flat.proportion=0.0 \
  env.scene.terrain.terrain_generator.sub_terrains.slope.proportion=0.0 \
  env.scene.terrain.terrain_generator.sub_terrains.inverted_slope.proportion=0.0 \
  env.scene.terrain.terrain_generator.sub_terrains.rough.proportion=1.0 \
  'env.scene.terrain.terrain_generator.sub_terrains.rough.noise_range=[-0.025,0.025]' \
  env.scene.terrain.terrain_generator.sub_terrains.rough.noise_step=0.005 \
  env.scene.terrain.terrain_generator.sub_terrains.rough.downsampled_scale=0.3
```

### M4. 원본 보상 · DR 파인튜닝

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-v0 \
  --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant/2026-10-05_21-47-43_baseline_ft_rough_blocks_original/model_1998.pt \
  env.scene.terrain.terrain_generator.seed=2405 \
  env.scene.terrain.terrain_generator.sub_terrains.flat.proportion=0.0 \
  env.scene.terrain.terrain_generator.sub_terrains.slope.proportion=0.0 \
  env.scene.terrain.terrain_generator.sub_terrains.inverted_slope.proportion=0.0 \
  env.scene.terrain.terrain_generator.sub_terrains.rough.proportion=1.0 \
  'env.scene.terrain.terrain_generator.sub_terrains.rough.noise_range=[-0.025,0.025]' \
  env.scene.terrain.terrain_generator.sub_terrains.rough.noise_step=0.005 \
  env.scene.terrain.terrain_generator.sub_terrains.rough.downsampled_scale=0.3
```

### M5. 추가 보상 · DR 파인튜닝

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-v0 \
  --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant/2026-10-05_23-50-19_baseline_ft_rough_blocks_light/model_1998.pt \
  env.scene.terrain.terrain_generator.seed=2405 \
  env.scene.terrain.terrain_generator.sub_terrains.flat.proportion=0.0 \
  env.scene.terrain.terrain_generator.sub_terrains.slope.proportion=0.0 \
  env.scene.terrain.terrain_generator.sub_terrains.inverted_slope.proportion=0.0 \
  env.scene.terrain.terrain_generator.sub_terrains.rough.proportion=1.0 \
  'env.scene.terrain.terrain_generator.sub_terrains.rough.noise_range=[-0.025,0.025]' \
  env.scene.terrain.terrain_generator.sub_terrains.rough.noise_step=0.005 \
  env.scene.terrain.terrain_generator.sub_terrains.rough.downsampled_scale=0.3
```

## 5. 결과 확인

평가가 끝나면 다음 형식으로 출력된다.

```text
[INFO] Completed first episodes: 100/100
[RESULT] Episode reward total: mean=..., std=...
[RESULT] Episode steps: mean=..., std=...
```

과제의 평가 리워드는 `Episode reward total`의 mean/std다. `100/100`은 첫 에피소드 통계 수집 완료를 뜻하며 보행 성공률 100%를 뜻하지 않는다.
