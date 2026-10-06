# Ant 지형·마찰 랜덤화와 외란 비교

원본 `Isaac-Ant-v0`, 로봇 USD, 학습·평가 스크립트를 변경하지 않는 별도 환경입니다.
기존 DR/DR-Push 두 학습군은 **외란 이벤트 유무만** 다릅니다. 원본 보상 7개와 가중치,
관측 60차원, 행동 8차원, PPO 설정, 에피소드 길이 16초를 유지합니다.
DR-Stable 학습군은 외란 없는 DR에 수직 속도와 롤·피치 각속도 벌점만 더합니다.
DR-Forward 학습군은 Stable에 몸통 기준 좌우 속도와 목표 방향 오차 벌점을 더합니다.
DR-Forward-Light는 Forward의 좌우 속도 벌점 가중치만 -0.25에서 -0.05로 낮춥니다.
모든 모델은 원본 보상 7개를 쓰는 공통 Eval 환경에서 평가합니다.

## 원본 기본 동작 복원 (2026-10-05)

요청한 지형·마찰·외란 비교·보상·파인튜닝 설정 외의 임의 변경을 제거했습니다.

- 시작 XY는 원본 Ant의 5 m 격자입니다. 100개는 10×10, 4096개는 64×64이며,
  같은 위치에 중복 배치하거나 리셋마다 18개 구역 중심 중 하나로 옮기지 않습니다.
- 카메라, `clone_in_fabric=True`, 물리 복제·충돌 필터, 지면 재질의 `average` 마찰 결합을
  원본에서 상속합니다. 별도 지형 높이 색칠도 제거했습니다.
- 물리 초기화가 첫 리셋보다 먼저 실행되므로, 그때만 로봇을 지형 최고점보다 원본 시작 높이만큼
  위에 생성합니다. 이 보정이 없으면 4096개 경사 혼합 환경에서 초기 지형 관통과 함께
  시뮬레이터 중단이 재현됐습니다. 로봇 모델·`init_state`·실제 에피소드 시작 높이는 유지합니다.
- 추가했던 `ground_probe` 센서를 제거했습니다. 지형 메시에서 높이를 직접 조회해,
  시작 높이(지면 + 0.5 m)와 넘어짐 기준(지면 대비 0.31 m)만 보정합니다.
- 지형 밖에서 ray가 빗나갔다는 이유만으로 즉시 종료하지 않습니다. 이 경우 원본의
  월드 높이 0.31 m 판정을 적용합니다. 평지 단계는 원본 종료 함수를 그대로 씁니다.
- 5 m 간격의 시작점들이 지형에 들어가도록 필요한 경우 구역 수만 늘립니다.
  100개에서는 기존 160×160 m, 4096개에서는 320×320 m입니다.
  지형 종류·높이·구역 크기·해상도·생성 확률은 유지합니다. 지형은 여전히 유한합니다.
- `play_one_episode.py`의 로봇 추적 카메라는 원래 해당 스크립트에 있던 설정이며 수정하지 않았습니다.

**이 문서 아래의 기존 학습·평가 수치는 복원 전 조건에서 얻은 기록입니다.**
기존 체크포인트와 로그는 그대로 보존하며 같은 명령으로 불러올 수 있지만,
복원 후 평가 수치는 시작 위치·마찰 결합·경계 종료 처리 변경 때문에 달라질 수 있습니다.
복원 전/후 수치를 같은 평가 조건으로 간주하면 안 됩니다.

복원 후 검증: 원본/보상 변경 태스크의 설정 비교, 원본 보상 7개 유지, 중복 없는 격자,
리셋 후 위치 유지, 시작 높이, 유한한 관측·보상, 외란 적용 범위를 검사했습니다.
실제 GPU에서 블록 평가 100개, 물결+외란 평가 100개, Light 파인튜닝 4096개,
Light 평지 단계 16개를 각각 120스텝 실행해 통과했습니다. 재학습은 실행하지 않았습니다.
원본 코드·기존 학습/평가 스크립트·기준 및 최종 체크포인트 75개 파일의 SHA-256도 보존했습니다.
검사 로그: `/tmp/ant_minimal_restore_final4096.log`,
`/tmp/ant_minimal_restore_final_blocks100.log`, `/tmp/ant_minimal_restore_final_flat16.log`,
`/tmp/ant_minimal_restore_final_waves_push100.log`.

## 환경 이름

| task | 용도 | 지형 | 접촉 마찰 | 외란 |
|---|---|---|---|---|
| `Isaac-Ant-v0` | 기존 기준 환경 | 원래 평지 | 원래 설정 | 없음 |
| `Isaac-Ant-DR-v0` | 학습 A | 평지·경사·요철 | 랜덤 | 없음 |
| `Isaac-Ant-DR-Stable-v0` | 상하·회전 안정성 보상 학습 | A와 같은 설정 | A와 같은 설정 | 없음 |
| `Isaac-Ant-DR-Forward-v0` | Stable + 정면 이동 보상 학습 | A와 같은 설정 | A와 같은 설정 | 없음 |
| `Isaac-Ant-DR-Forward-Light-v0` | Forward의 좌우 속도 벌점 완화 | A와 같은 설정 | A와 같은 설정 | 없음 |
| `Isaac-Ant-DR-Push-v0` | 학습 B | A와 같은 설정 | A와 같은 설정 | 있음 |
| `Isaac-Ant-DR-Eval-v0` | 공통 평가 | 학습에 없는 물결 지형 | A/B와 같은 범위 | 없음 |
| `Isaac-Ant-DR-Eval-Push-v0` | 공통 외란 평가 | 같은 물결 지형 | 같은 범위 | 있음 |

평가 환경에서는 학습하지 않습니다. 두 모델을 각각 **같은 평가 task**에서 평가합니다.
A를 외란 없는 환경에서만, B를 외란 있는 환경에서만 평가하면 공정한 비교가 아닙니다.
평가 수치가 나오기 전에는 어느 모델이 더 좋다고 판단하지 않습니다.

## 랜덤화 조건과 원본 대비 차이

- 지형: 20 m × 20 m 구역을 최소 8 × 8개 생성하고, 시작 격자가 클 때만 확장합니다.
  생성 시 종류 비율은 평지 30%,
  경사 15%, 역경사 15%, 작은 요철 40%입니다. 이는 생성 확률이며 실제 구역 수를
  정확히 그 비율로 강제하지 않습니다. 리셋 시 같은 격자 위치로 돌아갑니다.
- 경사 설정값: `slope_range=(0.02, 0.10)`. 생성기의 경사 파라미터이며 도(degree)가 아닙니다.
  경사 구역에는 중앙 2 m 플랫폼이 있으며, 시작점은 원본의 격자를 따릅니다.
- 요철: `noise_range=(-0.025, 0.025)` m, 높이 간격 0.005 m,
  공간 샘플링 0.4 m, 메시 해상도 0.2 m입니다.
- 시작 위치: 원본과 동일한 5 m 간격 격자입니다. 해당 XY의 지면 높이에
  원본 시작 높이 0.5 m를 더하며, 자세·속도·관절 초기화는 원본을 유지합니다.
- 마찰: Ant별 정지 마찰 `[0.5, 1.25]`, 동적 마찰 `[0.4, 1.0]`, 반발계수 0.
  동적 마찰은 정지 마찰 이하로 제한합니다. 64개 재질 후보에서 Ant마다 하나를 선택해
  해당 Ant의 모든 충돌 형상에 같은 재질을 적용합니다. 재질은 환경 생성 시 정하고
  에피소드 리셋 때마다 다시 만들지 않습니다.
- 공유 바닥의 마찰은 원본과 같은 1.0, 결합 모드는 `average`입니다.
  위 범위는 로봇 재질 값이며, 접촉 시에는 바닥과의 결합 규칙이 적용됩니다.
  **로봇 쪽 접촉 재질을 바꿔 접촉 쌍의 마찰을 랜덤화**합니다. 바닥 구역마다
  다른 마찰을 칠한 구현은 아닙니다. 로봇 형상·질량·관성·USD 파일은 바꾸지 않습니다.
- 외란: 각 환경에서 3~6초 간격으로 수평 속도에 x/y 각각 `[-0.3, 0.3]` m/s를
  더합니다. 수직 속도와 각속도는 바꾸지 않습니다. 지속적인 힘(N)이 아닌 속도 교란입니다.
- 넘어짐: 원본의 절대 높이 0.31 m 대신 **몸통과 바로 아래 지면의 높이 차 0.31 m**로
  판정합니다. 지형에서 높이를 직접 조회하며 센서·정책 입력은 추가하지 않습니다.
  지면을 찾지 못하면 원본의 월드 높이 판정으로 돌아갑니다.
- 정책의 높이 관측은 원본과 같은 세계 좌표 z입니다. 지면 높이를 정책에 제공하거나
  기존 관측을 지면 상대 높이로 바꾼 실험은 아닙니다.
- 모든 DR task가 원본 관측·행동 차원을 유지하므로 기존 Ant 체크포인트도 불러올 수 있습니다.
  Stable은 보상 두 항목, Forward는 네 항목을 추가하며, 평가 task는 원본 보상을 유지합니다.
  환경이 바뀌었으므로 기존 정책의 좋은 성능을 보장하지 않습니다.

2026-10-05에 학습 지형 생성 비중을 위와 같이 조정했습니다. 이전에 학습한 체크포인트는
기존 비중(평지 40%, 경사·역경사·요철 각 20%)으로 학습된 모델이며, 새 비중은 이후 학습에 적용됩니다.

## 학습

`lerobot-arena` 환경에서 `IsaacLab_RS` 디렉토리를 작업 위치로 사용합니다.
두 학습은 GPU 자원 경쟁을 피하도록 순서대로 실행하는 것이 좋습니다.

```bash
conda activate lerobot-arena
cd /home/oms/robotics_simulation/IsaacLab_RS

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-DR-v0 --num_envs 4096 --seed 42 \
  --max_iterations 1000 --run_name terrain_friction --headless

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-DR-Push-v0 --num_envs 4096 --seed 42 \
  --max_iterations 1000 --run_name terrain_friction_push --headless
```

결과는 각각 `logs/rsl_rl/ant_dr/`와 `logs/rsl_rl/ant_dr_push/`에 저장됩니다.
동일한 환경 수·학습 횟수·seed·PPO 설정을 유지해 비교하세요.
검증 목적으로 2회만 학습한 `smoke_*` 실행은 과제 제출용 모델이 아닙니다.

## 수직 속도 + 롤·피치 각속도 벌점 실험 (2026-10-05)

`Isaac-Ant-DR-Stable-v0`는 현재 요철 비율 40%의 `Isaac-Ant-DR-v0`에서
보상만 변경한 별도 학습 task입니다. 기존 보상 7개와 가중치는 그대로 두고 다음을 추가합니다.

| 항목 | 함수 출력 | 가중치 |
|---|---|---|
| `vertical_velocity` | 월드 좌표 수직 속도 `v_z²` | -0.5 |
| `roll_pitch_velocity` | 몸통 좌표 각속도 `ω_x² + ω_y²` | -0.05 |

RewardManager가 각 항목에 가중치와 제어 시간 간격 `dt=1/60 s`를 곱합니다.
새 항목에서 dt를 중복 적용하지 않습니다. 수직 속도는 몸통 좌표 z가 아닌 월드 z를 사용하며,
각속도 벌점은 몸통 x/y축 회전만 사용하므로 요 회전을 직접 벌점으로 주지 않습니다.
두 계수는 초기 실험값으로, 보행 개선이 검증된 값은 아닙니다. TensorBoard의
`Episode_Reward/vertical_velocity`, `Episode_Reward/roll_pitch_velocity`에서 기여도를 확인합니다.

지형·마찰·외란 없음·관측·로봇·PPO·종료 조건은 DR 비교군과 같습니다.
지형은 유한하므로, 경계 밖으로 걸어 나간 뒤 추락하는 경우와 지형 위에서 넘어지는 경우는 구분해야 합니다.
기존 모델은 기존 보상으로 학습된 결과이므로 새 보상 실험은 아래 명령으로 새로 학습합니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-DR-Stable-v0 \
  --num_envs 4096 --seed 42 --max_iterations 1000 \
  --run_name rough40_stable --headless
```

저장 위치는 `logs/rsl_rl/ant_dr_stable/<실험폴더>/model_999.pt`입니다.
**과제용 평가는 학습 task인 DR-Stable 대신 아래 공통 Eval task를 사용합니다.**
이때 새 벌점 두 개는 포함되지 않고 원본 Ant 보상 7개로 평균·표준편차를 계산합니다.
관측·행동 차원이 같으므로 모델을 그대로 불러올 수 있습니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-Eval-v0 --seed 24 --num_envs 100 \
  --checkpoint "logs/rsl_rl/ant_dr_stable/<실험폴더>/model_999.pt" \
  --headless
```

영상 저장 시 `--video --video_length 960`을 추가합니다. 기존 DR 모델도 같은 Eval task와
seed/환경 수로 평가해 비교하세요. DR-Stable task로 재생하면 학습 벌점을 포함한 점수가
출력되므로 과제 보고용 원본 보상 점수로 사용하면 안 됩니다.

## 몸통 정면 이동 보상 실험 (2026-10-05)

`Isaac-Ant-DR-Forward-v0`는 Stable에 다음 두 벌점만 추가한 별도 학습 task입니다.
원본 보상 7개와 수직 속도(-0.5), 롤·피치 각속도(-0.05)를 유지하므로 총 11개 항목입니다.

| 추가 항목 | 함수 출력 | 가중치 |
|---|---|---|
| `lateral_velocity` | 몸통 좌표 좌우 속도 `v_y²` | -0.25 |
| `heading_error` | 수평 방향 오차 `1 - cos(θ)` | -1.0 |

좌우 속도는 월드 y가 아닌 **몸통 y축** 기준입니다. `θ`는 몸통 +x축을 수평면에 투영한
정면 방향과 로봇에서 목표 `(1000, 0, 0)`로 향하는 수평 방향 사이의 각도입니다.
목표 바로 위/아래 등 xy 위치가 일치하면 방향 오차 벌점은 0으로 정의합니다.
방향 오차는 0°에서 0, 90°에서 1, 180°에서 2이며, 음수 가중치와 dt를 적용합니다.
기존 `move_to_target` 보상도 유지하므로 약 37° 이내의 최대 방향 보상을 받더라도
새 벌점은 정면을 더 정확히 맞추도록 유도합니다.

지형·마찰·관측·로봇·외란 없음·PPO·종료 조건은 Stable과 같습니다.
몸통 정면과 이동 방향의 일치를 유도하는 실험이며, 교대 보행이나 접촉 패턴을 강제하지 않습니다.
계수는 첫 실험값으로 보행 개선을 보장하지 않습니다. TensorBoard의
`Episode_Reward/lateral_velocity`, `Episode_Reward/heading_error`에서 크기를 확인하세요.

최근 Stable의 2,000회 학습과 비교하려면 같은 학습량으로 새로 시작합니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-DR-Forward-v0 \
  --num_envs 4096 --seed 42 --max_iterations 2000 \
  --run_name rough40_forward --headless
```

결과는 `logs/rsl_rl/ant_dr_forward/<실험폴더>/`에 저장됩니다.
처음부터 2,000회 학습하면 최종 파일은 `model_1999.pt`이고, 중간 `model_999.pt`와도 비교할 수 있습니다.
**과제용 평가는 기존 공통 Eval task를 사용하여 원본 보상 7개로 계산합니다.**
학습 task인 Forward로 평가하면 추가 벌점이 포함되므로 과제 보고용 점수로 사용하지 않습니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-Eval-v0 --seed 24 --num_envs 100 \
  --checkpoint "logs/rsl_rl/ant_dr_forward/<실험폴더>/model_1999.pt" \
  --headless
```

## 좌우 속도 벌점 완화 실험 (2026-10-05)

`Isaac-Ant-DR-Forward-Light-v0`는 기존 Forward의 좌우 속도 벌점만 1/5로 줄입니다.
같은 몸통 기준 좌우 속도에서 해당 벌점의 크기가 80% 작아집니다.

| 항목 | 기존 Forward | Forward-Light |
|---|---:|---:|
| 월드 수직 속도 | -0.5 | -0.5 |
| 몸통 롤·피치 각속도 | -0.05 | -0.05 |
| 몸통 좌우 속도 | -0.25 | **-0.05** |
| 수평 방향 오차 | -1.0 | -1.0 |

원본 보상 7개와 나머지 설정은 같습니다. 기존 Forward task도 그대로 남아 비교할 수 있습니다.
이는 최적값을 찾았다는 뜻이 아니라, 이동 성능과 안정성의 균형을 확인하기 위한 다음 실험입니다.

최근 Forward 실행은 1,000회 학습 모델이므로 같은 학습량으로 처음부터 비교합니다.
기존 체크포인트를 불러와 재생하는 것만으로는 가중치 변경이 보행에 반영되지 않습니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-DR-Forward-Light-v0 \
  --num_envs 4096 --seed 42 --max_iterations 1000 \
  --run_name rough40_forward_light --headless
```

결과는 `logs/rsl_rl/ant_dr_forward_light/<실험폴더>/model_999.pt`에 저장됩니다.
공통 평가 task에서는 원본 보상 7개만 적용합니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-Eval-v0 --seed 24 --num_envs 100 \
  --checkpoint "logs/rsl_rl/ant_dr_forward_light/<실험폴더>/model_999.pt" \
  --headless
```

## 공통 환경에서 평가

아래 두 경로의 `<실험폴더>`를 각 학습에서 실제 생성된 폴더명으로 바꿉니다.
최종 체크포인트는 1,000회 학습 기준 `model_999.pt`입니다.

```bash
NO_PUSH_CKPT="logs/rsl_rl/ant_dr/<실험폴더>/model_999.pt"
PUSH_CKPT="logs/rsl_rl/ant_dr_push/<실험폴더>/model_999.pt"

for task in Isaac-Ant-DR-Eval-v0 Isaac-Ant-DR-Eval-Push-v0; do
  for checkpoint in "$NO_PUSH_CKPT" "$PUSH_CKPT"; do
    ./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
      --task "$task" --seed 24 --num_envs 100 \
      --checkpoint "$checkpoint" --headless
  done
done
```

터미널의 `[RESULT] Episode reward total: mean=..., std=...`를 기록합니다.
기존 기준 모델도 **같은 평가 task**와 seed·환경 수로 실행하면 추가 비교가 가능합니다.
GUI를 보려면 `--headless`를 빼고, 화면 확인용으로 환경 수를 16으로 줄일 수 있습니다.
과제 제출 수치는 반드시 seed 24, 환경 100개로 다시 평가하세요.

평가 지형은 학습에 없는 `HfWaveTerrainCfg`를 사용합니다. 진폭 설정 범위는 0.03~0.06 m,
구역당 물결 수는 4 또는 8입니다. 마찰 범위는 학습과 같아 새 지형에 대한 일반화를
비교합니다. seed만 바꾼 같은 종류의 학습 지형이 아닙니다.
학습 지형 생성 seed는 1701, 평가 지형 생성 seed는 2401이며 CLI seed와 별개입니다.
CLI seed 24는 초기 상태, 재질 배정, 외란 등 실행 난수를 제어합니다. 학습 요철의
세부 높이는 NumPy 난수도 사용하므로 재현할 때는 지형 seed와 CLI seed를 모두 맞추세요.
최종 평가 환경 결과를 보고 반복 튜닝하면 검증용 환경이 되므로, 최종 보고용 조건은
미리 고정하거나 별도의 미사용 조건을 마련하세요.

## 설정 수정 위치와 검사

- `ant_env_cfg.py`: 지형 종류·범위, 마찰 범위, 외란 간격·세기, 평가 환경, Stable/Forward 보상 가중치.
- `mdp.py`: Ant별 접촉 재질 배정, 지면 기준 넘어짐 판정, 수직·좌우 속도 및 방향 오차 벌점.
- `terrain.py`: 원본 시작 격자 유지, 지형 크기의 최소 확장, 지면 높이 조회, 첫 리셋 전 생성 높이 보정.
- `agents_cfg.py`: 원본 PPO 설정을 상속하고 로그 이름만 분리.
- `__init__.py`: 새 task 등록. 원본 등록은 건드리지 않습니다.

추가 패키지 설치 없이 기존 editable 설치에서 새 task가 자동 탐색됩니다.

```bash
./isaaclab.sh -p scripts/environments/check_ant_randomization.py --headless
./isaaclab.sh -p scripts/environments/check_ant_randomization.py \
  --task Isaac-Ant-DR-Eval-v0 --headless
./isaaclab.sh -p scripts/environments/check_ant_randomization.py \
  --task Isaac-Ant-DR-Stable-v0 --headless
./isaaclab.sh -p scripts/environments/check_ant_randomization.py \
  --task Isaac-Ant-DR-Forward-v0 --headless
./isaaclab.sh -p scripts/environments/check_ant_randomization.py \
  --task Isaac-Ant-DR-Forward-Light-v0 --headless
```

검사는 원본 설정 보존, 외란만 다른 비교군, 분리된 평가 지형, 관측 차원,
Ant별 마찰과 재현성, 지면 추적, 외란의 실제 수평 속도 변화 및 유한한 보상을 확인합니다.

## 구현 시 검증 기록 (2026-10-04)

- RTX 6000 Ada에서 16개 환경 × 120스텝 물리 검사 통과.
- 두 학습 task 각각 64개 환경, PPO 2회 학습 및 `model_1.pt` 저장 완료.
- 두 검증용 모델 × 두 공통 평가 task의 4가지 조합에서 seed 24, 환경 100개 평가 완료.
  모두 `Completed first episodes: 100/100`과 누적 보상 평균·표준편차 출력을 확인했습니다.
- 원본 Ant 설정/등록 및 기존 train/play/play_one_episode 파일의 SHA-256이 작업 전과 동일합니다.
- 검증용 모델은 학습량이 매우 적으며 이 결과를 성능 비교나 과제 제출 수치로 사용하지 않습니다.

## Stable 보상 추가 검증 기록 (2026-10-05)

- 기존 DR 대비 두 보상 항목만 다른 설정이며, 원본/Eval 보상과 PPO 설정 보존 확인.
- 월드 수직 속도 좌표계, 양·음 수직 속도 대칭성, 순수 요 회전의 벌점 제외 검사 통과.
- RTX 6000 Ada에서 16개 환경 × 120스텝 물리 검사 통과.
  두 벌점의 음수 부호, 실제 상태와의 일치, dt 1회 적용 및 유한한 보상 확인.
- `Isaac-Ant-DR-Stable-v0`에서 64개 환경, PPO 2회 학습 및 체크포인트 저장 확인.
  실행 폴더: `logs/rsl_rl/ant_dr_stable/2026-10-05_01-27-04_smoke_stable/`.
- 해당 `model_1.pt`를 공통 `Isaac-Ant-DR-Eval-v0`, seed 24, 환경 100개로 평가하여
  `Completed first episodes: 100/100` 확인. 평가의 활성 보상 항목은 원본 7개임을 확인.
- 원본 Ant 설정/등록 및 기존 train/play/play_one_episode 파일의 SHA-256 보존 확인.
- 위 결과는 실행 검증이며, 보행 개선 효과나 제출용 성능을 검증한 결과가 아닙니다.

## Forward 보상 추가 검증 기록 (2026-10-05)

- Stable 대비 좌우 속도와 방향 오차 벌점만 추가되며 기존 원본/Eval 보상과 PPO 설정 보존 확인.
- 몸통을 90° 회전한 전진의 좌우 벌점 0, 좌우 속도 부호 대칭성 확인.
- 방향 오차 0°/90°/180°, ±π 경계, 목표 방향이 +x가 아닌 경우, 목표 위치 도달 검사 통과.
- RTX 6000 Ada에서 16개 환경 × 120스텝 물리 검사 통과.
  총 11개 보상, 네 벌점의 부호와 활성화, 실제 상태와의 일치 및 dt 1회 적용 확인.
- Forward에서 64개 환경, PPO 2회 학습 및 체크포인트 저장 확인.
  실행 폴더: `logs/rsl_rl/ant_dr_forward/2026-10-05_03-17-52_smoke_forward/`.
- 해당 `model_1.pt`를 공통 Eval task, seed 24, 환경 100개로 평가하여
  원본 보상 7개와 `Completed first episodes: 100/100` 확인.
- 원본 Ant 설정/등록 및 기존 train/play/play_one_episode 파일의 SHA-256 보존 확인.
- 위 검사는 실행 검증이며, 정면 보행 개선 여부는 본 학습 후 같은 조건에서 비교해야 합니다.

## Forward-Light 가중치 조정 검증 기록 (2026-10-05)

- 기존 Forward와 좌우 속도 가중치(-0.25 → -0.05) 하나만 다른 환경 설정임을 확인.
- 새 설정 생성 후에도 기존 Forward는 -0.25이며, PPO는 로그 이름만 다름을 확인.
- 원본/Eval 보상 7개 보존, 기존 함수의 좌표계·방향 오차 경계 검사 통과.
- RTX 6000 Ada에서 Forward-Light 16개 환경 × 120스텝 검사 통과.
  활성 보상 11개, 완화된 좌우 벌점 -0.05, 실제 물리 상태와 계산 일치 확인.
- 보상 함수 구현, 원본 Ant 및 train/play_one_episode 스크립트의 SHA-256 보존 확인.
- 이번 조정은 설정 및 물리 실행까지 검증했으며, 본 학습 후의 성능 개선 여부는 아직 확인하지 않았습니다.


## 평지 사전학습 + 두 번의 파인튜닝

설정 파일: `finetune_env_cfg.py`. 원본 보상과 현재 Forward-Light 보상을 각각
평지 → 약한 DR → 기존 DR 순서로 학습합니다. 두 분기는 보상만 다르고 단계별
환경/관측/행동/로봇/PPO/학습량은 같습니다. 모든 단계에 외란은 없습니다.

| 단계 | 추가 업데이트 | 지형 생성 비중 (평지/경사 합계/요철) | 요철 높이 | 경사 파라미터 | 정지/동적 마찰 |
|---|---:|---|---|---|---|
| 1. 평지 사전학습 | 1,000 | 무한 평지 | 0 | 0 | 1.0 / 1.0 고정 |
| 2. 약한 DR 파인튜닝 | 500 | 50% / 20% / 30% | ±0.01 m | 0.01~0.04 | 0.8~1.1 / 0.7~1.0 |
| 3. 기존 DR 파인튜닝 | 500 | 30% / 30% / 40% | ±0.025 m | 0.02~0.10 | 0.5~1.25 / 0.4~1.0 |

생성 비중은 확률입니다. 동적 마찰은 정지 마찰 이하로 제한하고, DR 마찰은
환경마다 시작 시 정해 리셋 간 유지합니다. 평지는 원본 종료 판정을 사용하고,
DR에서만 지면 상대 높이로 보정합니다. 추가 센서는 없습니다.

- 원본 분기: 원본 Ant 보상 7개를 세 단계 내내 유지.
- Light 분기: 원본 7개 + 수직 속도 -0.5, 롤·피치 각속도 -0.05,
  몸통 좌우 속도 -0.05, 방향 오차 -1.0을 세 단계 내내 유지.
- 각 분기마다 평지 학습부터 새로 시작합니다. 기존 다른 보상의 체크포인트를
  가져오지 않아 사전학습 도중 보상이 바뀌지 않습니다.
- 수동 명령을 사용할 때는 한 단계가 정상 종료한 뒤 다음 단계를 실행하세요.
  아래 자동 실행기를 사용하면 이 순서를 자동 처리합니다. 단계 이름으로 직전 모델을 구분합니다.
- `--resume`은 정책/가치망과 옵티마이저 상태를 불러옵니다. 시뮬레이터 상태는
  새로 시작하며 `--max_iterations`는 그 단계에서 **추가**로 실행할 업데이트 수입니다.
  기존 RSL-RL의 adaptive 학습률 처리도 그대로 사용합니다.
- 두 분기 모두 총 2,000회이며, 처음부터 DR만 2,000회 학습한 실험과 총학습량을
  맞춘 비교가 가능합니다. 각 보상 분기 안에서 직접 DR 학습군과 비교하세요.

### 한 번 실행해서 모든 단계 자동 학습

`scripts/reinforcement_learning/rsl_rl/train_ant_finetune.py`가 기존 `train.py`를
단계별로 순차 실행합니다. `both`는 원본 보상 3단계를 끝낸 뒤 Light 보상 3단계를
실행합니다. 새 실행마다 평지 사전학습부터 시작합니다.

```bash
conda activate lerobot-arena
cd /home/oms/robotics_simulation/IsaacLab_RS

# 두 실험 모두: 원본 평지 -> 약한 DR -> 현재 DR -> Light 평지 -> 약한 DR -> 현재 DR
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_ant_finetune.py \
  --reward both --num_envs 4096 --seed 42 --iterations 1000 500 500
```

한 실험만 자동 실행하려면 다음 중 하나를 사용합니다.

```bash
# 원본 보상 3단계만
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_ant_finetune.py --reward original

# 현재 Forward-Light 보상 3단계만
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train_ant_finetune.py --reward light
```

- 기본값: 환경 4096개, seed 42, 단계별 추가 업데이트 1000/500/500, headless 실행.
- `--iterations 평지횟수 약한DR횟수 현재DR횟수`로 예산을 바꾸면 체크포인트 번호도
  자동 계산합니다. 모든 단계에서 같은 seed를 사용하며 음수 seed는 허용하지 않습니다.
- 고유 실행 ID를 붙이고, 방금 완료한 실행 폴더명과 최종 체크포인트를 정확하게 지정해
  다음 단계에 전달합니다. 다른 실행의 최신 체크포인트를 선택하지 않습니다.
- 각 자식 학습이 정상 종료하고 최종 체크포인트가 존재해야 다음 단계가 시작됩니다.
  오류나 Ctrl+C가 발생하면 이후 단계와 나머지 보상 분기도 중단합니다.
- 모델 저장 위치는 기존과 같은 `ant_ft_original`, `ant_ft_forward_light` 아래입니다.
  폴더명에는 `auto_실행ID_보상_ft_stage...`가 포함되고, 완료 시 최종 모델 경로를 출력합니다.
- 진행 상태, 단계별 체크포인트 및 이전 모델 경로는
  `logs/rsl_rl/ant_finetune_chains/실행ID.json`에 기록합니다.
- 실행 계획만 확인하려면 `--dry-run`을 추가하세요. 이 옵션은 학습이나 파일 생성을 하지 않습니다.

### 원본 보상: 세 명령어를 순서대로 실행

```bash
conda activate lerobot-arena
cd /home/oms/robotics_simulation/IsaacLab_RS

# 1. 평지 사전학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-FT-Flat-v0 \
  --num_envs 4096 --seed 42 --max_iterations 1000 \
  --run_name ft_stage1_flat --headless

# 2. 평지 모델 -> 약한 DR
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-FT-Mild-v0 \
  --num_envs 4096 --seed 42 --max_iterations 500 \
  --resume --load_run '.*_ft_stage1_flat$' --checkpoint 'model_999\.pt$' \
  --run_name ft_stage2_mild --headless

# 3. 약한 DR 모델 -> 기존 DR
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-FT-DR-v0 \
  --num_envs 4096 --seed 42 --max_iterations 500 \
  --resume --load_run '.*_ft_stage2_mild$' --checkpoint 'model_1498\.pt$' \
  --run_name ft_stage3_dr --headless
```

로그 루트: `logs/rsl_rl/ant_ft_original/`.

### 현재 Forward-Light 보상: 세 명령어를 순서대로 실행

```bash
# 1. 현재 보상으로 평지 사전학습
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-FT-Light-Flat-v0 \
  --num_envs 4096 --seed 42 --max_iterations 1000 \
  --run_name ft_stage1_flat --headless

# 2. 평지 모델 -> 약한 DR
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-FT-Light-Mild-v0 \
  --num_envs 4096 --seed 42 --max_iterations 500 \
  --resume --load_run '.*_ft_stage1_flat$' --checkpoint 'model_999\.pt$' \
  --run_name ft_stage2_mild --headless

# 3. 약한 DR 모델 -> 기존 DR
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-FT-Light-DR-v0 \
  --num_envs 4096 --seed 42 --max_iterations 500 \
  --resume --load_run '.*_ft_stage2_mild$' --checkpoint 'model_1498\.pt$' \
  --run_name ft_stage3_dr --headless
```

로그 루트: `logs/rsl_rl/ant_ft_forward_light/`. 원본 분기와 로그 루트가 달라
동일한 단계 이름을 사용해도 서로의 체크포인트를 불러오지 않습니다.
각 `--load_run` 패턴은 해당 루트에서 이름이 일치하는 가장 최신 폴더를 선택합니다.
여러 seed/실험을 진행할 때는 정확한 폴더명을 지정해 연결 관계를 고정하세요.

현재 설치된 RSL-RL은 체크포인트의 마지막 반복 번호부터 다시 번호를 매깁니다.
위 예산에서 단계별 최종 파일은 `model_999.pt` → `model_1498.pt` → `model_1997.pt`입니다.
번호가 겹치지만 각 단계는 실제로 1000/500/500회씩 업데이트합니다.
예산을 바꾸면 다음 단계의 `--checkpoint`도 해당 최종 파일에 맞춰 변경하세요.
이 명령어는 완료된 단계의 정확한 파일을 요구하므로 중간 저장본을 자동 선택하지 않습니다.

### 공통 평가: 새 물결 지형 + 원본 보상

아래 `날짜시간_ft_stage3_dr`를 각 실험의 실제 폴더명으로 바꿉니다.
두 모델 모두 `Isaac-Ant-DR-Eval-v0`, seed 24, 환경 100개를 사용합니다.
평지 및 약한 DR 체크포인트도 같은 조건으로 평가하면 적응 과정의 변화를 비교할 수 있습니다.

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-Eval-v0 --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant_ft_original/날짜시간_ft_stage3_dr/model_1997.pt

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-Eval-v0 --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant_ft_forward_light/날짜시간_ft_stage3_dr/model_1997.pt
```

영상 저장은 평가 명령에 `--video --video_length 960`을 추가합니다.
평지 보행을 먼저 눈으로 확인하려면 학습한 분기의 `Isaac-Ant-FT-Flat-v0` 또는
`Isaac-Ant-FT-Light-Flat-v0`와 해당 평지 체크포인트로 `play.py`를 실행하세요.
평지에서도 전진이 부족하다면 다음 단계로 넘어가기 전에 평지 결과를 점검하세요.


### 파인튜닝 구현 검증 기록 (2026-10-05)

- 여섯 task 등록, 같은 단계의 두 분기가 보상만 다른지, 세 단계 내 보상/PPO 유지,
  설정 객체 격리, 마지막 단계와 기존 DR 설정의 완전 일치를 검사했습니다.
- Forward-Light 평지에서 16개 환경 × 120스텝 물리 검사 통과.
  고정 마찰, 지면 감지, 관측/보상 유한성 및 네 벌점의 실제 계산을 확인했습니다.
- 두 분기 모두 기존 `train.py`로 64개 환경 × 단계당 PPO 2회 × 3단계를 실행했습니다.
  단계별 체크포인트 `model_1.pt` → `model_2.pt` → `model_3.pt`가 생성되었고,
  Adam 상태의 업데이트 횟수가 40 → 80 → 120으로 이어지는 것을 확인했습니다.
  저장된 설정에서 원본 분기의 7개/Light 분기의 11개 보상이 각 단계에 유지됩니다.
- Light의 마지막 검증용 체크포인트를 기존 `play_one_episode.py`와 공통 Eval task,
  seed 24, 환경 100개로 실행하여 원본 보상 7개 및 첫 에피소드 100/100 완료를 확인했습니다.
- 검증 로그는 `/tmp/ant_ft_flat_check.log`, `/tmp/ant_ft_smoke_*.log`,
  `/tmp/ant_ft_eval_check.log`에 있습니다. 검증용 학습 폴더 이름은 `smoke_ft_...`이며
  위 본 학습 명령의 `ft_stage...` 선택 패턴과 겹치지 않습니다.
- 원본 Ant, 기존 DR/보상 구현, train/play/play_one_episode/CLI 파일은 변경 전
  SHA-256과 일치합니다. 본 학습과 보행 성능 비교는 위 명령으로 별도 진행해야 합니다.


### 자동 실행기 검증 기록 (2026-10-05)

- 두 분기 6단계 순서, 분기 간 초기화, 정확한 직전 체크포인트 선택, 변경한 예산의
  번호 계산, 자식 프로세스 실패/체크포인트 누락 시 중단 및 dry-run 무변경 검사를 통과했습니다.
- 실제 자동 실행기에서 Light 64개 환경 × 단계당 2회 × 3단계가 개입 없이 완료되었습니다.
  실행 ID: `20261005_060406_a5beae24`. manifest의 단계 연결과 체크포인트를 확인했고
  Adam 업데이트 횟수 40 → 80 → 120으로 이어졌습니다.
- 실행 로그: `/tmp/ant_finetune_auto_check.log`. 본 학습용 1000/500/500회 실행은
  위 명령으로 시작합니다. 기존 학습/평가 및 환경 설정 파일은 보존되었습니다.


## 기존 평지 baseline에서 한 번만 파인튜닝 (평지 + 잔요철 + 사각 격자 + 경사)

설정: `baseline_finetune_env_cfg.py`.
공통 출발점은 `logs/rsl_rl/ant/2026-09-17_13-21-32_ant_baseline/model_999.pt`입니다.
두 실험 모두 이 모델을 직접 불러와 **각각 1,000회만 추가 학습**합니다.
Light 실험이 원본 보상 파인튜닝 결과를 이어받는 구조가 아닙니다.

| 학습 지형 | 생성 확률 | 설정 |
|---|---:|---|
| 평지 | 10% | `MeshPlaneTerrainCfg` |
| 불규칙한 잔요철 | 30% | 높이 잡음 ±2.5 cm, 기존 잔요철 설정 |
| 사각 격자 요철 | 40% | `MeshRandomGridTerrainCfg`, 0.6×0.6 m 칸, 높이 -6~+6 cm 균등 무작위 |
| 경사 | 10% | 기존 경사 파라미터 0.02~0.10 |
| 역경사 | 10% | 기존 역경사 파라미터 0.02~0.10 |

독립된 평지 구역을 10% 확률로 생성합니다. 경사와 역경사는 각각 10%로 합계 20%입니다.
생성기 내부의 중앙 플랫폼과 테두리에도 평탄부가 있습니다. 사각 격자는 평가와 동일하게 `grid_width=0.6`,
`grid_height_range=(0.06, 0.06)`, `platform_width=2.0`, `holes=False`를 사용합니다.
중앙 플랫폼 윗면은 +6 cm이며, 인접 칸의 단차는 최대 약 12 cm입니다.
격자형은 한 구역을 규칙적인 정사각형 칸으로 채우므로 기존의 300회 장애물 배치
설정은 사용하지 않습니다. 잔요철과 경사의 설정은 유지합니다.

지형 선택은 첫 업데이트부터 위 비율로 섞이며 별도 난도 단계는 없습니다.
기존 마찰 랜덤화(정지 0.5~1.25, 동적 0.4~1.0)와 외란 없음 설정을 사용합니다.
원본 보상 task는 7개, `Isaac-Ant-Baseline-FT-Light-v0`는 원본 7개에 아래 4개 벌점을 추가합니다.

| 추가 벌점 | 현재 가중치 |
|---|---:|
| 수직 속도 | -0.05 |
| 롤·피치 각속도 | -0.05 |
| 몸통 좌우 속도 | -0.05 |
| 방향 오차 | -0.1 |

수직 속도와 방향 오차 가중치 조정은 이 단일 단계 Light 파인튜닝 task에만 적용됩니다.
기존 체크포인트에는 소급 적용되지 않으며, 이전 보고서의 실험은 당시 저장된 가중치를 기준으로 합니다.

특히 Light도 원본 보상으로 학습된 같은 baseline에서 시작하여 파인튜닝 때 보상이 바뀝니다.
두 task의 환경, 관측, 행동, 로봇, PPO 설정 및 추가 학습량은 같고 보상만 다릅니다.

### 두 실험을 한 번에 순차 실행

```bash
conda activate lerobot-arena
cd /home/oms/robotics_simulation/IsaacLab_RS

./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Baseline-FT-v0 \
  --num_envs 4096 --seed 42 --max_iterations 1000 \
  --resume --load_run '2026-09-17_13-21-32_ant_baseline$' \
  --checkpoint 'model_999\.pt$' \
  --run_name baseline_ft_rough_blocks_original --headless && \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Baseline-FT-Light-v0 \
  --num_envs 4096 --seed 42 --max_iterations 1000 \
  --resume --load_run '2026-09-17_13-21-32_ant_baseline$' \
  --checkpoint 'model_999\.pt$' \
  --run_name baseline_ft_rough_blocks_light --headless
```

`&&`는 원본 보상 실험이 정상 종료했을 때 Light 실험을 시작합니다.
따로 실행하려면 첫 명령 끝의 `&&`와 이어쓰기 기호를 제거하고 두 명령을 나눠 실행하세요.
이번 실험에서는 이전의 3단계 자동 실행기 `train_ant_finetune.py`를 사용하지 않습니다.

기존 `train.py`가 같은 실험 루트 안에서 체크포인트를 찾으므로 두 task의 루트는
원래 baseline과 같은 `logs/rsl_rl/ant/`입니다. 결과는 아래의 **서로 다른 새 폴더**에 저장됩니다.

- `logs/rsl_rl/ant/실행시각_baseline_ft_rough_blocks_original/model_1998.pt`
- `logs/rsl_rl/ant/실행시각_baseline_ft_rough_blocks_light/model_1998.pt`

원본 baseline 파일은 덮어쓰지 않습니다. 현재 설치된 RSL-RL은 저장된 반복 번호 999부터
새 번호를 매기므로 추가 1,000회 후 최종 번호가 1998입니다. 정책/가치망과 옵티마이저
상태를 불러오며, 학습 환경은 새로 초기화합니다.
격자 평가는 기존 `Isaac-Ant-DR-Eval-Blocks-v0 --seed 24 --num_envs 100`을 사용합니다.
새로 학습하는 모델에는 격자 종류와 높이 범위가 이미 포함되므로 이는 같은 지형 분포의 새 배치 평가입니다.
미학습 지형 종류 평가는 물결 `Isaac-Ant-DR-Eval-v0 --seed 24 --num_envs 100`으로 구분합니다.


### 기존 baseline 파인튜닝 검증 (2026-10-05, 격자형 교체 전)

아래는 기존 사각 블록/홈 ±2 cm 학습 설정의 기록입니다. 새 격자형 학습 결과가 아닙니다.

- 두 task가 보상만 다르고, 평지 구역 제외/원본 7개 또는 Light 11개 보상/PPO 일치 확인.
- 실제 사각 블록 메시의 높이 ±2 cm와 수직 모서리, 설정 객체 격리 검사 통과.
- Light 태스크에서 16개 환경 × 120스텝 GPU 물리 검사 통과.
- 두 task 모두 지정한 Sep 17 baseline의 `model_999.pt`에서 64개 환경 × 추가 2회
  학습하여 새 `model_1000.pt`를 저장했습니다. Adam 업데이트 횟수가 20000에서
  각각 20040으로 이어졌으며, 두 결과의 가중치가 유한하고 갱신됨을 확인했습니다.
- 저장된 환경/에이전트 설정에서 평지 제외와 원본 baseline 로드 경로를 확인했습니다.
- 기존 baseline 모델과 원본/DR/3단계 학습·평가 코드의 SHA-256을 보존했습니다.
  검증 로그: `/tmp/ant_baseline_finetune_check.log`,
  `/tmp/ant_baseline_finetune_original_train.log`, `/tmp/ant_baseline_finetune_light_train.log`.
- 위는 실행 검증입니다. 요청한 추가 1000회 본 학습은 위 명령으로 시작합니다.


## 사각 격자형 요철 평가 (기존 Blocks task)

Task: `Isaac-Ant-DR-Eval-Blocks-v0` — 태스크 이름과 실행 명령은 그대로 사용합니다.
설정: `block_eval_env_cfg.py`. 평가 지형을 `HfDiscreteObstaclesTerrainCfg`에서
**`MeshRandomGridTerrainCfg`**로 교체했습니다.

- 정사각형 격자 한 칸: **0.6 × 0.6 m**.
- 각 칸 높이: 기준면 대비 **-6~+6 cm 사이 균등 무작위 값**.
  `grid_height_range=(0.06, 0.06)`로 높낮이 한계를 고정합니다.
- 중앙 플랫폼 폭: 2 m. 이 생성기의 중앙 플랫폼 윗면은 +6 cm입니다.
- `holes=False`: 구멍 없이 격자로 채운 지형입니다.
- 구역 크기: 20×20 m. 한 구역에 33×33칸이 생성되며, 바깥에 폭 0.1 m씩 평탄한 테두리가 남습니다.
  현재 생성기는 양의 테두리 폭이 필요하므로, 20 m를 정확히 나누는 칸 크기는 사용하지 않습니다.
- 지형 생성 설정 seed: 2402, 평가 CLI seed: 24.
  칸 높이는 PyTorch 난수를 사용하므로 두 모델 평가에 같은 CLI seed를 사용하세요.

두 학습 task는 평지 10%·잔요철 30%·동일한 사각 격자 40%·경사/역경사 합계 20%를 사용합니다.
새 모델의 격자 평가는 학습한 지형 분포 안에서의 평가이며, 학습은 지형 seed 1701/CLI seed 42,
평가는 지형 seed 2402/CLI seed 24로 배치를 구분합니다. 아래 명령에 있는 17:10·17:31의
과거 최종 모델은 교체 전 블록/홈으로 학습했으므로 새 격자로 학습한 모델이 아닙니다.
새 학습 완료 후 평가할 때는 `--checkpoint`를 새 실행 폴더로 바꾸세요.
미학습 지형 종류 평가는 기존 물결 task `Isaac-Ant-DR-Eval-v0`로 구분합니다.

두 모델 모두 **원본 보상 7개**, 마찰 랜덤화, 외란 없음, seed 24, 환경 100개로 평가합니다.
`Completed first episodes: 100/100`은 모두 첫 에피소드를 마쳤다는 뜻이며 성공률이 아닙니다.

```bash
conda activate lerobot-arena
cd /home/oms/robotics_simulation/IsaacLab_RS

# 원본 보상으로 파인튜닝한 모델
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-Eval-Blocks-v0 --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant/2026-10-05_17-10-50_baseline_ft_rough_blocks_original/model_1998.pt

# 현재 Forward-Light 보상으로 파인튜닝한 모델
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play_one_episode.py \
  --task Isaac-Ant-DR-Eval-Blocks-v0 --seed 24 --num_envs 100 \
  --checkpoint logs/rsl_rl/ant/2026-10-05_17-31-33_baseline_ft_rough_blocks_light/model_1998.pt
```

화면 없이 수치만 계산하려면 각 명령 끝에 `--headless`를 추가합니다.
다른 체크포인트도 같은 평가 task에 `--checkpoint` 경로만 바꿔 사용할 수 있습니다.
이번에 실행한 두 평가의 결과 JSON과 터미널 로그를 각 모델 폴더의
`analysis/blocks_eval_seed24_100.json`, `analysis/blocks_eval_seed24_100.log`에 저장했습니다.
위 기본 평가 스크립트는 결과를 터미널에 출력하며, 이 analysis 파일은 이번 작업에서
별도 저장한 기록입니다.


### 블록 평가 결과 및 검증 (2026-10-05, 기본 동작 복원 전)

아래 수치와 `ASSIGNMENT1.md`·공유 보고서의 기존 블록 결과는 **교체 전 사각 블록/홈 ±2 cm 조건**입니다.
현재 사각 격자형 ±6 cm의 결과가 아니므로, 같은 task에서 모델들을 다시 평가해야 합니다.

- 기존 물결 평가와 지형 설정만 다른 task이며 원본 보상 7개, 마찰/관측/행동/종료 조건 동일 확인.
- 블록 평가 16개 환경 × 120스텝 물리 검사 통과.
- 두 실제 최종 `model_1998.pt`를 seed 24, 환경 100개로 평가했고 각각 첫 에피소드 100/100 완료.
- 학습 설정과 두 체크포인트, 기존 train/play_one_episode 파일의 SHA-256 보존 확인.

| 모델 | 원본 누적 보상 평균 | 표준편차 | 평균 에피소드 스텝 |
|---|---:|---:|---:|
| 원본 보상 파인튜닝 | 49.665639 | 26.486469 | 764.97 |
| Forward-Light 파인튜닝 | 50.397475 | 24.337423 | 774.33 |

두 모델의 보상 평균은 이번 조건에서 비슷합니다. 단일 학습/평가 seed 결과이므로
이 차이만으로 보상 설계의 우열이나 미학습 지형 전반에 대한 일반화를 단정하지 않습니다.
