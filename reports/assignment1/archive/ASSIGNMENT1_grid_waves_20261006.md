# 실습 과제 1 — 처음 보는 환경에서도 잘 걷는 Ant 만들기

**평지 베이스 · 처음부터 DR 학습 · DR 파인튜닝: 원본/추가 보상 5개 모델 비교**  
최종 갱신: 2026년 10월 6일 · 평가 기록 `ant_eval_unseen_video_20261006_041317` 기준  
[외부 공유 보고서](https://ant-robustness-assignment1.dhalstjr5247.chatgpt.site)

> **핵심 결과:** 미학습 격자 규격에서는 원본 보상으로 처음부터 DR 학습한 M2가 42.8943으로 가장 높았다. 미학습 물결에서는 추가 보상으로 파인튜닝한 M5가 55.2825로 가장 높았다. DR 학습과 추가 보상의 효과는 지형에 따라 달랐으며, 모든 환경에서 우세한 단일 모델은 확인되지 않았다.

## 1. 연구개요

### 연구 목적

평지에서 잘 걷는 Ant가 지면의 높이·형태와 마찰이 달라져도 전진할 수 있는지 확인한다. 학습하지 않은 환경에 대한 강건성을 높이는 방법으로 **환경 도메인 랜덤화(DR), 평지 사전학습 후 파인튜닝, 보상함수 설계**를 비교한다.

비교 모델은 총 다섯 개다. 평지 베이스를 기준으로, 혼합 지형에서 처음부터 학습한 두 모델과 같은 평지 베이스에서 각각 파인튜닝한 두 모델을 평가했다. 각 쌍은 원본 보상 유지/추가 벌점 적용으로 구분한다.

로봇의 링크 길이·관절 개수·USD 하드웨어 구조, 관측 차원과 신경망 구조는 변경하지 않았다. 발 들기 높이 보상과 별도의 센서 추가는 이번 실험에 포함하지 않았다.

### 새로운 환경의 정의

| 평가 | 학습 설정 | 평가 설정 | 검증 범위 |
|---|---|---|---|
| 사각 격자 | 칸 너비 0.6 m, 높이 ±6 cm | **칸 너비 0.7 m**, 높이 ±6 cm, 지형 seed 2403 | 학습하지 않은 격자 규격·배치로의 일반화 |
| 물결 | DR 혼합 지형에 물결 없음 | 진폭 파라미터 3~6 cm, 물결 개수 4/8 혼합, 지형 seed 2404 | 학습하지 않은 지형 종류로의 일반화 |

격자 생성 방식 자체는 학습에 포함되어 있으므로 이를 완전히 새로운 지형 종류라고 부르지 않는다. 물결은 이번 다섯 모델의 학습 지형에 없는 종류다. 두 지형의 원점수를 섞어 종합 순위를 만들지 않고 **같은 지형 안에서 모델들을 비교**한다.

## 2. 연구가설

| 번호 | 모델 | 초기 정책 | 학습 보상 | 총 학습량 |
|---|---|---|---|---:|
| M1 | 평지 베이스 | 무작위 초기화 | 원본 7개 | 평지 1,000회 |
| M2 | 원본 보상 · 처음부터 DR | 무작위 초기화 | 원본 7개 | DR 1,000회 |
| M3 | 추가 보상 · 처음부터 DR | 무작위 초기화 | 원본 7개 + 벌점 4개 | DR 1,000회 |
| M4 | 원본 보상 · DR 파인튜닝 | M1 | 원본 7개 | 평지 1,000 + DR 1,000회 |
| M5 | 추가 보상 · DR 파인튜닝 | M1 | 원본 7개 + 벌점 4개 | 평지 1,000 + DR 1,000회 |

- **H1 — 환경 다양화:** 혼합 지형과 마찰 변화에 노출되면 평지 학습보다 미학습 환경 성능이 향상될 것이다. 총 학습량이 같은 M1/M2와, 추가 적응을 수행한 M1/M4를 구분해 본다.
- **H2 — 학습 출발점:** 평지에서 배운 보행을 출발점으로 사용하면 처음부터 DR 학습하는 것과 다른 일반화 특성을 보일 것이다. M2/M4, M3/M5를 비교하되 총 학습량 차이를 함께 고려한다.
- **H3 — 보상 설계:** 몸통의 상하·좌우 움직임, 롤·피치 회전과 방향 오차 벌점을 추가하면 미학습 환경 성능이 향상될 것이다. 같은 학습 방식 안에서 M2/M3, M4/M5를 비교한다.

M4와 M5는 **동일한 M1 체크포인트에서 독립적으로 시작**했다. M5가 M4 결과를 이어 학습한 것은 아니다. M2/M3 및 M4/M5는 각각 같은 환경·학습 seed·PPO 설정과 업데이트 수를 사용하며, 쌍 안에서 보상 구성이 다르다.

## 3. Training 및 Rollout

### 3.1 컴퓨터 사양 및 주요 라이브러리

| 항목 | 사양·버전 |
|---|---|
| 운영체제 | Ubuntu 22.04.5 LTS, x86_64 |
| CPU | Intel Xeon Gold 5512U, 28코어 / 56스레드 |
| 메모리 | OS에서 인식한 총 메모리 약 251 GiB |
| GPU | NVIDIA RTX 6000 Ada Generation, 48 GB급 VRAM (`nvidia-smi`: 49,140 MiB) |
| NVIDIA 드라이버 | 580.178.04 |
| CUDA Toolkit / PyTorch CUDA 런타임 | 12.8 (`nvcc` 12.8.61) / 12.8 |
| Python / Conda 환경 | 3.11.16 / `lerobot-arena` |
| Isaac Sim | 5.1.0.0 |
| Isaac Lab 저장소 | `VERSION` 기준 2.3.0 |
| Isaac Lab 패키지 | `isaaclab` 0.47.2, `isaaclab_tasks` 0.11.6, `isaaclab_rl` 0.4.4 |
| PyTorch / torchvision | 2.7.0+cu128 / 0.22.0+cu128 |
| RSL-RL / Gymnasium | 3.0.1 / 1.2.0 |
| TensorBoard / NumPy | 2.21.0 / 1.26.0 |
| Hydra / Warp | 1.3.7 / 1.17.0 |

버전은 보고서 작성 시 실제 설치 환경에서 확인했다. `nvidia-smi`의 CUDA 13.0 표시는 드라이버 지원 상한이며, 사용 중인 Toolkit·PyTorch 런타임은 12.8이다. 과거 평지 학습 시점의 전체 패키지 스냅샷을 별도로 남긴 것은 아니다.

### 3.2 공통 실험 설정

#### 학습과 도메인 랜덤화

| 설정 | 값 |
|---|---|
| 학습 환경 수 / seed | 4,096 / 42 |
| 각 실행의 학습량 | 1,000 PPO 업데이트 |
| 업데이트당 수집량 | 환경당 32스텝, 총 131,072 transition |
| 각 1,000회 실행의 수집량 | 131,072,000 transition |
| Actor / Critic | 각각 은닉층 400–200–100, ELU |
| 관측 / 행동 | 60차원 / 관절 effort 8차원 |
| PPO | 초기 learning rate 0.0005, adaptive, 5 epochs, 4 minibatches |
| 할인율 / GAE λ / PPO clip | 0.99 / 0.95 / 0.2 |
| 물리 / 제어 주기 | 1/120초 / 1/60초 |
| 최대 에피소드 | 16초 = 960 제어 스텝 |
| 외란 | 없음 |

| DR 학습 지형 | 생성 확률 | 설정 |
|---|---:|---|
| 평지 | 10% | `MeshPlaneTerrainCfg` |
| 불규칙 잔요철 | 30% | 높이 ±2.5 cm, 높이 간격 0.5 cm, 공간 샘플링 0.4 m |
| 사각 격자 | 40% | `MeshRandomGridTerrainCfg`, 칸 0.6×0.6 m, 높이 −6~+6 cm |
| 경사 | 10% | slope 파라미터 0.02~0.10 |
| 역경사 | 10% | 같은 범위의 반대 경사 |

DR 학습 지형은 seed 1701로 생성한 20×20 m 타일 16×16개, 총 320×320 m다. 비율은 타일 생성 확률이며, 실제 타일 개수나 로봇의 경험 시간 비율을 보장하지 않는다. 지형을 매 스텝 또는 리셋마다 다시 생성하지 않는다. 격자는 `holes=False`, 중앙 플랫폼 폭 2 m를 사용하며, 인접 칸 단차는 최대 약 12 cm다. slope 값의 단위는 도(degree)가 아니다.

M2~M5의 로봇별 마찰은 정지 마찰 0.5~1.25, 동적 마찰 0.4~1.0에서 배정하고 동적 마찰이 정지 마찰보다 크지 않도록 한다. 64개 후보 중 배정하며 리셋 간 유지한다. 지면 마찰 1.0, 결합 방식 `average`를 사용한다. M1은 원본 평지 설정으로 학습했다.

#### 보상 설계 — 실제 학습 YAML 기준

| 원본 보상 항목 | 가중치 | 역할 |
|---|---:|---|
| progress | +1.0 | 목표까지의 거리 감소 |
| alive | +0.5 | 종료되지 않은 상태 유지 |
| upright | +0.1 | 몸통 수직 정렬, 투영값 기준 0.93 |
| move_to_target | +0.5 | 몸통 정면과 목표 방향 정렬, 투영값 기준 0.8 |
| action_l2 | −0.005 | 큰 행동 명령 억제 |
| energy | −0.05 | 행동·관절 속도 기반 에너지 사용 대용 지표 |
| joint_pos_limits | −0.1 | 관절 한계 근처 사용 억제 |

M3/M5의 추가 벌점은 다음과 같다. **이번 결과는 수직 속도 −0.05, 방향 오차 −0.1로 완화한 모델**이다. 이전 −0.5/−1.0 모델의 결과와 혼합하지 않는다.

| 추가 항목 | 식·가중치 | 의도 |
|---|---|---|
| 수직 속도 | `−0.05 × v_z_world²` | 몸통의 과도한 상하 움직임 억제 |
| 롤·피치 각속도 | `−0.05 × (ω_x_body² + ω_y_body²)` | 몸통 회전 흔들림 억제 |
| 좌우 속도 | `−0.05 × v_y_body²` | 몸통 좌우 움직임 억제 |
| 방향 오차 | `−0.1 × (1 − cos(목표 방위각 − 몸통 yaw))` | 목표 방향 정렬 유도 |

RewardManager는 각 항목의 가중치와 제어 시간 간격을 곱한다. 원본에도 방향 보상이 있으므로 보행 모양을 ‘방향 보상이 없어서’라고 설명하지 않는다. 벌점 추가가 특정 발 궤적이나 보행 형태를 보장하는 것은 아니다.

#### 평가·영상 조건

- 공식 `scripts/reinforcement_learning/rsl_rl/play_one_episode.py`로 평가했다. 모든 모델에 **원본 7개 보상만 적용**했다.
- **`--seed 24 --num_envs 100`**, 마찰 랜덤화 범위는 위 DR 설정과 같고 외란은 없다. 로봇 간 기본 배치는 5 m 간격이다.
- 격자 task: `Isaac-Ant-DR-Eval-Blocks-v0`, 지형 seed 2403, 칸 너비 0.7 m, 높이 ±6 cm.
- 물결 task: `Isaac-Ant-DR-Eval-v0`, 지형 seed 2404, `amplitude_range=(0.03, 0.06)`, `num_waves=4/8`를 50%씩 구성한다.
- CLI seed 24와 지형 생성 설정의 seed는 별개다. 다섯 모델에 지형별로 같은 설정을 적용했다.
- 각 로봇의 **첫 에피소드만 종료 스텝까지 포함**한다. mean/std는 100개 누적 보상의 평균/모집단 표준편차다. std는 평균의 신뢰구간이 아니다.
- 10개 평가 모두 `Completed first episodes: 100/100`을 확인했다. 이는 통계 수집 완료이며 성공률 100%를 뜻하지 않는다.

원본 평가 폴더는 `logs/ant_eval_unseen_video_20261006_041317`이다. 이번 보고서는 **사용자가 실행한 해당 로그를 직접 집계**했으며 보고서 작성을 위해 재평가하지 않았다. 평가 폴더의 모델 사본 해시가 각 학습 체크포인트와 일치함을 확인했다.

영상 10개는 모두 1280×720, 60fps, 960프레임(16초)이다. 카메라는 0번 로봇을 따라가며, 해당 로봇의 첫 에피소드가 일찍 끝나면 자동 리셋 이후 장면도 포함된다. 따라서 영상 길이가 생존 시간은 아니며 한 로봇의 영상이 100개 전체를 대표하지 않는다.

아래 TensorBoard 그래프는 각 실행의 실제 이벤트 파일에서 추출했다. 얇은 선은 원 로그, 굵은 선은 50회 이동평균이다. 표는 평활화 전 처음/마지막 100회 평균이다. 가로축은 **해당 실행의 업데이트 1~1,000회**이며 파인튜닝 이전의 평지 학습은 포함하지 않는다. 학습 지형과 보상 정의가 다르므로 학습 누적 보상을 모델 간 성능 순위로 직접 비교하지 않는다.

### 3.3 M1 — 평지 베이스

체크포인트: `logs/rsl_rl/ant/2026-09-17_13-21-32_ant_baseline/model_999.pt`

![M1 TensorBoard 학습 기록](reports/assignment1/assets_unseen/tensorboard_M1.png)

| 학습 지표 | 처음 100회 평균 | 마지막 100회 평균 |
|---|---:|---:|
| 학습 누적 보상 | 24.3946 | 131.8703 |
| 에피소드 길이 (스텝) | 704.9865 | 910.8404 |
| 진행 보상 성분 (에피소드 누적 / 16초) | 1.6220 | 8.3838 |

| 평가 지형 | 보상 평균 ± 표준편차 | 에피소드 길이 평균 ± 표준편차 (스텝) |
|---|---:|---:|
| 사각 격자 | 5.546510 ± 5.197220 | 439.32 ± 357.87 |
| 물결 | 52.240256 ± 29.625400 | 679.54 ± 316.60 |

격자 평균 보상은 5.5465로 다섯 모델 중 가장 낮았다. 반면 물결은 52.2403으로, 별도 물결 학습 없이도 M2/M3보다 높은 성능을 보였다. 평지에서 얻은 정책의 일반화가 지형 종류에 따라 다름을 보여준다.

[M1 사각 격자 평가 영상](reports/assignment1/assets_unseen/rollout_M1_grid.mp4)

[M1 물결 평가 영상](reports/assignment1/assets_unseen/rollout_M1_waves.mp4)


### 3.4 M2 — 원본 보상 · 처음부터 DR

체크포인트: `logs/rsl_rl/ant/2026-10-06_02-01-31_scratch_rough_blocks_original/model_999.pt`

![M2 TensorBoard 학습 기록](reports/assignment1/assets_unseen/tensorboard_M2.png)

| 학습 지표 | 처음 100회 평균 | 마지막 100회 평균 |
|---|---:|---:|
| 학습 누적 보상 | 8.1401 | 44.6866 |
| 에피소드 길이 (스텝) | 528.9255 | 579.7731 |
| 진행 보상 성분 (에피소드 누적 / 16초) | 0.4504 | 2.7769 |

| 평가 지형 | 보상 평균 ± 표준편차 | 에피소드 길이 평균 ± 표준편차 (스텝) |
|---|---:|---:|
| 사각 격자 | 42.894314 ± 25.629163 | 685.78 ± 358.47 |
| 물결 | 21.171280 ± 18.601088 | 412.21 ± 289.06 |

격자 평균 보상 42.8943으로 가장 높았고 M1의 약 7.73배였다. 물결은 21.1713으로 가장 낮았다. 학습에 포함된 격자 종류의 새 규격에는 강했지만 다른 종류로의 일반화는 제한적이었다.

[M2 사각 격자 평가 영상](reports/assignment1/assets_unseen/rollout_M2_grid.mp4)

[M2 물결 평가 영상](reports/assignment1/assets_unseen/rollout_M2_waves.mp4)


### 3.5 M3 — 추가 보상 · 처음부터 DR

체크포인트: `logs/rsl_rl/ant/2026-10-06_02-21-55_scratch_rough_blocks_light/model_999.pt`

![M3 TensorBoard 학습 기록](reports/assignment1/assets_unseen/tensorboard_M3.png)

| 학습 지표 | 처음 100회 평균 | 마지막 100회 평균 |
|---|---:|---:|
| 학습 누적 보상 | 4.6769 | 38.0231 |
| 에피소드 길이 (스텝) | 517.7339 | 626.4574 |
| 진행 보상 성분 (에피소드 누적 / 16초) | 0.2924 | 2.4069 |

| 평가 지형 | 보상 평균 ± 표준편차 | 에피소드 길이 평균 ± 표준편차 (스텝) |
|---|---:|---:|
| 사각 격자 | 37.144177 ± 22.444523 | 712.65 ± 365.67 |
| 물결 | 26.430444 ± 18.875023 | 548.87 ± 331.02 |

M2에 비해 격자 평균 보상은 13.4% 낮고 물결은 24.8% 높았다. 격자 평균 에피소드 길이는 685.78→712.65스텝으로 늘었지만 보상은 낮아, 오래 유지되는 것만으로 전진 성능 향상을 판단할 수 없다. 추가 벌점의 효과는 지형에 따라 달랐다.

[M3 사각 격자 평가 영상](reports/assignment1/assets_unseen/rollout_M3_grid.mp4)

[M3 물결 평가 영상](reports/assignment1/assets_unseen/rollout_M3_waves.mp4)


### 3.6 M4 — 원본 보상 · DR 파인튜닝

체크포인트: `logs/rsl_rl/ant/2026-10-05_21-47-43_baseline_ft_rough_blocks_original/model_1998.pt`

![M4 TensorBoard 학습 기록](reports/assignment1/assets_unseen/tensorboard_M4.png)

| 학습 지표 | 처음 100회 평균 | 마지막 100회 평균 |
|---|---:|---:|
| 학습 누적 보상 | 9.8183 | 18.7296 |
| 에피소드 길이 (스텝) | 421.9205 | 584.8142 |
| 진행 보상 성분 (에피소드 누적 / 16초) | 0.6422 | 1.1890 |

| 평가 지형 | 보상 평균 ± 표준편차 | 에피소드 길이 평균 ± 표준편차 (스텝) |
|---|---:|---:|
| 사각 격자 | 12.351667 ± 10.774248 | 619.52 ± 385.79 |
| 물결 | 54.159793 ± 29.305611 | 712.80 ± 320.55 |

격자 평균 보상은 M1 대비 2.23배로 개선됐지만 M2보다 낮았다. 물결은 54.1598로 M1보다 3.7% 높았다. 평지 기반 정책에서 출발한 파인튜닝은 이번 결과에서 물결 성능을 유지하면서 격자 성능을 개선했다.

[M4 사각 격자 평가 영상](reports/assignment1/assets_unseen/rollout_M4_grid.mp4)

[M4 물결 평가 영상](reports/assignment1/assets_unseen/rollout_M4_waves.mp4)


### 3.7 M5 — 추가 보상 · DR 파인튜닝

체크포인트: `logs/rsl_rl/ant/2026-10-05_23-50-19_baseline_ft_rough_blocks_light/model_1998.pt`

![M5 TensorBoard 학습 기록](reports/assignment1/assets_unseen/tensorboard_M5.png)

| 학습 지표 | 처음 100회 평균 | 마지막 100회 평균 |
|---|---:|---:|
| 학습 누적 보상 | 8.4674 | 16.3515 |
| 에피소드 길이 (스텝) | 419.2527 | 583.3942 |
| 진행 보상 성분 (에피소드 누적 / 16초) | 0.6437 | 1.1886 |

| 평가 지형 | 보상 평균 ± 표준편차 | 에피소드 길이 평균 ± 표준편차 (스텝) |
|---|---:|---:|
| 사각 격자 | 12.816671 ± 11.009050 | 573.42 ± 386.33 |
| 물결 | 55.282532 ± 28.416206 | 701.43 ± 300.99 |

격자는 12.8167, 물결은 55.2825로 M4보다 각각 3.8%, 2.1% 높았다. 물결의 평균은 가장 높지만 차이가 작아 반복 학습·평가 없이 확실한 우위로 단정할 수 없다. 두 지형의 평균 에피소드 길이는 M4보다 짧았다.

[M5 사각 격자 평가 영상](reports/assignment1/assets_unseen/rollout_M5_grid.mp4)

[M5 물결 평가 영상](reports/assignment1/assets_unseen/rollout_M5_waves.mp4)


### 3.8 결과 종합 및 가설 검토

| 모델 | 격자 보상 평균 ± 표준편차 | 물결 보상 평균 ± 표준편차 | 격자 평균 지속 시간 | 물결 평균 지속 시간 |
|---|---:|---:|---:|---:|
| M1 · 평지 베이스 | 5.5465 ± 5.1972 | 52.2403 ± 29.6254 | 7.32초 | 11.33초 |
| M2 · 원본 보상 · 처음부터 DR | 42.8943 ± 25.6292 | 21.1713 ± 18.6011 | 11.43초 | 6.87초 |
| M3 · 추가 보상 · 처음부터 DR | 37.1442 ± 22.4445 | 26.4304 ± 18.8750 | 11.88초 | 9.15초 |
| M4 · 원본 보상 · DR 파인튜닝 | 12.3517 ± 10.7742 | 54.1598 ± 29.3056 | 10.33초 | 11.88초 |
| M5 · 추가 보상 · DR 파인튜닝 | 12.8167 ± 11.0091 | 55.2825 ± 28.4162 | 9.56초 | 11.69초 |

![5개 모델의 미학습 환경 평가 보상](reports/assignment1/assets_unseen/evaluation_comparison.png)

표의 지속 시간은 평균 스텝 수 × 1/60초다. 정지해 있는 시간도 포함되므로 계속 전진한 시간으로 해석하지 않는다.

| 비교 | 격자 평균 보상 변화 | 물결 평균 보상 변화 | 해석 |
|---|---:|---:|---|
| M1 → M2: 같은 1,000회, 평지 → DR | +673.4% (7.73배) | −59.5% | 환경 다양화 효과가 지형별로 갈림 |
| M1 → M4: 원본 보상 파인튜닝 추가 | +122.7% (2.23배) | +3.7% | 격자 개선, 물결은 평균상 소폭 상승 |
| M2 → M3: 처음부터 DR에서 추가 보상 | −13.4% | +24.8% | 격자 성능 감소와 물결 성능 증가 |
| M4 → M5: 파인튜닝에서 추가 보상 | +3.8% | +2.1% | 평균 차이가 작아 확정적 우위 판단 보류 |

- **H1은 일부 조건에서만 지지됐다.** 같은 학습량의 M1/M2에서 격자는 크게 개선됐지만 물결은 낮아졌다. DR만으로 모든 미학습 환경에 대한 성능이 일괄 개선되지는 않았다.
- **H2에서는 지형별 순위 차이가 관찰됐다.** 격자는 처음부터 DR 학습한 M2/M3가, 물결은 평지 기반 M4/M5가 높았다. 평지에서 얻은 보행 특성이 파인튜닝 후에도 영향을 줄 수 있다는 해석은 가능하지만, 정책 내부나 접촉 동작을 정량 분석한 인과 결론은 아니다.
- **H3은 일관되게 지지되지 않았다.** 추가 보상의 효과는 지형·학습 방식에 따라 달랐다. 이번 로그에 몸통 속도·발 접촉·정지 비율의 정량 기록이 없으므로 안정성 개선 자체를 입증했다고 쓰지 않는다.

### 3.9 학습·평가 재현 명령어

아래 명령은 `lerobot-arena` 환경과 저장소 루트에서 실행한다. 보고서의 실제 실행 경로와 체크포인트는 각 모델 절에 명시했다.

```bash
conda activate lerobot-arena
cd /home/oms/robotics_simulation/IsaacLab_RS
```

#### M1 — 평지 베이스 학습

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-v0 \
  --num_envs 4096 --seed 42 --max_iterations 1000 \
  --run_name ant_baseline --headless
```

#### M2/M3 — 처음부터 DR 학습, 순차 실행

```bash
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Baseline-FT-v0 \
  --num_envs 4096 --seed 42 --max_iterations 1000 \
  --run_name scratch_rough_blocks_original --headless && \
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py \
  --task Isaac-Ant-Baseline-FT-Light-v0 \
  --num_envs 4096 --seed 42 --max_iterations 1000 \
  --run_name scratch_rough_blocks_light --headless
```

태스크 이름에 FT가 있지만 `--resume`이 없으면 무작위 초기화에서 시작한다. 서로의 최종 모델을 이어받지 않는다.

#### M4/M5 — 같은 평지 베이스에서 각각 파인튜닝

```bash
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

이 명령은 보고서에 사용한 기존 평지 체크포인트를 선택한다. 평지부터 다시 학습해 재현할 경우 새 평지 실행 폴더명으로 `--load_run`을 바꿔야 한다. `--max_iterations 1000`은 파인튜닝에서는 추가 업데이트 1,000회다. iteration 999부터 이어져 최종 파일명은 `model_1998.pt`다. 정책·가치망과 optimizer 상태를 함께 불러온다.

#### 다섯 모델 × 두 지형 평가 및 영상 저장

다음은 이번 평가와 동일한 명령이다. `reports/assignment1/evaluate_unseen.sh`로도 제공한다. 실행하면 새로운 시간 이름의 결과 폴더가 생긴다. 체크포인트 사본을 모델·지형별 폴더에 두므로 기존 학습 영상은 덮어쓰지 않는다.

[평가 명령 다운로드](reports/assignment1/evaluate_unseen.sh)

```bash
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
```

모델별 `grid/evaluation.log`, `waves/evaluation.log`에 mean/std가 기록된다. 영상은 각각 `videos/play/rl-video-step-0.mp4`에 저장된다. 모든 환경이 조기 종료되면 녹화 길이는 설정한 960스텝보다 짧아질 수 있다. 이번 제출용 평가에서는 10개 모두 16초 영상을 얻었다.

TensorBoard는 다음 명령으로 실행한 뒤 각 실행 폴더를 선택할 수 있다.

```bash
python -m tensorboard.main --logdir logs/rsl_rl/ant --port 6006
```

### 3.10 해석의 한계

1. **학습량:** M1~M3은 총 1,000회, M4/M5는 총 2,000회다. 처음부터 DR 2,000회와 평지 2,000회 대조군이 없으므로 파인튜닝 방식만의 인과 효과를 분리하지 못한다.
2. **반복 수:** 학습 seed 42, 평가 seed 24 한 쌍이다. 100개 병렬 환경은 독립적으로 학습한 100개 정책이 아니다. std는 환경 간 변동성이며 학습 seed 간 변동성이 아니다.
3. **추가 보상:** 네 벌점을 동시에 적용했으므로 어느 항목이 효과를 냈는지 구분하지 못한다. 발을 높이 드는 동작을 직접 보상한 실험도 아니다.
4. **종료 원인·정지:** 현재 공식 로그는 누적 보상과 에피소드 길이의 mean/std만 제공한다. 성공률, 넘어짐 수, 정지 비율, 이동거리 등은 이번 평가의 수치로 제시하지 않는다. 이전 별도 진단에서 측정한 값을 이번 결과에 재사용하지 않았다.
5. **초기 접촉·유한 지형:** 기존 초기화 방식에 따른 시작 직후 튐의 영향이 남아 있을 수 있다. 평가 지형은 160×160 m이며 일부 영상 표본에 지형 가장자리가 보인다. 경계 이탈과 넘어짐의 영향을 현재 요약 로그만으로 구분할 수 없다.
6. **평가 범위:** 격자는 같은 생성 방식의 새로운 규격, 물결은 새로운 종류다. 미학습 지형 두 조건에 대한 결과이며 모든 지형에 대한 강건성을 입증한 것은 아니다.

## 4. 결론

원본 보상으로 처음부터 DR 학습한 M2는 미학습 격자 규격에서 가장 높은 평균 보상 **42.8943**을 얻었다. 그러나 물결에서는 **21.1713**으로 평지 베이스 **52.2403**보다 낮았다. 다양한 학습 환경에 노출되는 것만으로 모든 미학습 지형의 성능이 개선되지는 않았다.

평지 베이스에서 파인튜닝한 M4/M5는 격자 보상을 베이스의 **5.5465**에서 **12.3517/12.8167**로 높였고, 물결에서는 **54.1598/55.2825**를 얻었다. 이번 조건에서는 평지에서 얻은 정책을 출발점으로 삼는 접근이 물결 성능을 유지하면서 격자 적응을 개선하는 결과를 보였다. 다만 총 학습량 차이를 포함한 비교다.

추가 보상은 처음부터 DR 학습할 때 격자 성능을 낮추고 물결 성능을 높였으며, 파인튜닝에서는 두 지형의 평균을 소폭 높였다. **보상 설계의 효과는 지형과 학습 방식에 따라 달랐고, 모든 조건에서 일관된 개선은 확인되지 않았다.**

이번 연구의 핵심은 최고 보상 하나를 선택하는 데 있지 않다. 학습한 지형 종류 안에서의 새 규격 적응과 새로운 지형 종류로의 일반화를 구분해야 한다는 점이다. 후속 검증에는 동일 총 학습량 비교, 여러 학습 seed, 보상 항목별 제거 실험, 종료 원인과 정지·접촉 동작의 정량 측정이 필요하다.

### 결과 파일과 제출 자료

- [평가 수치 원자료](reports/assignment1/assets_unseen/evaluation_summary.json): 공식 로그의 소수점 6자리 결과와 원본 로그 경로.
- [TensorBoard 요약](reports/assignment1/assets_unseen/tensorboard_summary.json), [학습 설정](reports/assignment1/assets_unseen/training_configs.json), [체크포인트·평가 로그·영상 해시](reports/assignment1/assets_unseen/evidence_manifest.json).
- [그래프·영상·실험 원자료](reports/assignment1/assets_unseen/): 다섯 모델의 학습 그래프와 두 지형 영상 10개.
- 평가 명령은 3.9절과 [evaluate_unseen.sh](reports/assignment1/evaluate_unseen.sh)에 있다.
- 이전 3개 모델 보고서와 원자료는 로컬 archive 및 assets_grid에 보존했으며 현재 결과 표와 섞지 않았다.
- 외부 링크는 보고서 공유용이다. 과제의 조별 GitHub 링크와 5분 발표 PPT는 별도 제출 항목이다.
