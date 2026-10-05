# Kubric Physics Dataset

Kubric의 PyBullet + Blender 파이프라인으로 동일 초기조건의 **정상 자유낙하 영상**과
**의도적으로 물리를 위반한 영상**을 쌍으로 생성합니다. 학습/평가 모델에는 RGB만
입력하고, simulator state와 렌더 패스는 supervision 및 정량 평가에 사용합니다.

## 생성되는 것

각 scene은 같은 seed를 공유하는 `normal/`과 `violation/` 쌍입니다.

```text
outputs/scene_000001/
├── normal/
│   ├── rgb.mp4
│   ├── rgba/
│   ├── segmentation/
│   ├── depth/
│   ├── forward_flow/
│   ├── backward_flow/
│   ├── state.npz
│   ├── trajectory_gt.npz
│   ├── trajectory_freefall_gt.npz
│   ├── metadata.json
│   └── scene.blend
├── violation/
│   └── ...
└── pair_metrics.json
```

`state.npz`에는 다음 배열이 저장됩니다.

- `position`, `velocity`, `acceleration`: `[T, 3]` world-space state
- `quaternion`, `angular_velocity`
- `contact_ball_floor`: 프레임별 접촉 여부
- `acceleration_valid`: impulse·접촉 주변에서 유한차분 가속도를 제외하기 위한 mask
- `frame`, `time`

`trajectory_gt.npz`는 렌더된 segmentation mask의 정확한 중심점과 depth를 사용한
Morpheus ScoreKit 호환 궤적입니다. `object_1` 배열의 shape은 `[T, 3]`, 좌표 순서는
`[row_y, col_x, depth]`입니다. 즉 RGB에서 SAM2가 추출한 궤적과 simulator GT를 같은
형식으로 바로 비교할 수 있습니다.

전체 `trajectory_gt.npz`에는 충돌과 반동도 포함됩니다. `trajectory_freefall_gt.npz`는
`contact_ball_floor`의 첫 접촉 직전까지만 잘라낸 순수 자유낙하 궤적이므로 Morpheus의
`falling_ball` 식에는 이 파일을 사용해야 합니다.

`metadata.json`에는 카메라 파라미터, segmentation instance 순서, 물체 속성,
intervention 종류·시작/종료 프레임, 요청 dose와 실제 적용 state, collision event가 기록됩니다. `pair_metrics.json`에는
정상 궤적 대비 위치/속도 deviation이 프레임별 및 요약값으로 저장됩니다.

## SSH GPU 서버 빠른 시작

필요 조건:

- Docker
- NVIDIA driver
- NVIDIA Container Toolkit (`docker run --gpus all ...`이 동작해야 함)

```bash
git clone https://github.com/Bo0sung/kubric-physics-dataset.git
cd kubric-physics-dataset
bash scripts/build_image.sh
```

GPU 0에서 정상/위반 쌍 하나를 생성합니다.

```bash
GPU_ID=0 bash scripts/generate_pair.sh \
  scene_000001 100 gravity_scale 12 18 0.5
```

인자의 의미:

```text
generate_pair.sh SCENE_ID SEED INTERVENTION START_FRAME END_FRAME DOSE...
```

지원 intervention:

```bash
# 중력을 12~18 frame 동안 0.5배
GPU_ID=0 bash scripts/generate_pair.sh scene_gravity 100 gravity_scale 12 18 0.5

# 12 frame에서 외부 접촉 없는 속도 kick (m/s)
GPU_ID=0 bash scripts/generate_pair.sh scene_kick 101 velocity_kick 12 12 1.0 0.0 0.0

# 12 frame에서 순간 위치 이동 (m)
GPU_ID=0 bash scripts/generate_pair.sh scene_teleport 102 teleport 12 12 0.4 0.0 0.0

# 12~18 frame 동안 공중 정지
GPU_ID=0 bash scripts/generate_pair.sh scene_freeze 103 freeze 12 18 0.0
```

결과 확인:

```bash
cat outputs/scene_000001/pair_metrics.json
ls outputs/scene_000001/normal
ls outputs/scene_000001/violation
```

같은 intervention으로 seed가 다른 100개 쌍을 순차 생성하려면:

```bash
GPU_ID=0 bash scripts/generate_dataset.sh 100 1000 gravity_scale 12 18 0.5
```

인자는 `COUNT START_SEED INTERVENTION START_FRAME END_FRAME DOSE...` 순서입니다.
중간에 실패해도 이미 `_SUCCESS`가 생성된 scene은 온전히 남으므로, 해당 scene을 확인한 뒤
나머지 seed 범위부터 다시 실행할 수 있습니다.

## GPU 없이 smoke test

PyBullet은 원래 CPU에서 실행됩니다. Blender 렌더도 CPU로 먼저 확인하려면:

```bash
USE_GPU=0 RENDER_DEVICE=CPU bash scripts/generate_pair.sh \
  smoke_test 0 velocity_kick 12 12 1.0 0.0 0.0
```

Blender가 요청한 CUDA/OptiX 장치를 찾지 못하면 worker는 CPU 렌더로 자동 fallback하고
실제 사용 장치를 `metadata.json`에 기록합니다.

## 렌더 없이 physics/GT만 점검

```bash
docker run --rm --user "$(id -u):$(id -g)" \
  -e PYTHONPATH=/workspace -v "$PWD:/workspace" \
  kubric-physics-dataset:latest \
  /usr/bin/python3 workers/freefall_worker.py \
  --variant normal --output-dir outputs/state_only \
  --seed 0 --no-render
```

## 데이터 설계 원칙

- 정상과 위반 영상은 seed, 초기조건, 카메라, 조명, 재질을 공유합니다.
- intervention만 다릅니다.
- train/validation/test 분할은 **pair 단위**로 수행해야 합니다.
- RGB 기반 trajectory와 GT world trajectory를 함께 평가하되 모델 입력에는 RGB만 줍니다.
- GT 3D 궤적을 카메라 파라미터로 2D 투영한 oracle score와 RGB 추적 score를 분리하면
  평가식 오류와 tracker 오류를 구분할 수 있습니다.

## 결과를 사용하는 방법

### 1. 궤적 추출기 학습/검증

- 입력: `rgb.mp4`
- 정답 2D 궤적: 전체 운동은 `trajectory_gt.npz`, 순수 낙하는
  `trajectory_freefall_gt.npz`의 `object_1`
- 정답 3D 상태: `state.npz`의 `position`, `velocity`, `acceleration`
- 보조 supervision: `segmentation/`, `depth/`, optical flow

학습/검증 분할은 같은 seed의 `normal/`과 `violation/`을 분리하지 말고 scene pair 단위로
나눠야 데이터 누수를 피할 수 있습니다.

### 2. Morpheus 점수 실험

Oracle 자유낙하 점수는 `trajectory_freefall_gt.npz`를 Morpheus ScoreKit에 직접 넣고,
영상 기반 점수는
`rgb.mp4`에서 SAM2로 새 궤적을 추출해 계산합니다. 둘의 차이가 tracker 오차이며,
normal/violation 점수 차이가 물리 위반에 대한 평가기의 민감도입니다.

```bash
# kubric-physics-dataset과 morpheus-scorekit이 같은 상위 폴더에 있다고 가정
cd ../morpheus-scorekit
source .venv/bin/activate
export PYTHONPATH="$PWD/src"

python -m morpheus_scorekit.cli score \
  ../kubric-physics-dataset/outputs/scene_000001/normal/trajectory_freefall_gt.npz \
  --experiment falling_ball \
  --output-dir outputs/scene_000001/normal_scores \
  --generated --epochs 10000
```

### 3. 영상 생성 모델 학습

Morpheus 자체는 이 데이터로 범용 trajectory 모델을 학습하지 않습니다. 별도의 video 또는
trajectory predictor를 학습한다면 RGB/condition을 입력으로 하고 `state.npz` 또는
`trajectory_gt.npz`를 supervision으로 사용합니다. Morpheus는 정상/위반 결과를 평가하는
지표로 두는 것이 맞습니다.

## 로컬 단위 테스트

Kubric/Blender 없이 pair metric 코드를 검사할 수 있습니다.

```bash
python -m pip install -e ".[dev]"
pytest -q
```

## 구현상 주의

Kubric의 공개 `PyBullet.run()`을 여러 구간으로 나누면 keyframe 전송 과정에서 물리 state가
구간 끝 frame으로 되감길 수 있습니다. 이 저장소는 공식 step loop를 한 번만 수행하면서
frame boundary에 intervention을 적용하고, 전체 simulation이 끝난 뒤 Blender keyframe을
기록합니다. 따라서 violation frame과 GT state가 일치합니다.

Kubric 자체는 Google Research의 별도 프로젝트이며 Apache-2.0 라이선스를 따릅니다.
이 저장소의 코드는 MIT 라이선스입니다.
