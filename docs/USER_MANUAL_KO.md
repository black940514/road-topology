# Road Topology Segmentation 사용자 가이드

**문서 버전:** 1.0
**최종 업데이트:** 2026년 1월
**대응 영문 매뉴얼:** USER_MANUAL.md v0.1.0

---

## 목차

1. [시작하기](#1-시작하기)
   - [프로젝트 소개](#11-프로젝트-소개)
   - [시스템 요구사항](#12-시스템-요구사항)
   - [설치 방법](#13-설치-방법)
   - [설치 확인](#14-설치-확인)
2. [핵심 개념](#2-핵심-개념)
   - [도로 토폴로지 클래스](#21-도로-토폴로지-클래스)
   - [파이프라인 개요](#22-파이프라인-개요)
   - [용어 정리](#23-용어-정리)
3. [빠른 시작](#3-빠른-시작-5분-가이드)
   - [3단계 추론 튜토리얼](#31-3단계-추론-튜토리얼)
4. [주요 워크플로우](#4-주요-워크플로우)
   - [워크플로우 A: 영상에서 학습 데이터 생성](#41-워크플로우-a-영상에서-학습-데이터-생성)
   - [워크플로우 B: 모델 학습](#42-워크플로우-b-모델-학습)
   - [워크플로우 C: 추론](#43-워크플로우-c-추론)
   - [전체 파이프라인 통합](#44-전체-파이프라인-통합)
5. [CLI 명령어 레퍼런스](#5-cli-명령어-레퍼런스)
   - [전역 명령어](#51-전역-명령어-global-commands)
   - [pseudolabel 명령어](#52-pseudolabel-명령어-pseudo-label-commands)
   - [train 명령어](#53-train-명령어-training-commands)
   - [infer 명령어](#54-infer-명령어-inference-commands)
   - [evaluate 명령어](#55-evaluate-명령어-evaluation-commands)
6. [데이터셋 가이드](#6-데이터셋-가이드)
   - [지원 데이터셋 비교표](#61-지원-데이터셋-비교표)
   - [KITTI Road 다운로드 및 준비](#62-kitti-road-다운로드-및-준비)
   - [BDD100K 다운로드](#63-bdd100k-다운로드-프로덕션)
   - [Cityscapes 다운로드](#64-cityscapes-다운로드-고품질)
   - [커스텀 데이터셋 준비](#65-커스텀-데이터셋-준비)
   - [데이터셋 검증](#66-데이터셋-검증)
7. [설정 파일 가이드](#7-설정-파일-가이드)
   - [설정 파일 목록](#71-설정-파일-목록)
   - [주요 파라미터 10개](#72-주요-파라미터-10개)
   - [환경 변수](#73-환경-변수-road_topo_)
8. [FAQ](#8-faq)
   - [일반 질문](#일반-질문)
   - [데이터 관련 질문](#데이터-관련-질문)
   - [모델/학습 질문](#모델학습-질문)
   - [배포 관련 질문](#배포-관련-질문)
9. [문제 해결](#9-문제-해결)
   - [설치 관련 오류](#설치-관련-오류)
   - [데이터 관련 오류](#데이터-관련-오류)
   - [학습 관련 오류](#학습-관련-오류)
   - [추론 관련 오류](#추론-관련-오류)
   - [일반 디버깅 팁](#일반-디버깅-팁)

**부록**
- [부록 A. 영문 매뉴얼 참조 가이드](#부록-a-영문-매뉴얼-참조-가이드)

---

## 1. 시작하기

### 1.1 프로젝트 소개

Road Topology Segmentation은 CCTV 영상에서 도로 인프라 요소를 자동으로 분할하는 종합 시스템입니다. 이 시스템은 차량 궤적 분석, Segment Anything Model(SAM)을 이용한 제로샷 분할, 그리고 심층 신경망을 결합한 다단계 파이프라인을 사용합니다.

주요 기능:
- **자동 의사 레이블 생성**: 차량 궤적에서 학습 데이터를 자동 생성
- **제로샷 분할**: SAM 활용으로 추가 학습 없이 분할 수행
- **고정밀 추론**: SegFormer/Mask2Former 모델로 운영 환경 추론
- **5가지 도로 클래스 분류**: Road, Lane, Crosswalk, Sidewalk, Background

### 1.2 시스템 요구사항

| 항목 | 요구사항 | 비고 |
|------|--------|------|
| Python | 3.10 이상 | 3.11, 3.12 권장 |
| CUDA | 11.8 이상 | GPU 가속 (NVIDIA) |
| MPS | - | Apple Silicon (M1/M2/M3) 자동 지원 |
| GPU 메모리 | 16 GB 이상 | 학습: 16GB, 추론: 8GB 최소 |
| 시스템 메모리 | 16 GB 이상 | 데이터 로딩 및 전처리용 |
| 디스크 공간 | 50 GB 이상 | 모델, 데이터셋, 출력물 저장 |
| OS | Linux/macOS/Windows | WSL2 환경도 지원 |

### 1.3 설치 방법

#### 1단계: 패키지 관리자 설치

먼저 `uv` 패키지 관리자를 설치합니다. `uv`는 매우 빠른 Python 패키지 설치 도구입니다.

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -ExecutionPolicy BypassPolicy -c "irm https://astral.sh/uv/install.ps1 | iex"
```

설치 후 재시작하거나 다음 명령으로 PATH를 갱신합니다:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

#### 2단계: 가상환경 생성 및 활성화

```bash
# 가상환경 생성
uv venv .venv

# 가상환경 활성화
source .venv/bin/activate              # Linux/macOS
# 또는
.venv\Scripts\activate                 # Windows
```

#### 3단계: 의존성 설치

```bash
# 프로젝트를 개발 모드로 설치 (선택사항)
uv pip install -e ".[dev]"

# 또는 표준 모드로 설치
uv pip install -e .

# PyTorch 설치 (플랫폼별)
# NVIDIA GPU (CUDA)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Apple Silicon (MPS) - 기본 설치로 MPS 자동 지원
uv pip install torch torchvision

# CPU 전용
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

설치가 완료되면 `road-topo` 명령어를 사용할 수 있습니다.

### 1.4 설치 확인

설치가 올바르게 완료되었는지 다음 명령으로 확인합니다:

```bash
# 버전 확인
road-topo version

# 시스템 정보 확인
road-topo info
```

출력 예시:

```
# CUDA 환경
road-topology version 0.1.0

Road Topology Segmentation
PyTorch version: 2.1.0
CUDA available: True
CUDA device: NVIDIA A100

# Apple Silicon 환경
road-topology version 0.1.0

Road Topology Segmentation
PyTorch version: 2.1.0
MPS available: True
Default device: Apple Silicon (arm64)
```

---

## 2. 핵심 개념

### 2.1 도로 토폴로지 클래스

이 시스템은 도로 장면을 다음 5가지 클래스로 분류합니다:

| Class ID | 영문명 | 색상 | 한국어 설명 |
|----------|--------|------|-----------|
| 0 | Background | 검은색 (0, 0, 0) | 배경: 건물, 나무, 하늘 등 도로가 아닌 영역 |
| 1 | Road | 회색 (128, 128, 128) | 도로 표면: 아스팔트 및 포장도로 |
| 2 | Lane | 흰색 (255, 255, 255) | 차선: 차선 마킹 및 구분선 |
| 3 | Crosswalk | 노란색 (255, 255, 0) | 횡단보도: 보행자 횡단 지역 |
| 4 | Sidewalk | 초록색 (0, 255, 0) | 보도: 보행자 보도 및 인도 |

### 2.2 파이프라인 개요

Road Topology 시스템은 다음과 같은 단계를 거쳐 동작합니다:

```
입력 영상
    |
    v
차량 객체 탐지 (YOLOv8)
    |
    v
차량 추적 (ByteTrack)
    |
    v
궤적 추출 및 평활화
    |
    v
의사 레이블 생성 (궤적 -> 마스크)
    |
    v
신뢰도 맵 생성 및 가중치 적용
    |
    v
신뢰도 맵이 포함된 학습 데이터셋
    |
    v
세그멘테이션 모델 학습 (SegFormer/Mask2Former)
    |
    v
최종 세그멘테이션 모델
    |
    v
추론 수행 (이미지/영상/배치)
```

**주요 단계 설명:**

- **차량 탐지**: YOLOv8을 사용하여 도로 위 차량을 감지
- **차량 추적**: ByteTrack 알고리즘으로 프레임 간 차량 추적
- **궤적 분석**: 추적된 차량의 이동 경로를 수학적으로 분석
- **의사 레이블**: 궤적으로부터 자동 생성된 세그멘테이션 마스크
- **신뢰도 맵**: 각 픽셀의 신뢰도를 수치로 표현
- **모델 학습**: 신뢰도 맵으로 가중치를 조정한 지도 학습
- **추론**: 학습된 모델을 새로운 데이터에 적용

### 2.3 용어 정리

자주 사용되는 주요 용어를 설명합니다:

| 용어 | 정의 | 예시 |
|------|------|------|
| **의사 레이블** | 자동 생성되는 학습 레이블 (수동 주석 대체) | 차량 궤적에서 자동 생성된 마스크 |
| **궤적** | 차량이 시간에 따라 이동하는 경로 | 5초간 차량이 이동한 선 |
| **신뢰도 맵** | 각 픽셀 분류의 확실성을 나타내는 값 (0~1) | 0.9 = 매우 확실, 0.3 = 불확실 |
| **추론** | 학습된 모델이 새 데이터를 예측하는 과정 | 입력 이미지 → 모델 → 세그멘테이션 마스크 |
| **세그멘테이션** | 이미지의 각 픽셀을 클래스로 분류하는 작업 | 도로/차선/보도 인식 |
| **세그멘테이션 마스크** | 픽셀 단위 클래스 정보를 담은 이미지 | H×W 크기의 정수 배열 |
| **배치 크기** | 한 번에 처리하는 샘플 개수 | batch_size=8은 8개 이미지 동시 처리 |
| **에폭** | 전체 학습 데이터를 1회 학습하는 주기 | 100 에폭 = 데이터 100회 반복 |
| **학습률** | 모델 가중치를 조정하는 속도 | 작을수록 천천히, 클수록 빠르게 학습 |
| **체크포인트** | 학습 중 모델 상태를 저장한 파일 | epoch_50.pth, best_model.pth |

---

## 3. 빠른 시작 (5분 가이드)

이 섹션에서는 사전 학습 모델을 사용하여 이미지에 대한 도로 세그멘테이션을 5분 안에 수행합니다.

### 3.1 3단계 추론 튜토리얼

#### 단계 1: 모델 다운로드

먼저 사전 학습된 모델을 다운로드합니다. 첫 실행 시에만 필요합니다.

```bash
# 스크립트를 사용한 모델 다운로드 (권장)
python scripts/download_models.py --sam vit_h

# 또는 YOLOv8 모델도 함께 준비 (자동 다운로드됨)
python -c "from ultralytics import YOLO; YOLO('yolov8m.pt')"
```

다운로드된 모델은 다음 경로에 저장됩니다:

```
./models/sam/sam_vit_h_4b8939.pth        # SAM ViT-H 모델 (약 2.5 GB)
./models/sam/sam_vit_b_01ec64.pth        # SAM ViT-B 모델 (약 375 MB)
~/.local/share/ultralytics/              # YOLOv8 모델 (약 49 MB, 자동 다운로드)
```

**참고:**
- SAM 모델은 프로젝트의 `./models/sam/` 디렉토리에 저장됩니다
- YOLOv8 모델은 자동으로 다운로드되어 ultralytics 캐시 디렉토리에 저장됩니다
- **추론용 세그멘테이션 모델** (`./outputs/best_model.pth`)은 학습 완료 후 생성되는 파일입니다 (다음 단계 참조)

#### 단계 2: 이미지 추론 실행

이제 학습된 모델을 사용하여 도로 이미지를 분석합니다.
실제 테스트 도로 이미지는 `./output_dir3/` 디렉토리에 있으며, 총 88개의 JPG 파일이 포함되어 있습니다 (예: 01CT000000123.jpg, 01CT000000134.jpg 등):

**중요:** 아래 명령어의 `--model` 경로는 **학습 완료 후** 생성되는 모델 파일입니다.
학습을 완료하면 `./outputs/` 디렉토리에 `best_model.pth` 파일이 생성됩니다.

```bash
# 기본 추론 (결과 화면에 표시)
road-topo infer image \
  --model ./outputs/model_v1/best_model.pth \
  --input ./output_dir3/01CT000000123.jpg

# 결과를 파일로 저장 (화면 표시 없음)
road-topo infer image \
  --model ./models/sam/sam_vit_b_01ec64.pth \
  --input ./output_dir3/01CT000000123.jpg \
  --output segmentation_result1.png \
  --no-visualize

**출력 설명:**
- 컬러 마스크: 각 픽셀이 해당 클래스 색상으로 표현
- 신뢰도: 각 분할 영역의 신뢰도 점수
- 처리 시간: GPU 추론 속도 (약 50-200ms)

#### 단계 3: 결과 확인 및 분석

추론이 완료되면 생성된 파일을 확인합니다:

```bash
# 결과 이미지 확인
ls -lh segmentation_result.png

# 메타데이터 확인 (선택사항)
file segmentation_result.png
identify segmentation_result.png     # ImageMagick 필요
```

결과 해석:
- **검은색 영역**: Background (배경)
- **회색 영역**: Road (도로 표면)
- **흰색 영역**: Lane (차선 마킹)
- **노란색 영역**: Crosswalk (횡단보도)
- **초록색 영역**: Sidewalk (보도)

**배치 추론 (여러 이미지 동시 처리):**

```bash
# output_dir3 디렉토리의 모든 JPG 이미지 추론 (88개)
road-topo infer batch \
  --model ./outputs/model_v1/best_model.pth \
  --input ./output_dir3 \
  --output ./results \
  --pattern "*.jpg"

# 결과 확인
ls -lh ./results/
```

**영상 추론 (비디오 파일):**

```bash
# 비디오 파일 전체 추론
road-topo infer video \
  --model ./outputs/model_v1/best_model.pth \
  --input test_video.mp4 \
  --output result_video.mp4

# 진행률 모니터링
# 약 30fps 처리 속도 (GPU 성능에 따라 변동)
```

---

## 4. 주요 워크플로우

이 섹션에서는 Road Topology Segmentation 시스템의 세 가지 핵심 워크플로우를 다룹니다. 각 워크플로우는 독립적으로 사용할 수 있으며, 함께 연동되어 완전한 파이프라인을 구성합니다.

### 4.1 워크플로우 A: 영상에서 학습 데이터 생성

이 워크플로우는 CCTV 영상으로부터 자동으로 학습 데이터를 생성합니다. 차량 궤적 추적을 기반으로 의사 레이블을 생성하며, 신뢰도 맵을 함께 제공합니다.

#### 기본 명령어

```bash
road-topo pseudolabel generate \
  --video /path/to/video.mp4 \
  --output ./data/pseudo_labels \
  --trajectory-width 50 \
  --threshold 0.1
```

#### 파라미터 설명

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `--video, -v` | 필수 | 입력 영상 파일 경로 |
| `--output, -o` | 필수 | 출력 디렉토리 경로 |
| `--config, -c` | None | 설정 YAML 파일 (선택사항) |
| `--trajectory-width` | 50 | 궤적 마스크의 너비(픽셀) |
| `--threshold` | 0.1 | 마스크 생성을 위한 신뢰도 임계값 |
| `--frame-skip` | 0 | 프레임 스킵 간격 (0 = 모든 프레임 처리) |

#### 출력 디렉토리 구조

```
./data/pseudo_labels/
├── masks/
│   ├── frame_0000.png       # 의미 있는 분할 마스크
│   ├── frame_0001.png       # 클래스별 색상: 0=배경, 1=도로, 2=차선, 3=횡단보도, 4=보도
│   └── ...
├── confidence/
│   ├── frame_0000.npy       # 신뢰도 맵 (0.0~1.0)
│   ├── frame_0001.npy       # 궤적 기반 확신 점수
│   └── ...
├── images/
│   ├── frame_0000.jpg       # 추출된 원본 프레임
│   ├── frame_0001.jpg
│   └── ...
├── trajectories.json        # 차량 궤적 데이터
└── metadata.json            # 생성 메타데이터 (커버리지, 프레임 수 등)
```

#### 실전 예제

**예제 1: 기본 의사 레이블 생성**

```bash
road-topo pseudolabel generate \
  --video video.mp4 \
  --output ./data/pseudo_labels
```

**출력 예:**
```
[blue]Processing video: video.mp4[/blue]
Generating pseudo-labels... ████████████████████ 100%
[green]Saved pseudo-labels to ./data/pseudo_labels[/green]
  Trajectories: 247
  Coverage: 68.5%
```

**예제 2: 프레임 스킵으로 빠른 처리**

매 두 번째 프레임마다만 처리하여 처리 속도 2배 향상:

```bash
road-topo pseudolabel generate \
  --video video.mp4 \
  --output ./data/pseudo_labels \
  --frame-skip 1 \
  --trajectory-width 40
```

**예제 3: 커스텀 설정 파일 사용**

```bash
road-topo pseudolabel generate \
  --video video.mp4 \
  --output ./data/pseudo_labels \
  --config configs/custom_detection.yaml \
  --trajectory-width 60 \
  --threshold 0.15
```

#### 시각화

생성된 의사 레이블을 확인하려면:

```bash
# 마스크 표시
road-topo pseudolabel visualize \
  --mask ./data/pseudo_labels/masks/frame_0000.png

# 원본 프레임과 함께 시각화 후 저장
road-topo pseudolabel visualize \
  --mask ./data/pseudo_labels/masks/frame_0000.png \
  --video video.mp4 \
  --output visualization_with_overlay.png
```

#### 워크플로우 A 팁과 트러블슈팅

**Tip 1: 신뢰도 임계값 조정**
- 높은 threshold (0.3 이상): 더 보수적인 레이블, 고품질 데이터
- 낮은 threshold (0.05 이하): 더 많은 데이터, 하지만 노이즈 증가

**Tip 2: 궤적 너비 최적화**
- 좁은 차선 도로: trajectory_width = 30~40
- 넓은 다중 차선 도로: trajectory_width = 60~80

**Issue: 커버리지가 너무 낮음 (< 30%)**
```bash
# 해결책: 차량 감지 임계값 낮추기
# configs/detection.yaml 수정
model:
  confidence_threshold: 0.3  # 기본값 0.5에서 감소
```

---

### 4.2 워크플로우 B: 모델 학습

의사 레이블 또는 수동 주석 데이터로부터 의미 있는 분할 모델을 학습합니다.

#### 데이터셋 디렉토리 구조

학습을 시작하기 전에 다음 구조로 데이터셋을 구성해야 합니다:

```
data/dataset/
├── train/
│   ├── images/              # 훈련용 이미지
│   │   ├── image_0001.jpg
│   │   ├── image_0002.jpg
│   │   └── ... (약 800장)
│   ├── masks/               # 클래스 인덱스 마스크
│   │   ├── image_0001.png   # 값: 0~4 (클래스 ID)
│   │   ├── image_0002.png
│   │   └── ...
│   └── confidence/          # 신뢰도 가중치 (선택사항)
│       ├── image_0001.npy   # 값: 0.0~1.0
│       └── ...
├── val/
│   ├── images/              # 검증용 이미지 (약 100장)
│   ├── masks/
│   └── confidence/
└── test/
    ├── images/              # 테스트용 이미지 (약 100장)
    ├── masks/
    └── confidence/
```

#### 데이터셋 준비 명령어

의사 레이블을 데이터셋 구조로 변환:

```bash
# 디렉토리 생성
mkdir -p ./data/dataset/{train,val,test}/{images,masks,confidence}

# 의사 레이블에서 파일 복사
cp ./data/pseudo_labels/images/* ./data/dataset/train/images/
cp ./data/pseudo_labels/masks/* ./data/dataset/train/masks/
cp ./data/pseudo_labels/confidence/* ./data/dataset/train/confidence/
```

또는 Python API 사용:

```python
from road_topology.segmentation.dataset import RoadTopologyDataset
from road_topology.segmentation.transforms import get_train_transforms, get_val_transforms

# 자동으로 train/val/test 분할 (80/10/10)
train_ds, val_ds, test_ds = RoadTopologyDataset.create_splits(
    root="./data/dataset",
    train_ratio=0.8,
    val_ratio=0.1,
    seed=42,
    transforms_train=get_train_transforms((512, 512)),
    transforms_val=get_val_transforms((512, 512)),
)

print(f"Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)}")
```

#### 학습 명령어

```bash
road-topo train run \
  --config configs/training.yaml \
  --data ./data/dataset \
  --output ./outputs/model_v1 \
  --epochs 100 \
  --batch-size 8
```

#### 학습 파라미터 테이블

| 파라미터 | 기본값 | 범위 | 설명 |
|---------|-------|------|------|
| `--epochs` | 100 | 50~200 | 훈련 에포크 수 |
| `--batch-size` | 8 | 4~32 | 배치 크기 (GPU 메모리에 따라) |
| `--lr` (learning rate) | 6e-5 | 1e-5~1e-3 | 학습률 (설정 파일에서) |
| `--weight-decay` | 0.01 | 0.001~0.1 | L2 정규화 계수 |

#### 실전 예제

**예제 1: 기본 학습**

```bash
road-topo train run \
  --config configs/training.yaml \
  --data ./data/dataset
```

**예상 출력:**
```
[blue]Training with config: configs/training.yaml[/blue]
  Device: cuda
  Epochs: 100
  Batch size: 8

Epoch 1/100: loss=0.523, val_loss=0.456, val_mIoU=0.542
Epoch 2/100: loss=0.412, val_loss=0.398, val_mIoU=0.601
...
Epoch 100/100: loss=0.201, val_loss=0.215, val_mIoU=0.754

[green]Training complete! Best mIoU: 0.7892[/green]
```

**예제 2: 하이퍼파라미터 오버라이드**

```bash
road-topo train run \
  --config configs/training.yaml \
  --data ./data/dataset \
  --epochs 150 \
  --batch-size 16 \
  --output ./outputs/experiment_v2
```

**예제 3: 체크포인트에서 재개**

이전 학습을 중단한 경우 계속 진행:

```bash
road-topo train run \
  --config configs/training.yaml \
  --data ./data/dataset \
  --resume ./outputs/checkpoint_epoch_50.pth \
  --output ./outputs/model_v1_resumed
```

#### 학습 결과 확인

학습이 완료되면 다음 파일들이 생성됩니다:

```
./outputs/model_v1/
├── best_model.pth              # 최고 mIoU 성능 모델
├── last_model.pth              # 마지막 에포크 모델
├── checkpoint_epoch_10.pth      # 에포크별 체크포인트
├── checkpoint_epoch_20.pth
├── ...
├── training_log.json           # 훈련 이력 (손실, mIoU 등)
├── metrics.json                # 최종 평가 지표
└── config.yaml                 # 사용된 설정
```

#### 검증

학습된 모델을 검증 데이터셋에서 평가:

```bash
road-topo train validate \
  --model ./outputs/model_v1/best_model.pth \
  --data ./data/dataset \
  --batch-size 16
```

**출력 예:**
```
[blue]Validating model: ./outputs/model_v1/best_model.pth[/blue]

[bold]Results:[/bold]
  mIoU: 0.7892
  road: 0.8234
  lane: 0.7891
  crosswalk: 0.6543
  sidewalk: 0.7234
```

#### 워크플로우 B 팁과 트러블슈팅

**Tip 1: 학습률 스케줄**
```yaml
# configs/training.yaml
scheduler:
  type: "polynomial"
  power: 0.9    # 다항식 감소

# 또는 코사인 어닐링
scheduler:
  type: "cosine"
  T_max: 100
```

**Tip 2: 메모리 부족 해결**
```bash
# GPU 메모리 부족 시
road-topo train run \
  --config configs/training.yaml \
  --data ./data/dataset \
  --batch-size 4              # 배치 크기 감소
  --epochs 200                # 에포크 증가로 보정
```

**Issue: mIoU가 개선되지 않음**

**해결책:** 학습률 감소
```yaml
# configs/training.yaml
optimizer:
  lr: 3e-5      # 기본값 6e-5에서 감소
  weight_decay: 0.01
```

**Issue: 메모리 부족 (OOM)**

**해결책:** 배치 크기 또는 이미지 크기 감소
```yaml
# configs/training.yaml 수정
training:
  batch_size: 4           # 8에서 감소

# 또는 segmentation.yaml 수정
input:
  image_size: [384, 384]  # 512에서 감소
```

---

### 4.3 워크플로우 C: 추론

학습된 모델로 새로운 이미지나 영상에 대해 의미 있는 분할을 수행합니다.

#### 이미지 추론

**명령어:**

```bash
road-topo infer image \
  --model ./outputs/model_v1/best_model.pth \
  --input ./output_dir3/01CT000000123.jpg \
  --output result.png \
  --visualize/--no-visualize \
  --overlay/--no-overlay
```

**파라미터:**

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `--model, -m` | 필수 | 모델 체크포인트 경로 |
| `--input, -i` | 필수 | 입력 이미지 파일 |
| `--output, -o` | None | 결과 저장 경로 (선택사항) |
| `--visualize` | True | 결과 창에 표시 |
| `--overlay` | True | 원본 이미지와 오버레이 |

**예제:**

```bash
# 기본 추론 (결과 창에 표시)
road-topo infer image \
  --model ./outputs/best_model.pth \
  --input street_photo.jpg

# 결과 저장 (표시 없음)
road-topo infer image \
  --model ./outputs/best_model.pth \
  --input street_photo.jpg \
  --output segmentation_result.png \
  --no-visualize

# 오버레이 없이 순수 마스크만 저장
road-topo infer image \
  --model ./outputs/best_model.pth \
  --input street_photo.jpg \
  --output mask_only.png \
  --no-overlay
```

#### 비디오 추론

**명령어:**

```bash
road-topo infer video \
  --model ./outputs/model_v1/best_model.pth \
  --input input_video.mp4 \
  --output output_video.mp4 \
  --fps 30 \
  --overlay/--no-overlay
```

**파라미터:**

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `--model, -m` | 필수 | 모델 체크포인트 경로 |
| `--input, -i` | 필수 | 입력 비디오 파일 |
| `--output, -o` | 필수 | 출력 비디오 경로 |
| `--fps` | None | 출력 FPS (기본: 입력과 동일) |
| `--overlay` | True | 원본 프레임과 오버레이 |

**예제:**

```bash
# 동일 프레임 레이트로 처리
road-topo infer video \
  --model ./outputs/best_model.pth \
  --input input.mp4 \
  --output output.mp4

# 커스텀 FPS로 처리 (빠른 재생)
road-topo infer video \
  --model ./outputs/best_model.pth \
  --input input.mp4 \
  --output output_fast.mp4 \
  --fps 15

# 마스크만 저장 (오버레이 없음)
road-topo infer video \
  --model ./outputs/best_model.pth \
  --input input.mp4 \
  --output masks_only.mp4 \
  --no-overlay
```

**처리 시간:** 1080p 영상 기준 약 1초/프레임 (GPU 사용 시)

#### 배치 추론

**명령어:**

```bash
road-topo infer batch \
  --model ./outputs/model_v1/best_model.pth \
  --input ./output_dir3 \
  --output ./results \
  --pattern "*.jpg" \
  --batch-size 8
```

**파라미터:**

| 파라미터 | 기본값 | 설명 |
|---------|-------|------|
| `--model, -m` | 필수 | 모델 체크포인트 경로 |
| `--input, -i` | 필수 | 입력 이미지 디렉토리 |
| `--output, -o` | 필수 | 결과 저장 디렉토리 |
| `--pattern, -p` | `*.jpg` | 파일 매칭 패턴 |
| `--batch-size, -b` | 8 | 배치 처리 크기 |

**예제:**

```bash
# JPG 파일 처리 (모두)
road-topo infer batch \
  --model ./outputs/best_model.pth \
  --input ./output_dir3 \
  --output ./results

# PNG 파일 처리 (배치 크기 16)
road-topo infer batch \
  --model ./outputs/best_model.pth \
  --input ./output_dir3 \
  --output ./results \
  --pattern "*.png" \
  --batch-size 16

# 특정 패턴 매칭
road-topo infer batch \
  --model ./outputs/best_model.pth \
  --input ./output_dir3 \
  --output ./results \
  --pattern "road_*.jpg"
```

**출력 구조:**
```
./results/
├── image_001_seg.png     # 원본 + 분할 오버레이
├── image_002_seg.png
├── image_003_seg.png
└── ...
```

#### 추론 성능 최적화

**배치 크기 영향:**

| 배치 크기 | GPU 메모리 | 처리 속도 | 권장 상황 |
|---------|----------|---------|---------|
| 1 | 2 GB | 기준 | 메모리 부족 환경 |
| 8 | 8 GB | 4~6배 빠름 | 표준 추론 |
| 16 | 14 GB | 7~9배 빠름 | 고속 처리 |
| 32 | 24+ GB | 10~12배 빠름 | 대량 처리 |

#### 추론 결과 해석

**클래스별 색상 표시:**

```
클래스 0 (배경):     검정색 (0, 0, 0)       - 건물, 나무, 하늘
클래스 1 (도로):     회색 (128, 128, 128)   - 포장도로
클래스 2 (차선):     흰색 (255, 255, 255)   - 차선 표시
클래스 3 (횡단보도): 노란색 (255, 255, 0)   - 보행자 횡단보도
클래스 4 (보도):     녹색 (0, 255, 0)       - 인도
```

#### 워크플로우 C 팁과 트러블슈팅

**Tip 1: 추론 속도 향상**
- 배치 크기 증가 (GPU 메모리 허용 범위)
- 더 작은 모델 사용 (segformer-b0 또는 b1)
- 모델 양자화 또는 ONNX 변환

**Tip 2: 메모리 사용량 감소**
- 배치 크기 1 사용
- 이미지 크기 감소 (configs/segmentation.yaml)
- 모델 양자화

---

### 4.4 전체 파이프라인 통합

세 워크플로우를 함께 실행하는 완전한 예:

```bash
#!/bin/bash

# 설정
VIDEO="data/video.mp4"
DATASET="data/dataset"
OUTPUT="outputs"

# Step 1: 영상에서 의사 레이블 생성 (워크플로우 A)
echo "Step 1: 의사 레이블 생성..."
road-topo pseudolabel generate \
  --video "$VIDEO" \
  --output "$OUTPUT/pseudo_labels" \
  --trajectory-width 50

# Step 2: 데이터셋 구조 준비
echo "Step 2: 데이터셋 준비..."
mkdir -p "$DATASET"/{train,val,test}/{images,masks,confidence}
cp "$OUTPUT/pseudo_labels/images"/* "$DATASET/train/images/"
cp "$OUTPUT/pseudo_labels/masks"/* "$DATASET/train/masks/"
cp "$OUTPUT/pseudo_labels/confidence"/* "$DATASET/train/confidence/"

# Step 3: 모델 학습 (워크플로우 B)
echo "Step 3: 모델 학습..."
road-topo train run \
  --config configs/training.yaml \
  --data "$DATASET" \
  --output "$OUTPUT/model" \
  --epochs 100 \
  --batch-size 8

# Step 4: 검증
echo "Step 4: 모델 검증..."
road-topo train validate \
  --model "$OUTPUT/model/best_model.pth" \
  --data "$DATASET"

# Step 5: 추론 (워크플로우 C)
echo "Step 5: 새로운 이미지에 대한 추론..."
road-topo infer batch \
  --model "$OUTPUT/model/best_model.pth" \
  --input "$OUTPUT/output_dir3" \
  --output "$OUTPUT/segmentations"

echo "완료! 결과는 $OUTPUT에 저장됨"
```

이 스크립트를 실행하려면:

```bash
chmod +x pipeline.sh
./pipeline.sh
```

---

**요점 정리:**
- **워크플로우 A**: 영상 → 의사 레이블 데이터
- **워크플로우 B**: 의사 레이블 → 학습된 모델
- **워크플로우 C**: 학습된 모델 → 새로운 데이터에 대한 예측

각 워크플로우는 독립적으로 사용 가능하며, 함께 조합하여 완전한 시스템을 구축할 수 있습니다.

---

## 5. CLI 명령어 레퍼런스

Road Topology Segmentation의 모든 기능은 `road-topo` 커맨드라인 인터페이스를 통해 접근할 수 있습니다. 이 섹션은 각 명령어의 사용법, 옵션, 그리고 실제 예시를 제공합니다.

### 5.1 전역 명령어 (Global Commands)

프로젝트의 기본 정보를 확인하는 전역 명령어입니다.

#### road-topo version

패키지 버전 정보를 표시합니다.

```bash
$ road-topo version
road-topology version 0.1.0
```

**사용 사례:**
- 설치된 패키지의 버전 확인
- 의존성 호환성 검증

---

#### road-topo info

시스템과 설정 정보를 표시합니다.

```bash
$ road-topo info
Road Topology Segmentation
PyTorch version: 2.1.0
CUDA available: True
CUDA device: NVIDIA A100-PCIE-40GB
```

**출력 정보:**
- PyTorch 버전
- CUDA 사용 가능 여부
- 사용 가능한 GPU 장치명

---

### 5.2 pseudolabel 명령어 (Pseudo-Label Commands)

비디오에서 자동으로 의사 라벨(pseudo-label)을 생성하는 명령어입니다.

#### road-topo pseudolabel generate

비디오의 차량 궤적으로부터 의사 라벨을 생성합니다.

**기본 문법:**
```bash
road-topo pseudolabel generate \
  --video VIDEO_PATH \
  --output OUTPUT_DIR \
  [옵션들]
```

**옵션 테이블:**

| 옵션 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `--video, -v` | Path | 예 | - | 입력 비디오 파일 경로 |
| `--output, -o` | Path | 예 | - | 출력 디렉토리 경로 |
| `--config, -c` | Path | 아니오 | None | 설정 YAML 파일 경로 |
| `--trajectory-width` | int | 아니오 | 50 | 궤적 마스크 너비 (픽셀 단위) |
| `--threshold` | float | 아니오 | 0.1 | 신뢰도 임계값 (0.0~1.0) |
| `--frame-skip` | int | 아니오 | 0 | 처리 간 프레임 건너뛰기 (0=모두 처리) |

**사용 예시:**

```bash
# 기본 의사 라벨 생성
road-topo pseudolabel generate \
  --video video.mp4 \
  --output ./pseudo_labels

# 더 좁은 궤적과 높은 임계값 사용
road-topo pseudolabel generate \
  --video video.mp4 \
  --output ./pseudo_labels \
  --trajectory-width 30 \
  --threshold 0.3

# 프레임 건너뛰기로 빠른 처리
road-topo pseudolabel generate \
  --video video.mp4 \
  --output ./pseudo_labels \
  --frame-skip 2 \
  --trajectory-width 40

# 커스텀 설정 파일 사용
road-topo pseudolabel generate \
  --video video.mp4 \
  --output ./pseudo_labels \
  --config configs/custom.yaml
```

**출력 구조:**
```
pseudo_labels/
├── masks/                 # 의미 분할 마스크 (클래스 인덱스, 0-4)
│   ├── frame_0000.png
│   ├── frame_0001.png
│   └── ...
├── confidence/            # 신뢰도 맵 (0.0-1.0 값)
│   ├── frame_0000.npy
│   ├── frame_0001.npy
│   └── ...
├── images/                # 추출된 프레임 (원본 이미지)
│   ├── frame_0000.jpg
│   ├── frame_0001.jpg
│   └── ...
├── trajectories.json      # 차량 궤적 데이터
└── metadata.json          # 생성 메타데이터
```

---

#### road-topo pseudolabel visualize

의사 라벨 마스크를 시각화합니다 (선택 사항: 비디오 프레임 오버레이).

**기본 문법:**
```bash
road-topo pseudolabel visualize \
  --mask MASK_PATH \
  [옵션들]
```

**옵션 테이블:**

| 옵션 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `--mask, -m` | Path | 예 | - | 마스크 PNG 파일 경로 |
| `--video, -v` | Path | 아니오 | None | 오버레이용 비디오 파일 경로 |
| `--output, -o` | Path | 아니오 | None | 시각화 저장 경로 (없으면 화면 표시) |

**사용 예시:**

```bash
# 마스크만 화면에 표시
road-topo pseudolabel visualize \
  --mask ./pseudo_labels/masks/frame_0000.png

# 비디오 프레임과 오버레이하여 표시
road-topo pseudolabel visualize \
  --mask ./pseudo_labels/masks/frame_0000.png \
  --video video.mp4

# 시각화 결과를 파일로 저장
road-topo pseudolabel visualize \
  --mask ./pseudo_labels/masks/frame_0000.png \
  --video video.mp4 \
  --output visualization_result.png
```

---

### 5.3 train 명령어 (Training Commands)

의미 분할 모델을 학습하는 명령어입니다.

#### road-topo train run

준비된 데이터셋에서 분할 모델을 학습합니다.

**기본 문법:**
```bash
road-topo train run \
  --config CONFIG_FILE \
  --data DATA_DIR \
  [옵션들]
```

**옵션 테이블:**

| 옵션 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `--config, -c` | Path | 예 | - | 학습 설정 YAML 파일 |
| `--data, -d` | Path | 예 | - | 데이터셋 루트 디렉토리 |
| `--output, -o` | Path | 아니오 | `./outputs` | 체크포인트 및 로그 저장 디렉토리 |
| `--epochs` | int | 아니오 | None | 학습 에포크 수 (설정 파일 값 무시) |
| `--batch-size, -b` | int | 아니오 | None | 배치 크기 (설정 파일 값 무시) |
| `--resume, -r` | Path | 아니오 | None | 체크포인트에서 재개 |

**사용 예시:**

```bash
# 기본 설정으로 학습
road-topo train run \
  --config configs/training.yaml \
  --data ./dataset

# 하이퍼파라미터 오버라이드
road-topo train run \
  --config configs/training.yaml \
  --data ./dataset \
  --epochs 50 \
  --batch-size 16 \
  --output ./outputs/experiment_v2

# 이전 체크포인트에서 재개
road-topo train run \
  --config configs/training.yaml \
  --data ./dataset \
  --resume ./outputs/checkpoint_epoch_30.pth

# 큰 배치 크기로 빠른 학습 (GPU 메모리 충분한 경우)
road-topo train run \
  --config configs/training.yaml \
  --data ./dataset \
  --batch-size 32 \
  --epochs 100
```

**출력 구조:**
```
outputs/
├── best_model.pth             # mIoU 기준 최상 체크포인트
├── last_model.pth             # 마지막 에포크 체크포인트
├── checkpoint_epoch_10.pth    # 주기적 체크포인트
├── checkpoint_epoch_20.pth
├── training_log.json          # 학습 이력
├── metrics.json               # 최종 평가 지표
└── config.yaml                # 사용된 설정 파일
```

---

#### road-topo train validate

학습된 모델을 검증 데이터셋에서 평가합니다.

**기본 문법:**
```bash
road-topo train validate \
  --model MODEL_CHECKPOINT \
  --data DATA_DIR \
  [옵션들]
```

**옵션 테이블:**

| 옵션 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `--model, -m` | Path | 예 | - | 모델 체크포인트 파일 경로 |
| `--data, -d` | Path | 예 | - | 검증 데이터셋 디렉토리 |
| `--batch-size, -b` | int | 아니오 | 8 | 검증 배치 크기 |

**사용 예시:**

```bash
# 최상 모델 검증
road-topo train validate \
  --model ./outputs/best_model.pth \
  --data ./dataset

# 더 큰 배치 크기로 검증 (빠른 처리)
road-topo train validate \
  --model ./outputs/best_model.pth \
  --data ./dataset \
  --batch-size 32

# 다양한 체크포인트 검증
road-topo train validate \
  --model ./outputs/checkpoint_epoch_50.pth \
  --data ./dataset
```

**출력 예시:**
```
Validating model: ./outputs/best_model.pth

Results:
  mIoU: 0.6542
  background: 0.8234
  road: 0.7234
  lane: 0.6891
  crosswalk: 0.5543
  sidewalk: 0.6234
```

---

### 5.4 infer 명령어 (Inference Commands)

학습된 모델로 추론을 수행하는 명령어입니다.

#### road-topo infer image

단일 이미지에 대해 추론을 실행합니다.

**기본 문법:**
```bash
road-topo infer image \
  --model MODEL_CHECKPOINT \
  --input IMAGE_PATH \
  [옵션들]
```

**옵션 테이블:**

| 옵션 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `--model, -m` | Path | 예 | - | 모델 체크포인트 경로 |
| `--input, -i` | Path | 예 | - | 입력 이미지 파일 |
| `--output, -o` | Path | 아니오 | None | 결과 저장 경로 |
| `--visualize` | bool | 아니오 | True | 결과를 화면에 표시 |
| `--overlay` | bool | 아니오 | True | 원본 이미지에 오버레이 |

**사용 예시:**

```bash
# 기본 추론 및 시각화
road-topo infer image \
  --model ./outputs/best_model.pth \
  --input ./output_dir3/01CT000000123.jpg

# 결과를 파일로 저장 (화면 표시 안 함)
road-topo infer image \
  --model ./outputs/best_model.pth \
  --input ./output_dir3/01CT000000123.jpg \
  --output result.png \
  --no-visualize

# 오버레이 없이 순수 분할 마스크 생성
road-topo infer image \
  --model ./outputs/best_model.pth \
  --input ./output_dir3/01CT000000123.jpg \
  --output mask_only.png \
  --no-overlay

# 화면 표시만 (저장하지 않음)
road-topo infer image \
  --model ./outputs/best_model.pth \
  --input ./output_dir3/01CT000000123.jpg \
  --no-overlay
```

**출력 예시:**
```
Running inference on: ./output_dir3/01CT000000123.jpg

Detected classes:
  Class 0 (background): 45.23%
  Class 1 (road): 32.15%
  Class 2 (lane): 12.34%
  Class 3 (crosswalk): 5.28%
  Class 4 (sidewalk): 5.00%
```

---

#### road-topo infer video

비디오 파일에 대해 추론을 실행합니다.

**기본 문법:**
```bash
road-topo infer video \
  --model MODEL_CHECKPOINT \
  --input VIDEO_PATH \
  --output OUTPUT_PATH \
  [옵션들]
```

**옵션 테이블:**

| 옵션 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `--model, -m` | Path | 예 | - | 모델 체크포인트 경로 |
| `--input, -i` | Path | 예 | - | 입력 비디오 파일 |
| `--output, -o` | Path | 예 | - | 출력 비디오 경로 |
| `--fps` | int | 아니오 | None | 출력 FPS (미설정 시 입력 FPS 사용) |
| `--overlay` | bool | 아니오 | True | 분할 결과를 프레임에 오버레이 |

**사용 예시:**

```bash
# 기본 비디오 처리 (입력과 동일한 FPS)
road-topo infer video \
  --model ./outputs/best_model.pth \
  --input input.mp4 \
  --output output.mp4

# 커스텀 FPS로 처리 (빠른 재생)
road-topo infer video \
  --model ./outputs/best_model.pth \
  --input input.mp4 \
  --output output.mp4 \
  --fps 15

# 느린 FPS로 상세 분석
road-topo infer video \
  --model ./outputs/best_model.pth \
  --input input.mp4 \
  --output output.mp4 \
  --fps 5

# 오버레이 없이 순수 마스크만 생성
road-topo infer video \
  --model ./outputs/best_model.pth \
  --input input.mp4 \
  --output masks.mp4 \
  --no-overlay
```

**진행 상황 출력:**
```
Processing video: input.mp4
Processing frames... 100%|████████████| 1000/1000
Saved output video to output.mp4
  Processed 1000 frames
```

---

#### road-topo infer batch

디렉토리의 이미지 배치에 대해 추론을 실행합니다.

**기본 문법:**
```bash
road-topo infer batch \
  --model MODEL_CHECKPOINT \
  --input INPUT_DIR \
  --output OUTPUT_DIR \
  [옵션들]
```

**옵션 테이블:**

| 옵션 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `--model, -m` | Path | 예 | - | 모델 체크포인트 경로 |
| `--input, -i` | Path | 예 | - | 입력 이미지 디렉토리 |
| `--output, -o` | Path | 예 | - | 결과 저장 디렉토리 |
| `--pattern, -p` | str | 아니오 | `*.jpg` | 파일 매칭 글롭 패턴 |
| `--batch-size, -b` | int | 아니오 | 8 | 배치 처리 크기 |

**사용 예시:**

```bash
# 모든 JPEG 파일 처리
road-topo infer batch \
  --model ./outputs/best_model.pth \
  --input ./output_dir3 \
  --output ./results

# PNG 파일 처리 (큰 배치 크기)
road-topo infer batch \
  --model ./outputs/best_model.pth \
  --input ./output_dir3 \
  --output ./results \
  --pattern "*.png" \
  --batch-size 16

# 특정 패턴 매칭
road-topo infer batch \
  --model ./outputs/best_model.pth \
  --input ./output_dir3 \
  --output ./results \
  --pattern "road_*.jpg"

# 여러 확장자 처리 (for 루프 활용)
for ext in jpg png; do
  road-topo infer batch \
    --model ./outputs/best_model.pth \
    --input ./output_dir3 \
    --output ./results \
    --pattern "*.$ext"
done
```

**진행 상황 출력:**
```
Processing images in: ./output_dir3
Found 88 images
Processing images... 100%|████████████| 88/88
Processed 88 images
Results saved to ./results
```

---

### 5.5 evaluate 명령어 (Evaluation Commands)

모델 성능을 평가하는 명령어입니다.

#### road-topo evaluate metrics

테스트 세트에서 모델 성능을 평가합니다.

**기본 문법:**
```bash
road-topo evaluate metrics \
  --model MODEL_CHECKPOINT \
  --data DATA_DIR \
  [옵션들]
```

**옵션 테이블:**

| 옵션 | 타입 | 필수 | 기본값 | 설명 |
|------|------|------|--------|------|
| `--model, -m` | Path | 예 | - | 모델 체크포인트 경로 |
| `--data, -d` | Path | 예 | - | 데이터셋 디렉토리 |
| `--split, -s` | str | 아니오 | `val` | 데이터셋 분할 (train/val/test) |
| `--batch-size, -b` | int | 아니오 | 8 | 평가 배치 크기 |
| `--output, -o` | Path | 아니오 | None | 결과를 JSON으로 저장 (미설정 시 화면만 출력) |

---

## 6. 데이터셋 가이드

### 6.1 지원 데이터셋 비교표

도로 위상 분할 프로젝트는 다음 공개 데이터셋을 지원합니다:

| 데이터셋 | 이미지 수 | 인증 필요 | 권장 용도 | 다운로드 시간 |
|----------|----------|----------|----------|----------|
| **KITTI Road** | 289 | 불필요 | 테스트/프로토타입 | 5분 |
| **BDD100K** | 10,000+ | Kaggle 계정 | 본격 학습/프로덕션 | 2-4시간 |
| **Cityscapes** | 5,000 | 계정 필요 | 고품질 도시 장면 | 1-2시간 |

**선택 기준:**
- **빠른 테스트**: KITTI Road 선택 (289개 이미지, 즉시 사용 가능)
- **프로덕션 모델**: BDD100K 또는 Cityscapes (충분한 데이터, 다양한 조건)
- **크로스워크 감지**: BDD100K (횡단보도 주석 포함)

### 6.2 KITTI Road 다운로드 및 준비

KITTI는 인증이 필요 없어 가장 빠르게 시작할 수 있습니다.

#### 자동 다운로드

```bash
# 다운로드 및 자동 처리
python scripts/download_datasets.py kitti --output-dir ./data

# 출력 구조
./data/kitti/
├── train/
│   ├── images/          # 학습 이미지 (180장)
│   └── masks/           # 학습 마스크
└── test/
    ├── images/          # 테스트 이미지 (109장)
    └── masks/           # 테스트 마스크
```

#### 검증

```bash
# 데이터셋 통계 확인
python scripts/download_datasets.py demo --samples 5

# 또는 Python에서 확인
python -c "
from pathlib import Path
kitti_dir = Path('./data/kitti/train/images')
print(f'이미지 수: {len(list(kitti_dir.glob(\"*.png\")))}')
"
```

### 6.3 BDD100K 다운로드 (프로덕션)

BDD100K는 가장 대규모이며 다양한 조건의 데이터를 포함합니다.

#### 준비물

- Kaggle 계정 (https://www.kaggle.com)
- Kaggle API 키

#### 설정 단계

```bash
# 1. Kaggle CLI 설치
pip install kaggle

# 2. Kaggle 계정 설정
# kaggle.com/account → "Create New API Token" 클릭
# ~/.kaggle/kaggle.json에 저장
mkdir -p ~/.kaggle
mv kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# 3. 다운로드 (선택 사항: 자동 다운로드)
python scripts/download_datasets.py bdd100k --output-dir ./data
```

### 6.4 Cityscapes 다운로드 (고품질)

Cityscapes는 고품질 도시 장면 데이터셋입니다.

#### 등록 및 다운로드

```bash
# 1. 계정 등록
# https://www.cityscapes-dataset.com/register/

# 2. 로그인 후 다운로드
# https://www.cityscapes-dataset.com/login/

# 필요한 파일:
# - leftImg8bit_trainvaltest.zip (이미지)
# - gtFine_trainvaltest.zip (주석)

# 3. 추출
unzip leftImg8bit_trainvaltest.zip -d ./data/cityscapes
unzip gtFine_trainvaltest.zip -d ./data/cityscapes

# 4. 처리
python scripts/download_datasets.py cityscapes --output-dir ./data
```

### 6.5 커스텀 데이터셋 준비

자신의 데이터셋을 준비하는 방법:

#### 디렉토리 구조

```
data/custom_dataset/
├── train/
│   ├── images/              # RGB 이미지 (JPG/PNG)
│   │   ├── image_001.jpg
│   │   ├── image_002.jpg
│   │   └── ...
│   └── masks/               # 마스크 (8비트 그레이스케일 PNG)
│       ├── image_001.png    # 클래스 ID: 0-4
│       ├── image_002.png
│       └── ...
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

#### 마스크 형식 상세

```
마스크 파일 요구사항:
- 형식: PNG (8비트 그레이스케일)
- 해상도: 이미지와 동일
- 클래스 ID:
  0: Background (건물, 나무, 하늘 등)
  1: Road (도로 표면)
  2: Lane (차선 표시)
  3: Crosswalk (횡단보도)
  4: Sidewalk (보도)

예시:
image.jpg (1280x720)
image.png (1280x720, 값: 0-4)
```

### 6.6 데이터셋 검증

데이터셋이 올바르게 준비되었는지 확인:

```bash
# 데이터셋 무결성 확인
python -c "
from pathlib import Path
import cv2
from road_topology.segmentation.dataset import RoadTopologyDataset

# 데이터셋 로드
ds = RoadTopologyDataset('./data/kitti', 'train')
print(f'샘플 수: {len(ds)}')

# 첫 번째 샘플 확인
sample = ds[0]
print(f'이미지 형태: {sample[\"image\"].shape}')
print(f'마스크 형태: {sample[\"mask\"].shape}')

# 클래스 분포 확인
import numpy as np
mask = sample['mask'].numpy()
unique, counts = np.unique(mask, return_counts=True)
print(f'클래스: {unique}')
print(f'픽셀 수: {counts}')
"
```

**출력 예시:**
```
샘플 수: 180
이미지 형태: torch.Size([3, 512, 512])
마스크 형태: torch.Size([512, 512])
클래스: [0 1 2 3 4]
픽셀 수: [150000 120000  80000  20000  30000]
```

---

## 7. 설정 파일 가이드

### 7.1 설정 파일 목록

프로젝트의 모든 설정은 `configs/` 디렉토리에 위치합니다:

```
configs/
├── default.yaml          # 기본 프로젝트 설정
├── detection.yaml        # YOLOv8 감지 설정
├── tracking.yaml         # ByteTrack 추적 설정
├── segmentation.yaml     # SegFormer 분할 설정
├── training.yaml         # 학습 하이퍼파라미터
├── sam.yaml              # SAM 영점사격 설정
└── cvat.yaml             # CVAT 통합 설정
```

### 7.2 주요 파라미터 10개

#### 1. 디바이스 (Device)

```yaml
project:
  device: "auto"  # "auto", "cuda", "mps", "cpu"
```

- `auto`: 자동 감지 (CUDA > MPS > CPU 우선순위)
- `cuda`: NVIDIA GPU (권장, 빠름)
- `mps`: Apple Metal (Mac M1/M2/M3)
- `cpu`: CPU만 사용 (느림, 메모리 필요)

**선택 기준:** `auto` 사용 시 시스템에서 가장 빠른 디바이스 자동 선택

#### 2. 배치 크기 (Batch Size)

```yaml
training:
  batch_size: 8  # 범위: 1-32
```

| 배치 크기 | GPU 메모리 | 속도 | 안정성 |
|----------|----------|------|--------|
| 4 | 6 GB | 느림 | 매우 안정 |
| 8 | 10 GB | 중간 | 권장 |
| 16 | 18 GB | 빠름 | 덜 안정 |
| 32 | 32+ GB | 매우 빠름 | 불안정 |

**OOM 오류 해결:** 배치 크기를 4로 줄임

#### 3. 학습률 (Learning Rate)

```yaml
optimizer:
  lr: 6e-5  # 범위: 1e-6 ~ 1e-3
```

**가이드라인:**
- 사전학습 모델 미세조정: `1e-5 ~ 5e-5` (권장: 6e-5)
- 처음부터 학습: `1e-4 ~ 1e-3`
- 손실이 감소하지 않음: 학습률을 50% 감소

#### 4. 에포크 (Epochs)

```yaml
training:
  epochs: 100  # 범위: 10-500
```

**일반적인 값:**
- 빠른 테스트: 10-20
- 일반 학습: 100
- 최적화: 150-200

#### 5. 이미지 크기 (Image Size)

```yaml
segmentation:
  image_size: [512, 512]  # 높이, 너비
```

| 크기 | GPU 메모리 | 정확도 | 추론 속도 |
|-----|----------|--------|----------|
| 256x256 | 4 GB | 낮음 | 빠름 (100ms) |
| 512x512 | 8 GB | 중간 | 중간 (200ms) |
| 768x768 | 16 GB | 높음 | 느림 (400ms) |

**권장:** 512x512 (정확도와 속도 균형)

#### 6. 모델 백본 (Model Backbone)

```yaml
segmentation:
  backbone: "nvidia/segformer-b2-finetuned-cityscapes-1024-1024"
```

**옵션 (빠름 → 정확도 높음):**
- `b0`: 매우 빠름, 낮은 정확도
- `b1`: 빠름, 중간 정확도
- `b2`: **권장**, 균형있음
- `b3`: 느림, 높은 정확도
- `b4, b5`: 매우 느림, 최고 정확도

#### 7. 손실 함수 가중치 (Loss Weights)

```yaml
loss:
  ce_weight: 0.5      # Cross-entropy
  dice_weight: 0.5    # Dice loss
```

**조정:**
- 클래스 불균형: CE 가중치 증가 (0.7)
- 경계 정확도: Dice 가중치 증가 (0.7)

#### 8. 증강 수준 (Augmentation Level)

```yaml
augmentation:
  level: "medium"  # "light", "medium", "heavy"
```

**레벨별 변환:**
- `light`: 작은 회전(-5°~5°), 수평 뒤집기
- `medium`: 회전(-15°~15°), 자르기, 흐림
- `heavy`: 강한 회전(-30°~30°), 탄성 변형

**과적합 증가 시:** heavy로 설정

#### 9. 조기 종료 인내심 (Early Stopping Patience)

```yaml
checkpointing:
  early_stopping_patience: 10
```

- 10 에포크 동안 개선 없으면 학습 중단
- **증가:** 더 오래 학습하고 싶을 때
- **감소:** 빠르게 중단하고 싶을 때

#### 10. 혼합 정밀도 (Mixed Precision)

```yaml
training:
  mixed_precision: true  # FP16/FP32 자동 혼합
```

**효과:**
- 속도: 30% 더 빠름
- 메모리: 50% 절약
- 정확도: 거의 영향 없음

---

### 7.3 환경 변수 (ROAD_TOPO_*)

YAML 파일 대신 환경 변수로 설정 가능:

```bash
# 디바이스 설정
export ROAD_TOPO_PROJECT__DEVICE=cuda

# 로깅 레벨
export ROAD_TOPO_LOGGING__LEVEL=DEBUG

# 학습 파라미터
export ROAD_TOPO_TRAINING__EPOCHS=50
export ROAD_TOPO_TRAINING__BATCH_SIZE=16
export ROAD_TOPO_OPTIMIZER__LR=3e-5

# 코드에서 사용
from road_topology.core.config import load_config
config = load_config("configs/training.yaml")
print(config.training.epochs)  # 환경 변수 우선
```

**환경 변수 우선순위:** 환경 변수 > YAML 파일

---

## 8. FAQ

### 일반 질문

**Q1: GPU 없이도 사용 가능한가요?**

A: 네, 가능하지만 매우 느립니다. CPU에서 실행:

```bash
# 설정 파일 수정
sed -i 's/device: "cuda"/device: "cpu"/' configs/default.yaml

# 또는 환경 변수
export ROAD_TOPO_PROJECT__DEVICE=cpu
```

추천: GPU 사용 (10-50배 빠름)

---

**Q2: 최적 배치 크기는 얼마인가요?**

A: GPU 메모리에 따라 다릅니다:

```
GPU 메모리        권장 배치 크기
─────────────────────────
4 GB             1-2
6 GB             4
8 GB             8 (권장)
12 GB            16
16+ GB           32
```

**테스트 방법:**
```python
import torch
torch.cuda.empty_cache()
print(f"사용 가능 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

---

**Q3: 학습에 얼마나 걸리나요?**

A: 하드웨어와 데이터에 따라 다릅니다:

| 조건 | 시간 |
|------|------|
| KITTI (289장), RTX 3080, 100 에포크 | 1-2시간 |
| KITTI, CPU | 8-12시간 |
| BDD100K (10K), RTX 3080, 100 에포크 | 20-30시간 |

**빠르게 하는 방법:**
```yaml
training:
  batch_size: 16        # 증가
  num_workers: 8        # 증가
  mixed_precision: true # 활성화
```

---

**Q4: 모델을 저장하고 불러올 수 있나요?**

A: 네, 체크포인트로 저장 및 불러오기:

```python
from road_topology.segmentation.models import SegFormerModel

# 저장
model = SegFormerModel(...)
model.save("model_v1.pth")

# 불러오기
loaded_model = SegFormerModel.load("model_v1.pth")

# 재개
road-topo train run \
  --config configs/training.yaml \
  --data ./data/dataset \
  --resume ./outputs/checkpoint_epoch_50.pth
```

---

**Q5: 여러 GPU를 사용할 수 있나요?**

A: 현재 단일 GPU 지원입니다. 다중 GPU는 향후 지원 예정입니다.

**대안:** 배치 크기 증가 또는 모델 양자화

---

### 데이터 관련 질문

**Q6: 자신의 비디오에서 의사 레이블을 생성할 수 있나요?**

A: 네, 차량 궤적을 마스크로 변환합니다:

```bash
road-topo pseudolabel generate \
  --video my_video.mp4 \
  --output ./pseudo_labels \
  --trajectory-width 50
```

**주의:** 차량이 충분히 이동해야 합니다 (최소 10 프레임)

---

**Q7: 이미지가 마스크와 다른 크기입니다. 어떻게 하나요?**

A: 자동으로 크기 조정되지만, 수동으로 확인하세요:

```python
import cv2
from pathlib import Path

img_dir = Path("data/dataset/train/images")
mask_dir = Path("data/dataset/train/masks")

for img_path in img_dir.glob("*.jpg"):
    img = cv2.imread(str(img_path))
    mask = cv2.imread(str(mask_dir / f"{img_path.stem}.png"), 0)

    if img.shape[:2] != mask.shape:
        print(f"불일치: {img_path.name}")
        # 마스크 크기 조정
        mask_resized = cv2.resize(mask, (img.shape[1], img.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(str(mask_dir / f"{img_path.stem}.png"), mask_resized)
```

---

**Q8: 데이터셋을 결합할 수 있나요?**

A: 네, 여러 데이터셋을 함께 사용:

```python
from torch.utils.data import ConcatDataset, DataLoader
from road_topology.segmentation.dataset import RoadTopologyDataset

kitti = RoadTopologyDataset("data/kitti", "train")
bdd = RoadTopologyDataset("data/bdd100k", "train")
cityscapes = RoadTopologyDataset("data/cityscapes", "train")

combined = ConcatDataset([kitti, bdd, cityscapes])
loader = DataLoader(combined, batch_size=8, shuffle=True)

print(f"총 샘플: {len(combined)}")
```

---

### 모델/학습 질문

**Q9: 검증 mIoU가 개선되지 않습니다. 어떻게 해야 하나요?**

A: 여러 가지 시도:

```yaml
# 1. 학습률 감소
optimizer:
  lr: 3e-5  # 6e-5에서 감소

# 2. 증강 강화
augmentation:
  level: "heavy"

# 3. 더 오래 학습
training:
  epochs: 200  # 100에서 증가

# 4. 조기 종료 해제
checkpointing:
  early_stopping_patience: 20  # 10에서 증가
```

---

**Q10: 모델 추론이 너무 느립니다. 어떻게 빠르게 할 수 있나요?**

A: 여러 가지 최적화:

```python
# 1. 더 작은 모델 사용
model = SegFormerModel(
    backbone="nvidia/segformer-b0-finetuned-cityscapes-1024-1024"
)

# 2. ONNX로 내보내기
import torch
torch.onnx.export(model, dummy_input, "model.onnx", opset_version=11)

# 3. PyTorch 2.0 컴파일 (Python 3.10+)
import torch
model = torch.compile(model)

# 4. 양자화
model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
```

---

### 배포 관련 질문

**Q11: 모델을 프로덕션에 배포하려면?**

A: 단계별 배포:

```bash
# 1. ONNX 내보내기
python -c "
from road_topology.segmentation.models import SegFormerModel
model = SegFormerModel.load('best_model.pth')
import torch
torch.onnx.export(model, torch.randn(1,3,512,512), 'model.onnx')
"

# 2. ONNX 검증
python -c "
import onnx
onnx_model = onnx.load('model.onnx')
onnx.checker.check_model(onnx_model)
print('ONNX 모델 유효!')
"

# 3. Docker로 패킹
# (Dockerfile 작성 후)
docker build -t road-topology:v1 .
docker run -p 5000:5000 road-topology:v1
```

---

**Q12: REST API로 노출할 수 있나요?**

A: Flask/FastAPI로 쉽게 구성:

```python
from fastapi import FastAPI, File
from PIL import Image
import torch
from road_topology.segmentation.models import SegFormerModel

app = FastAPI()
model = SegFormerModel.load("best_model.pth")

@app.post("/predict")
async def predict(file: bytes = File(...)):
    image = Image.open(file)
    image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1)

    with torch.no_grad():
        pred = model.predict(image_tensor)

    return {"prediction": pred.tolist()}
```

실행: `uvicorn app:app --host 0.0.0.0 --port 5000`

---

## 9. 문제 해결

### 설치 관련 오류

#### 오류 1: CUDA Not Found (CUDA를 찾을 수 없음)

**증상:**
```
torch.cuda.is_available() → False
```

**원인:** PyTorch가 CPU 버전으로 설치됨

**해결책:**

```bash
# 1. 기존 PyTorch 제거
pip uninstall torch torchvision

# 2. CUDA 버전 확인
nvidia-smi

# 3. 올바른 버전 설치
# CUDA 11.8의 경우:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. 검증
python -c "import torch; print(torch.cuda.is_available())"
```

---

#### 오류 2: Out of Memory During Installation (설치 중 메모리 부족)

**증상:**
```
OSError: [Errno 28] No space left on device
MemoryError: ...
```

**해결책:**

```bash
# 1. 더 효율적인 패키지 관리자 사용 (uv)
pip install uv
uv pip install -e .

# 2. 또는 패키지별로 설치
pip install torch torchvision  # 먼저 큰 패키지
pip install -q segment-anything
pip install -q ultralytics

# 3. 캐시 정리
pip cache purge
```

---

### 학습 관련 오류

#### 오류 3: GPU Out of Memory (GPU 메모리 부족)

**증상:**
```
RuntimeError: CUDA out of memory. Tried to allocate 2.50 GB on ...
```

**원인:** 배치 크기가 너무 크거나 이미지 해상도 높음

**해결책:**

```yaml
# 방법 1: 배치 크기 감소
training:
  batch_size: 4  # 8에서 감소

# 방법 2: 이미지 크기 감소
segmentation:
  image_size: [384, 384]  # 512에서 감소

# 방법 3: 그래디언트 누적
training:
  batch_size: 8
  gradient_accumulation: 4  # 효과적 배치 크기 = 32

# 방법 4: 주의 깊은 모델 선택
segmentation:
  backbone: "nvidia/segformer-b0-finetuned-cityscapes-1024-1024"  # b2에서 b0으로
```

---

### 일반 디버깅 팁

#### 로깅 활성화

```bash
# DEBUG 모드로 상세 로그 출력
export ROAD_TOPO_LOGGING__LEVEL=DEBUG

road-topo train run \
  --config configs/training.yaml \
  --data ./dataset

# 또는 직접 설정
# configs/default.yaml
logging:
  level: "DEBUG"
  file: "./debug.log"
```

#### 시스템 정보 확인

```bash
road-topo info

# 출력:
# Road Topology Segmentation
# PyTorch version: 2.1.0
# CUDA available: True
# CUDA device: NVIDIA RTX 3080
# CUDA version: 11.8
```

#### 메모리 모니터링

```bash
# GPU 메모리 실시간 모니터링
watch -n 1 nvidia-smi

# CPU/메모리 모니터링 (대안)
htop
```

---

## 부록 A. 영문 매뉴얼 참조 가이드

Road Topology Segmentation 프로젝트의 상세한 영문 매뉴얼과 문서 참조 가이드입니다.

### A.1 주요 문서 위치

| 문서 | 경로 | 용도 |
|------|------|------|
| **메인 USER MANUAL** | `/docs/USER_MANUAL.md` | 상세한 설명, 예제 코드 |
| **프로젝트 README** | `/README.md` | 빠른 시작, 프로젝트 개요 |
| **QUICKSTART** | `/scripts/QUICKSTART.md` | 즉시 시작하기 |
| **클래스 매핑** | `/scripts/CLASS_MAPPINGS.md` | 데이터셋 클래스 변환 규칙 |
| **설정 파일** | `/configs/` | 각 컴포넌트별 기본 설정 |

### A.2 API 문서

#### Core Types
```python
from road_topology.core.types import Trajectory, BoundingBox

# 문서 위치: USER_MANUAL.md → "API Reference" → "Type Definitions"
help(Trajectory)
help(BoundingBox)
```

#### Segmentation Models
```python
from road_topology.segmentation.models import SegFormerModel

# 문서 위치: USER_MANUAL.md → "API Reference" → "SegFormerModel"
help(SegFormerModel)
```

#### Dataset Loading
```python
from road_topology.segmentation.dataset import RoadTopologyDataset

# 문서 위치: USER_MANUAL.md → "API Reference" → "RoadTopologyDataset"
help(RoadTopologyDataset)
```

### A.3 고급 주제 매핑

| 관심 분야 | USER_MANUAL 섹션 |
|----------|-----------------|
| CVAT 휴먼-인-더-루프 | "Advanced Topics" → "Human-in-the-Loop" |
| 이전 학습 (Transfer Learning) | "Advanced Topics" → "Transfer Learning from Cityscapes" |
| 모델 앙상블 | "Advanced Topics" → "Model Ensemble" |
| 커스텀 손실 함수 | "Advanced Topics" → "Custom Loss Functions" |
| ONNX 내보내기 | "Inference Guide" → "ONNX Export for Production" |
| 모델 양자화 | "Inference Guide" → "Model Quantization" |

### A.4 추가 리소스

**공식 문서:**
- PyTorch: https://pytorch.org/docs/
- Segment Anything: https://github.com/facebookresearch/segment-anything
- YOLOv8: https://github.com/ultralytics/ultralytics
- SegFormer: https://huggingface.co/nvidia/segformer-b2-finetuned-cityscapes-1024-1024

**데이터셋:**
- BDD100K: https://www.bdd100k.com/
- Cityscapes: https://www.cityscapes-dataset.com/
- KITTI: http://www.cvlibs.net/datasets/kitti/

---

## 라이선스

MIT License - 자유롭게 사용, 수정, 배포 가능합니다.

## 지원

문제 발생 시:
1. 영문 문서의 "Troubleshooting" 섹션 확인
2. 프로젝트 GitHub Issues 검색
3. 커뮤니티 포럼에서 질문
