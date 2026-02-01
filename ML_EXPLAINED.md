# Road Topology ML 가이드

Road Topology 프로젝트의 기계학습 아키텍처와 구현을 설명하는 종합 가이드입니다.

## 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [아키텍처 다이어그램](#아키텍처-다이어그램)
3. [핵심 컴포넌트 상세 분석](#핵심-컴포넌트-상세-분석)
4. "왜?" 섹션](#왜-섹션)
5. [자주 하는 실수](#자주-하는-실수)
6. [학습 로드맵](#학습-로드맵)

---

## 프로젝트 개요

### 프로젝트 목표

자율주행 자동차의 도로 인지를 위해 **차선(lane)과 횡단보도(crosswalk)를 인스턴스 레벨에서 분리**하는 것이 목표입니다.

### 기술 스택

| 컴포넌트 | 기술 |
|----------|------|
| **Framework** | PyTorch + HuggingFace Transformers |
| **Base Model** | SegFormer-B5 (84.7M params) |
| **Backbone** | Cityscapes Pretrained |
| **Task** | Semantic Segmentation + Instance Embeddings |

### 핵심 특징

- **Dual-Head Architecture**: 의미론적 분할(semantic) + 인스턴스 임베딩(instance)의 동시 학습
- **6 Semantic Classes**: background, road, lane_marking, crosswalk, sidewalk, lane_boundary
- **Instance Separation**: Discriminative Loss를 통한 차선 인스턴스 분리
- **Left-to-Right Ordering**: 클러스터링된 차선을 좌측부터 우측 순서로 정렬

### 데이터 구조

```
root/
  {split}/  (train, val, test)
    images/
      image_001.jpg
    semantic_masks/
      image_001.png          # 클래스 ID (0-5)
    instance_masks/
      image_001.png          # 인스턴스 ID (0, 1, 2, ...)
```

---

## 아키텍처 다이어그램

### 1. 전체 파이프라인 (Data → Model → Loss → Training)

```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT DATA                               │
│  RGB Image (H, W, 3) + Semantic Mask + Instance Mask        │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │      DATA AUGMENTATION (Albumentations)│
        │  - Random Crop/Resize                  │
        │  - Horizontal Flip                     │
        │  - Color Jitter                        │
        └────────────────────┬───────────────────┘
                             │
                             ▼
        ┌────────────────────────────────────────┐
        │    NORMALIZATION & TENSORIZATION       │
        │  - RGB 값을 [0, 1] 범위로 정규화      │
        │  - (B, H, W, 3) → (B, 3, H, W)       │
        └────────────────────┬───────────────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
         ┌──────────────────┐  ┌──────────────────┐
         │  SEMANTIC PATH   │  │  INSTANCE PATH   │
         │  (Class Pred)    │  │  (Embeddings)    │
         └──────────────────┘  └──────────────────┘
                    │                 │
                    ▼                 ▼
         ┌──────────────────┐  ┌──────────────────┐
         │  SEMANTIC LOSS   │  │ DISCRIMINATIVE   │
         │  CE + Dice       │  │ LOSS             │
         │  (1.0 weight)    │  │ (0.5 weight)     │
         └────────┬─────────┘  └────────┬─────────┘
                  │                     │
                  └──────────┬──────────┘
                             │
                             ▼
                   ┌─────────────────────┐
                   │  TOTAL LOSS         │
                   │ semantic_loss +     │
                   │ 0.5 * instance_loss │
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │  BACKWARD PASS      │
                   │  (Mixed Precision)  │
                   │  (Gradient Clipping)│
                   └──────────┬──────────┘
                              │
                              ▼
                   ┌─────────────────────┐
                   │ OPTIMIZER STEP      │
                   │ (AdamW)             │
                   │ lr_scale: encoder*0.1
                   │           decoder*1.0
                   └─────────────────────┘
```

### 2. SegFormer Dual-Head 아키텍처

```
                    INPUT IMAGE
                   (B, 3, H, W)
                        │
                        ▼
        ┌───────────────────────────────┐
        │  SegFormer ENCODER            │
        │  (Shared Backbone - B5)       │
        │                               │
        │  Stage 1: 1/4 resolution      │
        │  Stage 2: 1/8 resolution      │
        │  Stage 3: 1/16 resolution     │
        │  Stage 4: 1/32 resolution     │
        │                               │
        │  Multi-scale Feature Maps     │
        │  C1, C2, C3, C4               │
        └───────────────────────────────┘
                        │
                        ├──────────────────────────┐
                        │                          │
                        ▼                          ▼
        ┌──────────────────────────┐  ┌──────────────────────────┐
        │ SEMANTIC HEAD            │  │ INSTANCE HEAD            │
        │                          │  │                          │
        │ 1. MLP Decoder           │  │ 1. MLP Decoder           │
        │    (Multi-scale fusion)  │  │    (Multi-scale fusion)  │
        │                          │  │                          │
        │ 2. Upsample 4x           │  │ 2. Upsample 4x           │
        │    (1/4 → 1/1)           │  │    (1/4 → 1/1)           │
        │                          │  │                          │
        │ 3. Softmax (6 classes)   │  │ 3. L2 Normalize          │
        │    Output: (B, 6, H, W)  │  │    (32-dim embeddings)   │
        │                          │  │    Output: (B, 32, H, W) │
        └──────────────────────────┘  └──────────────────────────┘
                        │                          │
                        ▼                          ▼
        ┌──────────────────────────┐  ┌──────────────────────────┐
        │ SEMANTIC LOGITS          │  │ EMBEDDING VECTORS        │
        │ Shape: (B, 6, H, W)      │  │ Shape: (B, 32, H, W)     │
        │ Values: class logits     │  │ Values: normalized 32-d  │
        │        (0=background)    │  │        feature vectors   │
        │        (1=road)          │  │                          │
        │        (2=lane_marking)  │  │ Purpose: Instance        │
        │        (3=crosswalk)     │  │ Separation via           │
        │        (4=sidewalk)      │  │ Clustering               │
        │        (5=boundary)      │  │                          │
        └──────────────────────────┘  └──────────────────────────┘
```

### 3. Loss Function Composition (손실 함수 조합)

```
                        TOTAL LOSS
                            │
                ┌───────────┬┴────────────┐
                │           │            │
                ▼           ▼            ▼
        ┌──────────────┐ ┌──────────────┐
        │ SEMANTIC LOSS│ │ INSTANCE     │
        │ (Weight: 1.0)│ │ LOSS         │
        │              │ │ (Weight: 0.5)│
        └──────┬───────┘ └──────┬───────┘
               │                │
               ▼                ▼
        ┌──────────────┐ ┌──────────────────────┐
        │ CrossEntropy │ │ DISCRIMINATIVE LOSS  │
        │   + Dice     │ │                      │
        │              │ │ 3 Components:        │
        │ CrossEntropy │ │                      │
        │ (CE):        │ │ 1. Variance Term     │
        │ - Penalizes  │ │    (Pull pixels to   │
        │   incorrect  │ │     instance mean)   │
        │   class pred │ │    delta_var = 0.5   │
        │              │ │                      │
        │ Dice Loss:   │ │ 2. Distance Term     │
        │ - High when  │ │    (Push instance    │
        │   pred and   │ │     means apart)     │
        │   gt overlap │ │    delta_dist = 1.5  │
        │   poorly     │ │                      │
        │ - Addresses  │ │ 3. Regularization    │
        │   class      │ │    (Keep means near  │
        │   imbalance  │ │     origin)          │
        │              │ │    gamma = 0.001     │
        └──────────────┘ └──────────────────────┘
               │                │
               └────────┬───────┘
                        │
                        ▼
        ┌──────────────────────────────┐
        │ TOTAL LOSS =                 │
        │ 1.0 * semantic_loss +        │
        │ 0.5 * discriminative_loss    │
        └──────────────────────────────┘
```

### 4. 임베딩 공간의 인스턴스 분리 (Discriminative Loss)

```
Embedding 공간 시각화:

초기 상태 (학습 전):
  ┌─────────────────────────────────────┐
  │  O O X X                            │  O = Lane 1 pixels
  │  O O X X                            │  X = Lane 2 pixels
  │  O O X X      ← 무질서            │
  │  O O X X                            │  문제: 같은 차선의 pixels이
  │                                     │        서로 떨어져 있음
  └─────────────────────────────────────┘

  → Variance Loss (δ_var = 0.5):
    각 lane의 pixels을 lane 중심으로 모음

  → Distance Loss (δ_dist = 1.5):
    서로 다른 lane의 중심을 밀어냄

최종 상태 (학습 후):
  ┌─────────────────────────────────────┐
  │  O O     X X                        │  O가 모여있음
  │  O O     X X                        │  (variance loss 효과)
  │ O O     X X                        │
  │  O O     X X                        │  O와 X 사이 거리 증가
  │                                     │  (distance loss 효과)
  │  ∘ = Lane 1 center                 │
  │  ˣ = Lane 2 center                 │
  └─────────────────────────────────────┘
```

---

## 핵심 컴포넌트 상세 분석

### 1. Loss Functions (손실 함수) - losses.py

#### 1.1 DiceLoss

**개념**: Dice 계수에 기반한 손실 함수

```python
Dice = (2 * TP) / (2 * TP + FP + FN)
Loss = 1 - Dice
```

**왜 CrossEntropy가 아닌 Dice Loss를 함께 사용하는가?**

| 특성 | CrossEntropy | Dice Loss | 결합의 장점 |
|-----|--------------|-----------|-----------|
| 클래스 불균형 | 취약함 | 강함 | 두 가지 문제 동시 해결 |
| 경계선 포커스 | 약함 | 강함 | 세밀한 세그멘테이션 |
| 수렴 속도 | 빠름 | 느림 | 안정적이고 빠른 학습 |
| Gradient Scale | 크기 변함 | 안정적 | 일정한 학습 신호 |

**구현 분석**:

```python
# 1. Softmax 적용: 확률로 변환
pred = F.softmax(pred, dim=1)  # (B, C, H, W)

# 2. One-hot 인코딩: 타겟을 원-핫으로 변환
target_one_hot = F.one_hot(target, num_classes)  # (B, H, W, C) → (B, C, H, W)

# 3. Intersection 계산: 교집합
intersection = (pred * target_one_hot).sum(dim=(0, 2, 3))  # (C,)

# 4. Union 계산: 합집합
union = pred.sum(dim=(0, 2, 3)) + target_one_hot.sum(dim=(0, 2, 3))  # (C,)

# 5. Dice 계수 계산
dice = (2.0 * intersection + smooth) / (union + smooth)  # (C,)

# 6. Loss: 1 - Dice
loss = 1.0 - dice.mean()
```

**smooth = 1.0의 의미**: 분모가 0이 되는 것을 방지하며, 그래디언트가 매우 작은 클래스도 학습에 기여하도록 함

#### 1.2 FocalLoss

**개념**: 어려운 샘플에 더 큰 가중치를 주는 손실 함수

```python
FL(p) = -α(1-p)^γ * log(p)
       ├─ p: 정답 클래스의 확률
       ├─ γ: Focusing parameter (gamma=2.0)
       └─ α: 클래스 가중치
```

**gamma=2.0의 의미**:

```
예를 들어 어떤 샘플의 확률 p:

p=0.9 (쉬운 샘플):
  (1-0.9)^2 = 0.01
  가중치 = 0.01 * loss  ← 무시함

p=0.5 (어려운 샘플):
  (1-0.5)^2 = 0.25
  가중치 = 0.25 * loss  ← 중요하게 봄

p=0.1 (매우 어려운 샘플):
  (1-0.1)^2 = 0.81
  가중치 = 0.81 * loss  ← 매우 중요하게 봄

결과: 클래스 불균형 해결 + 어려운 샘플 포커스
```

**구현**:

```python
# 1. Cross-Entropy 계산
ce_loss = F.cross_entropy(pred, target, reduction="none")  # (B, H, W)

# 2. 확률 계산
p = torch.exp(-ce_loss)  # 범위 [0, 1]

# 3. Focal weight 적용
focal_weight = (1 - p) ** self.gamma  # gamma=2.0
focal_loss = focal_weight * ce_loss

# 4. 최종 loss
loss = focal_loss.mean()
```

**언제 사용하는가?**

- 차선 마킹(lane_marking) vs 도로 배경이 매우 불균형할 때
- 횡단보도 같은 소수 클래스를 잘 학습시키고 싶을 때

#### 1.3 DiscriminativeLoss (인스턴스 분리를 위한 손실)

**논문**: [Semantic Instance Segmentation with a Discriminative Loss Function](https://arxiv.org/abs/1708.02551)

**목표**: 같은 인스턴스 내 pixels을 모으고, 다른 인스턴스를 분리

**3가지 손실 항**:

##### (1) Variance Term (분산항, δ_var = 0.5)

목표: 각 인스턴스 내의 pixels을 인스턴스 중심으로 모음

```
L_var = Σ_i max(0, d_i - δ_var)^2

d_i: pixel i에서 자신이 속한 인스턴스 중심까지의 거리
δ_var: 여유 마진 (0.5)

예시:
- 차선 1이 embedding 공간 [0, 0]에 중심
- Pixel A: [0.2, 0.3] → d_A = 0.36 > 0.5 → 손실 = 0
- Pixel B: [0.6, 0.4] → d_B = 0.72 > 0.5 → 손실 = (0.72-0.5)^2 = 0.05
→ Pixel B를 중심으로 더 끌어당김
```

##### (2) Distance Term (거리항, δ_dist = 1.5)

목표: 서로 다른 인스턴스의 중심을 밀어냄

```
L_dist = Σ_{i<j} max(0, 2*δ_dist - ||μ_i - μ_j||)^2

μ_i, μ_j: 인스턴스 i, j의 중심
δ_dist: 최소 거리 여유 마진 (1.5)

예시:
- 차선 1 중심: [0, 0]
- 차선 2 중심: [2, 0]
- 거리 = 2.0 > 2*1.5 = 3.0? 아니오
→ 손실 = (3.0 - 2.0)^2 = 1.0
→ 두 차선의 중심을 더 멀리 밀어냄
```

##### (3) Regularization Term (정규화항, γ = 0.001)

목표: 인스턴스 중심들을 원점 근처에 유지 (embedding 공간 안정화)

```
L_reg = γ * Σ_i ||μ_i||

예시:
- 차선 1 중심: [5, 5] → ||μ|| = 7.07
- 차선 2 중심: [6, 4] → ||μ|| = 7.21
- L_reg = 0.001 * (7.07 + 7.21) = 0.0142
→ 중심들을 원점으로 향하게 끌어당김
```

**총 손실**:

```python
L_total = α * L_var + β * L_dist + γ * L_reg
                (1.0)   (1.0)    (0.001)
```

**구현 상세**:

```python
# 배치 내 각 샘플에 대해
for b in range(batch_size):
    # 1. 고유 인스턴스 찾기 (0 제외)
    unique_instances = torch.unique(batch_labels)
    unique_instances = unique_instances[unique_instances > 0]  # 배경 제외

    # 2. 각 인스턴스의 중심 계산
    instance_means = []
    for instance_id in unique_instances:
        instance_mask = batch_labels == instance_id
        instance_embeddings = batch_embeddings[instance_mask]
        mean = instance_embeddings.mean(dim=0)  # (32,)
        instance_means.append(mean)

    # 3. Variance Loss 계산
    for i, instance_id in enumerate(unique_instances):
        instance_mask = batch_labels == instance_id
        instance_embeddings = batch_embeddings[instance_mask]
        mean = instance_means[i]

        # L2 거리 계산
        distances = torch.norm(instance_embeddings - mean, p=2, dim=1)

        # Hinge loss: max(0, distance - delta_var)
        var_loss += torch.clamp(distances - self.delta_var, min=0).sum()

    # 4. Distance Loss 계산 (모든 인스턴스 쌍)
    for i in range(len(unique_instances)):
        for j in range(i+1, len(unique_instances)):
            dist = torch.norm(instance_means[i] - instance_means[j], p=2)
            # Hinge loss: max(0, 2*delta_dist - distance)
            dist_loss += torch.clamp(2*self.delta_dist - dist, min=0)

    # 5. Regularization Loss
    reg_loss = torch.norm(instance_means, p=2, dim=1).sum()
```

**하이퍼파라미터 선택의 이유**:

| 파라미터 | 값 | 이유 |
|---------|-----|------|
| δ_var = 0.5 | 작은 값 | Embedding 공간의 거리 스케일에 맞춤. 32-dim에서 평균 거리가 이 정도면 적절 |
| δ_dist = 1.5 | 중간 값 | 인스턴스 간 분리 마진. 0.5의 3배로, 충분한 여유 제공 |
| γ = 0.001 | 매우 작은 값 | Regularization을 약하게 하여 주요 손실이 주도권 유지 |

---

### 2. 모델 구조 (segformer.py)

#### 2.1 SegFormer 선택 이유

**CNN vs SegFormer (Transformer-based)**:

```
┌──────────────────────────────────────────────────────────┐
│ CNN (e.g., ResNet)                   │ SegFormer (Transformer)
├──────────────────────────────────────────────────────────┤
│ 수용장(receptive field): 피라미드    │ 수용장: Global (처음부터)
│ 로컬 특징 먼저 → 글로벌 특징        │ 글로벌 + 로컬 특징 동시
│                                      │
│ 파라미터: 매우 많음                  │ 파라미터: 더 효율적
│ 계산: 느림                           │ 계산: 더 빠름
│ 먼 픽셀 의존성: 약함                 │ 먼 픽셀 의존성: 강함
│                                      │
│ 차선: 긴 선형 구조 ← 문제            │ 차선 추적: 매우 우수
│ (로컬 정보만으로 전체 그림 어려움)   │ (전체 문맥 이해)
└──────────────────────────────────────────────────────────┘
```

**SegFormer-B5 사양**:

```
Layer Name        | Depth | Hidden Dim | Params
─────────────────────────────────────────────────
Efficient (ViT)   | 3     | 64         | 12M
Efficient         | 8     | 128        | 20M
Efficient         | 27    | 320        | 45M
Efficient         | 3     | 512        | 7M
─────────────────────────────────────────────
Total             |       |            | 84.7M
```

**Cityscapes 사전학습의 이점**:

- Cityscapes: 도시 가로 이미지 + 30개 클래스 세분화
- 도로, 차선, 보도 같은 로드 토폴로지 학습 완료
- Fine-tuning만으로도 빠른 수렴

#### 2.2 모델 구조 코드 분석

```python
class SegFormerModel(nn.Module):
    BACKBONES = {
        "b0": "nvidia/segformer-b0-...",  # 3.7M
        "b1": "nvidia/segformer-b1-...",  # 13.2M
        "b2": "nvidia/segformer-b2-...",  # 24.7M
        "b3": "nvidia/segformer-b3-...",  # 47.3M
        "b4": "nvidia/segformer-b4-...",  # 62.0M
        "b5": "nvidia/segformer-b5-...",  # 84.7M (사용)
    }

    def __init__(self, backbone="b5", num_classes=6, pretrained=True):
        # Cityscapes 사전학습 가중치 로드
        self.model = SegformerForSemanticSegmentation.from_pretrained(
            backbone_name,
            num_labels=num_classes,
            ignore_mismatched_sizes=True  # 클래스 수 변경 시 자동 조정
        )
```

**ignore_mismatched_sizes=True의 의미**:

```
원본 모델: Cityscapes 19 클래스
우리 모델: 6 클래스

without ignore_mismatched_sizes:
  → 에러: shape mismatch (19 vs 6)

with ignore_mismatched_sizes=True:
  → 분류 헤드만 초기화하고 로드
  → 인코더 가중치는 모두 유지
  → 효율적인 전이학습!
```

#### 2.3 get_trainable_params() - 차등 학습률

```python
def get_trainable_params(self):
    encoder_params = []  # backbone
    decoder_params = []  # head

    for name, param in self.model.named_parameters():
        if "encoder" in name:
            encoder_params.append(param)
        else:
            decoder_params.append(param)

    return [
        {"params": encoder_params, "lr_scale": 0.1},   # 느린 학습
        {"params": decoder_params, "lr_scale": 1.0},   # 빠른 학습
    ]
```

**왜 encoder에 0.1배 학습률을 적용하는가?**

```
Cityscapes 사전학습 가중치 상태:
  encoder: 매우 좋은 상태 (이미 학습됨)
  decoder: 우리 데이터셋에 맞게 초기화 (새로 학습 필요)

만약 같은 lr로 학습하면:
  → encoder의 좋은 가중치도 빠르게 변경 ← 재앙!
  → catastrophic forgetting 발생

차등 학습률:
  encoder lr = base_lr * 0.1  ← 미세 조정만 함
  decoder lr = base_lr * 1.0  ← 강력하게 학습

결과:
  ✓ Cityscapes 사전학습 지식 보존
  ✓ 새 데이터셋에만 적응
  ✓ 빠른 수렴 + 높은 정확도
```

---

### 3. 데이터셋 (dataset.py)

#### 3.1 LaneInstanceDataset 구조

```python
class LaneInstanceDataset(RoadTopologyDataset):
    """듀얼 마스크 로딩 데이터셋"""

    def __init__(self, root, split="train"):
        # 기대하는 디렉토리 구조
        self.semantic_masks_dir = root / split / "semantic_masks"
        self.instance_masks_dir = root / split / "instance_masks"
```

**Semantic Mask 형식** (cv2.IMREAD_GRAYSCALE):

```
픽셀값    | 의미
0         | background
1         | road
2         | lane_marking
3         | crosswalk
4         | sidewalk
5         | lane_boundary

예시:
┌────────────────────┐
│  00011110011000    │  값: class ID (0-5)
│  00011110011000    │  형식: uint8 (0-255)
│  11111111111111    │  단일 채널 (H, W)
│  22222222222222    │
│  33333333333333    │
└────────────────────┘
```

**Instance Mask 형식** (RGB 또는 Grayscale):

```
Grayscale 형식 (권장):
┌──────────────────────┐
│  00000000011111111111 │  값: instance ID (0, 1, 2, ...)
│  00000000011111111111 │  0 = background/non-lane
│  22222222222222222222 │  1 = lane instance 1
│  22222222222222222222 │  2 = lane instance 2
│  33333333333333333333 │  3 = lane instance 3
└──────────────────────┘

RGB 형식 (자동 변환):
┌──────────────────────────────────────┐
│  R=0  G=0  B=0    (instance 0)       │
│  R=255 G=0 B=0    (instance 1)       │
│  R=0 G=255 B=0    (instance 2)       │
│  etc...                              │
└──────────────────────────────────────┘

변환 함수:
_rgb_to_instance_ids(rgb_mask) → 각 고유 색상을 instance ID로
```

#### 3.2 __getitem__ 프로세스

```python
def __getitem__(self, idx):
    # 1. 이미지 로드 (BGR → RGB)
    image = cv2.imread(str(img_path))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    shape: (H, W, 3), range: [0, 255]

    # 2. Semantic 마스크 로드
    semantic_mask = cv2.imread(str(semantic_path), cv2.IMREAD_GRAYSCALE)
    shape: (H, W), range: [0, 5]

    # 3. Instance 마스크 로드
    instance_mask = cv2.imread(str(instance_path), cv2.IMREAD_GRAYSCALE)
    shape: (H, W), range: [0, N_instances]

    # RGB 인스턴스 마스크면 자동 변환
    if instance_mask is None:  # 컬러로 다시 시도
        instance_mask_rgb = cv2.imread(str(instance_path))
        instance_mask = self._rgb_to_instance_ids(instance_mask_rgb)

    # 4. Augmentation 적용 (Albumentations)
    if self.transforms:
        transformed = self.transforms(
            image=image,
            masks=[semantic_mask, instance_mask]  # 두 마스크 동시 처리
        )
        image = transformed["image"]
        semantic_mask = transformed["masks"][0]
        instance_mask = transformed["masks"][1]

    # 5. 텐서로 변환
    image = torch.from_numpy(image.transpose(2,0,1)).float() / 255.0
    # shape: (3, H, W), range: [0.0, 1.0]

    semantic_mask = torch.from_numpy(semantic_mask).long()
    # shape: (H, W), dtype: int64

    instance_mask = torch.from_numpy(instance_mask).long()
    # shape: (H, W), dtype: int64

    return {
        "image": image,              # (3, H, W) [0, 1]
        "semantic_mask": semantic_mask,  # (H, W) [0-5]
        "instance_mask": instance_mask,  # (H, W) [0-N]
        "image_path": str(img_path)
    }
```

**RGB to Instance IDs 변환**:

```python
@staticmethod
def _rgb_to_instance_ids(rgb_mask):
    """각 고유 색상 → unique instance ID"""
    h, w = rgb_mask.shape[:2]
    instance_mask = np.zeros((h, w), dtype=np.int32)

    # 고유 색상 찾기
    unique_colors = np.unique(rgb_mask.reshape(-1, 3), axis=0)

    for instance_id, color in enumerate(unique_colors):
        if np.all(color == 0):  # 검정색(배경) 스킵
            continue

        # 이 색상 위치 찾기
        mask = np.all(rgb_mask == color, axis=2)
        instance_mask[mask] = instance_id

    return instance_mask
```

---

### 4. 학습 루프 (trainer_lane.py)

#### 4.1 Mixed Precision Training (AMP)

**목표**: 메모리 절약 + 속도 향상

```python
self.scaler = get_grad_scaler(device)  # GradScaler 초기화

# 학습 루프에서
with autocast_context(device):  # fp16 연산 구간
    outputs = self.model(pixel_values)
    semantic_loss, semantic_losses_dict = self.semantic_criterion(...)
    instance_loss, instance_losses_dict = self.instance_criterion(...)
    total_loss = self.semantic_weight * semantic_loss + \
                 self.instance_weight * instance_loss

# Backward with scaling
if self.mixed_precision:
    self.scaler.scale(total_loss).backward()
    self.scaler.unscale_(self.optimizer)  # gradient clipping 전 필수
else:
    total_loss.backward()
```

**왜 Mixed Precision을 사용하는가?**

| 계산 부분 | 정밀도 | 크기 | 이유 |
|----------|--------|------|------|
| Forward | float16 | 작음 | 메모리 절약 |
| Backward | float16 | 작음 | 메모리 절약 |
| 손실 축적 | float32 | 큼 | 정밀도 유지 |
| 모델 가중치 | float32 | 큼 | 정밀도 유지 |

**메모리 효과**:
```
FP32 모델 크기: 84.7M params * 4 bytes = ~340 MB
FP16 모델 크기: 84.7M params * 2 bytes = ~170 MB
절약: 50%
```

#### 4.2 Gradient Clipping (그래디언트 클리핑)

**목표**: Gradient explosion 방지

```python
torch.nn.utils.clip_grad_norm_(
    self.model.parameters(),
    max_norm=1.0
)
```

**왜 필요한가?**

```
정상적인 그래디언트:
  batch 1: grad norm = 0.5
  batch 2: grad norm = 0.3
  batch 3: grad norm = 0.8
  → 정상 학습

Gradient explosion:
  batch 1: grad norm = 0.5
  batch 2: grad norm = 50.0  ← 폭발!
  batch 3: grad norm = 1000.0 ← 더 폭발!
  → 모델 발산, 손실 = NaN

해결책 - Gradient Clipping:
  max_norm = 1.0으로 설정
  batch 2: 0.5/50.0 * 1.0 = 0.01 (정규화)
  batch 3: 1000.0/1000.0 * 1.0 = 1.0 (정규화)
  → 안정적인 학습
```

#### 4.3 Early Stopping

```python
if val_loss < self.best_val_loss:
    self.best_val_loss = val_loss
    self.epochs_without_improvement = 0
    # 최적 모델 저장
    self.model.save(output_dir / "best_model.pth")
else:
    self.epochs_without_improvement += 1
    if self.epochs_without_improvement >= patience:
        logger.info("Early stopping triggered")
        break
```

**목표**: 과적합 방지 + 학습 시간 단축

#### 4.4 Differential Learning Rates

```python
optimizer = torch.optim.AdamW([
    {
        "params": encoder_params,
        "lr": base_lr * 0.1  # 느린 학습
    },
    {
        "params": decoder_params,
        "lr": base_lr * 1.0  # 정상 학습
    }
], betas=(0.9, 0.999), weight_decay=0.01)
```

**설정 값의 의미**:

| 파라미터 | 값 | 의미 |
|---------|-----|------|
| betas=(0.9, 0.999) | 표준 | β1=0.9 (momentum), β2=0.999 (RMSprop) |
| weight_decay=0.01 | 정규화 | L2 정규화로 큰 가중치 페널티 |
| lr_encoder=base_lr*0.1 | 보수적 | 사전학습 가중치 보존 |
| lr_decoder=base_lr*1.0 | 적극적 | 새 데이터셋에 맞게 학습 |

---

### 5. 후처리 (postprocess.py)

#### 5.1 LanePostProcessor 플로우

```python
def process(semantic_pred, embeddings, lane_class_id=1):
    """
    Input:
      - semantic_pred: (H, W) 클래스 ID [0-5]
      - embeddings: (D, H, W) embedding 벡터, D=32
      - lane_class_id: 1 (lane_marking)

    Output:
      - semantic_mask: (H, W)
      - lane_instances: (H, W) instance ID [0, 1, 2, ...]
      - lane_count: int
      - lane_centers: (N, 2) 차선 중심 좌표
    """

    # 1. Lane marking 픽셀 추출
    lane_mask = semantic_pred == lane_class_id
    lane_pixels = np.count_nonzero(lane_mask)

    if lane_pixels < min_pixels:
        return empty_result  # 차선 불충분

    # 2. Lane 픽셀의 embedding 추출
    lane_embeddings = embeddings[:, lane_mask].T  # (N_pixels, D)

    # 3. Embedding 클러스터링
    if self.clustering_method == "meanshift":
        from sklearn.cluster import MeanShift
        ms = MeanShift(bandwidth=0.5)
        cluster_labels = ms.fit_predict(lane_embeddings)
    else:
        from sklearn.cluster import DBSCAN
        dbscan = DBSCAN(eps=0.3, min_samples=10)
        cluster_labels = dbscan.fit_predict(lane_embeddings)

    # 4. 작은 클러스터 필터링
    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    valid_labels = unique_labels[counts >= min_pixels]

    # 5. 좌→우 순서로 정렬
    lane_centers = []
    for label in valid_labels:
        lane_mask_filtered = cluster_labels == label
        x_coords = np.where(lane_mask_filtered)[1]  # 가로 좌표
        center_x = x_coords.mean()
        lane_centers.append(center_x)

    sorted_indices = np.argsort(lane_centers)
    valid_labels_sorted = valid_labels[sorted_indices]

    # 6. Instance ID 할당 (1, 2, 3, ...)
    lane_instances = np.zeros((h, w), dtype=np.int32)
    for new_id, old_label in enumerate(valid_labels_sorted, start=1):
        # old_label인 픽셀들을 new_id로 할당
        pixel_coords = np.where(embeddings_to_pixel_map[old_label])
        lane_instances[pixel_coords] = new_id

    return {
        "semantic_mask": semantic_pred,
        "lane_instances": lane_instances,
        "lane_count": len(valid_labels_sorted),
        "lane_centers": np.array(lane_centers)[sorted_indices]
    }
```

#### 5.2 MeanShift vs DBSCAN

| 특성 | MeanShift | DBSCAN |
|-----|----------|--------|
| 클러스터 수 | 자동 결정 | 수동 지정 |
| 노이즈 처리 | -1 레이블 | -1 레이블 |
| 속도 | 중간 | 빠름 |
| 원형 클러스터 | 우수 | 유연함 |
| 고차원 | 약함 | 강함 |
| 권장 | embedding 공간 | 거리 기반 |

**Road Topology에서 MeanShift 권장 이유**:

```
Embedding 공간이 Discriminative Loss로 최적화되어
가우시안 분포 형태를 띔 → MeanShift 적합
```

---

## "왜?" 섹션

### Q1: SegFormer를 선택한 이유는?

**A**: 차선 감지는 **긴 거리 의존성(long-range dependency)**이 중요합니다.

```
차선 A: 이미지 좌측에서 우측까지 길게 뻗어있음
  ← CNN의 국소적 수용장(receptive field)만으로 추적 어려움
  ← SegFormer의 전역 주의(global attention)로 해결
```

### Q2: Dice Loss + CrossEntropy를 함께 사용하는 이유는?

**A**: 두 손실의 장점을 모두 활용:

```
CrossEntropy: 매 픽셀마다 정확한 클래스 예측 강제
Dice Loss: 예측된 마스크와 실제 마스크의 형태 유사성 강제

결합 효과:
  ✓ 정확도 향상
  ✓ 클래스 불균형 완화
  ✓ 세밀한 경계선 추출
```

### Q3: delta_var=0.5, delta_dist=1.5의 의미는?

**A**: Embedding 공간의 거리 스케일에 맞는 선택:

```
32-dim embedding의 일반적인 거리 분포:
  - 같은 인스턴스 내 거리: 0.2 ~ 0.5
  - 다른 인스턴스 간 거리: 1.0 ~ 2.5

delta_var = 0.5:
  같은 인스턴스 내 픽셀들을 중심 ±0.5 범위 내로 모음

delta_dist = 1.5:
  다른 인스턴스 중심 사이에 최소 3.0 (2*1.5) 거리 유지
  (너무 작으면 인스턴스 겹침, 너무 크면 과하게 밀어냄)
```

### Q4: 인코더 학습률을 0.1배로 하는 이유는?

**A**: Catastrophic Forgetting 방지:

```
Cityscapes에서 학습한 일반 도로 특징(edges, textures 등)
↓
우리 데이터셋으로 빠르게 재학습하면...
↓
일반 특징이 망가짐 (catastrophic forgetting)
↓
해결: 인코더는 천천히 미세조정만 함 (lr_scale=0.1)
      디코더는 새 데이터에 맞게 강하게 학습 (lr_scale=1.0)
```

### Q5: Mixed Precision Training을 사용하는 이유는?

**A**: 메모리 효율성과 속도 향상:

```
일반 FP32 학습 (84.7M params):
  - 모델 저장: ~340 MB
  - 옵티마이저 상태: ~340 MB  (Adam의 momentum, variance)
  - 활성화값: ~1.5 GB (배치)
  총: ~2.2 GB GPU 메모리

Mixed Precision 학습:
  - 계산: FP16 (활성화, 그래디언트)
  - 축적: FP32 (손실, 모델 가중치)
  메모리 절약: 약 30-40%
  속도 향상: 10-20% (GPU 연산 효율성)
```

### Q6: 왜 Instance Mask가 필요한가?

**A**: Semantic Segmentation만으로는 불충분:

```
Semantic 만으로:
┌─────────────────────────────┐
│ 2 2 2 2 2 2 2 2 2 2 2 2 2 2 │
│ 2 2 2 2 2 2 2 2 2 2 2 2 2 2 │  클래스 ID: 2 = lane_marking
│ 2 2 2 2 2 2 2 2 2 2 2 2 2 2 │  → 모든 차선이 같음
│ 2 2 2 2 2 2 2 2 2 2 2 2 2 2 │  → 개별 차선 구분 불가
└─────────────────────────────┘

Semantic + Instance로:
┌─────────────────────────────┐
│ 1 1 1 1 2 2 2 2 3 3 3 3 4 4 │  Instance ID: 개별 차선 ID
│ 1 1 1 1 2 2 2 2 3 3 3 3 4 4 │  → Lane 1, Lane 2, Lane 3, ...
│ 1 1 1 1 2 2 2 2 3 3 3 3 4 4 │  → 각 차선을 독립적으로 추적
│ 1 1 1 1 2 2 2 2 3 3 3 3 4 4 │
└─────────────────────────────┘

Instance로부터:
  ✓ 개별 차선 추적
  ✓ 차선 특성 추출 (곡률, 위치 등)
  ✓ 자율주행 경로 결정
```

---

## 자주 하는 실수

### 1. model.eval() 빼먹기

```python
# ❌ 잘못된 예
model.load_state_dict(checkpoint)
output = model(image)  # 여전히 training mode

# 문제:
# - Dropout 활성화 (노이즈 추가)
# - BatchNorm이 배치 통계 사용 (배치 크기 1일 때 부정확)
# - 결과 불안정

# ✓ 올바른 예
model.load_state_dict(checkpoint)
model.eval()  # 평가 모드로 전환
with torch.no_grad():  # 그래디언트 계산 스킵
    output = model(image)
```

### 2. Device Mismatch

```python
# ❌ 잘못된 예
model = model.to("cuda")
image = image.to("cpu")  # ← 미스매치!
output = model(image)  # RuntimeError!

# ✓ 올바른 예
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
image = image.to(device)
output = model(image)
```

### 3. 인스턴스 마스크 형식 오류

```python
# ❌ 잘못된 예 - RGB 마스크를 그대로 사용
instance_mask = cv2.imread(path)  # (H, W, 3) RGB
# 데이터셋에서:
instance_mask = torch.from_numpy(instance_mask)
# → (H, W, 3) 3개 채널이 됨 ← 잘못됨!

# ✓ 올바른 예 1 - Grayscale 로드
instance_mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  # (H, W)
instance_mask = torch.from_numpy(instance_mask).long()
# → (H, W) 단일 채널 ✓

# ✓ 올바른 예 2 - RGB에서 자동 변환
if instance_mask.ndim == 3:  # RGB
    instance_mask = dataset._rgb_to_instance_ids(instance_mask)
instance_mask = torch.from_numpy(instance_mask).long()
```

### 4. ignore_index 미처리

```python
# ❌ 잘못된 예
loss = nn.CrossEntropyLoss()
loss_val = loss(logits, semantic_mask)
# 값이 이상하게 높음? 0 class가 과하게 페널티?

# ✓ 올바른 예 - 배경 클래스 무시
loss = nn.CrossEntropyLoss(ignore_index=0)  # 배경 무시
loss_val = loss(logits, semantic_mask)

# Dice Loss도 마찬가지
dice_loss = DiceLoss(ignore_index=0)
```

### 5. 메모리 누수 - 배치 처리

```python
# ❌ 잘못된 예
all_outputs = []
for batch in dataloader:
    output = model(batch)
    all_outputs.append(output)  # 메모리 계속 쌓임
    # GPU OOM!

# ✓ 올바른 예
for batch in dataloader:
    with torch.no_grad():
        output = model(batch)
        # 바로 처리하고 버림
        loss = criterion(output)
        # del 필요 없음 (자동 해제)
```

### 6. Discriminative Loss 계산 오류

```python
# ❌ 잘못된 예
# instance_mask에 배경이 0이 아님 (1부터 시작)
instance_mask = torch.tensor([
    [1, 1, 2, 2],
    [1, 1, 2, 2]
])
# Discriminative Loss는 0을 배경으로 가정
# → 배경 픽셀 손실 계산 ← 잘못됨!

# ✓ 올바른 예
# 배경은 반드시 0으로 표현
instance_mask = torch.tensor([
    [0, 1, 1, 2, 2, 0],  # 0 = 배경
    [0, 1, 1, 2, 2, 0],
])
instance_mask = torch.where(mask_is_valid, instance_id, torch.tensor(0))
```

### 7. Augmentation Transform 불일치

```python
# ❌ 잘못된 예 - semantic과 instance mask 별도 augment
aug = A.Compose([A.HorizontalFlip()])
semantic_transformed = aug(image=image, mask=semantic_mask)["mask"]
instance_transformed = aug(image=image, mask=instance_mask)["mask"]
# 다른 seed로 실행 → 뒤틀림 불일치!

# ✓ 올바른 예 - 함께 augment
aug = A.Compose([A.HorizontalFlip()],
                 additional_targets={"mask2": "mask"})
transformed = aug(
    image=image,
    mask=semantic_mask,
    mask2=instance_mask
)
# 또는 우리 코드처럼
transformed = self.transforms(
    image=image,
    masks=[semantic_mask, instance_mask]
)
```

### 8. Gradient Accumulation 버그

```python
# ❌ 잘못된 예
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    output = model(batch)
    loss = criterion(output)
    loss.backward()

    if (i+1) % accumulation_steps == 0:
        optimizer.step()
        # ← optimizer.zero_grad() 빼먹음!

# ✓ 올바른 예
for i, batch in enumerate(dataloader):
    output = model(batch)
    loss = criterion(output)
    (loss / accumulation_steps).backward()  # 스케일링

    if (i+1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()  # ← 필수!
```

---

## 학습 로드맵

초보자를 위한 추천 코드 읽기 순서:

### Phase 1: 설정 이해 (1-2시간)

1. **config.py** 읽기
   - `LaneSegmentationConfig` 클래스
   - 각 하이퍼파라미터의 의미
   - 핵심: backbone, num_classes, embedding_dim, 손실 가중치

### Phase 2: 데이터 흐름 (2-3시간)

2. **dataset.py** 읽기
   - `LaneInstanceDataset.__getitem__()`
   - 마스크 형식 이해
   - Augmentation 파이프라인
   - 핵심: semantic과 instance mask의 동시 로드

3. **실습**: 자신의 데이터셋 준비
   ```bash
   # 폴더 구조 만들기
   mkdir -p data/my_dataset/{train,val}/{images,semantic_masks,instance_masks}
   # 이미지 복사
   cp my_images.jpg data/my_dataset/train/images/
   # 마스크 생성 (convert_tusimple.py 참고)
   ```

### Phase 3: 모델 구조 (2-3시간)

4. **segformer.py** 읽기
   - `SegFormerModel.__init__()`: 모델 로드
   - `predict()`: 추론 과정
   - `get_trainable_params()`: 차등 학습률
   - 핵심: 왜 SegFormer인가? 차등 학습률의 효과

5. **실습**: 모델 로드 및 추론
   ```python
   from road_topology.segmentation.models.segformer import SegFormerModel

   model = SegFormerModel(backbone="b5", num_classes=6)
   image = torch.randn(1, 3, 512, 512)
   output = model.predict(image)
   print(output["logits"].shape)  # (1, 6, 512, 512)
   ```

### Phase 4: 손실 함수 (3-4시간)

6. **losses.py** 읽기 - 상세히!
   - `DiceLoss`: softmax + one-hot + intersection/union 계산
   - `FocalLoss`: focusing parameter의 효과
   - `DiscriminativeLoss`: 3가지 항 (variance, distance, regularization)
   - 핵심: 각 손실이 어떤 문제를 해결하는가?

7. **실습**: 손실 계산 이해
   ```python
   from road_topology.segmentation.losses import DiceLoss, DiscriminativeLoss

   dice = DiceLoss(ignore_index=0)
   logits = torch.randn(2, 6, 256, 256)
   target = torch.randint(0, 6, (2, 256, 256))
   loss = dice(logits, target)
   print(f"Dice Loss: {loss.item():.4f}")

   discrim = DiscriminativeLoss(delta_var=0.5, delta_dist=1.5)
   embeddings = torch.randn(2, 32, 256, 256)
   instance_labels = torch.randint(0, 5, (2, 256, 256))
   loss, loss_dict = discrim(embeddings, instance_labels)
   print(f"Components: {loss_dict}")
   ```

### Phase 5: 학습 루프 (4-5시간)

8. **trainer_lane.py** 읽기
   - `train_epoch()`: 배치 처리, 손실 계산, backward
   - Mixed Precision Training (AMP)
   - Gradient Clipping
   - Early Stopping
   - 핵심: 각 기법이 왜 필요한가?

9. **실습**: 간단한 학습 루프 구현
   ```python
   from road_topology.segmentation.trainer_lane import LaneSegmentationTrainer

   trainer = LaneSegmentationTrainer(
       model=model,
       optimizer=optimizer,
       semantic_criterion=semantic_loss,
       device="cuda"
   )

   for epoch in range(10):
       train_loss = trainer.train_epoch(train_loader)
       val_loss = trainer.validate(val_loader)
       print(f"Epoch {epoch}: train={train_loss:.4f}, val={val_loss:.4f}")
   ```

### Phase 6: 후처리 (2-3시간)

10. **postprocess.py** 읽기
    - `LanePostProcessor.process()`: 클러스터링
    - MeanShift vs DBSCAN
    - Left-to-right 정렬
    - 핵심: Embedding에서 Instance ID로

11. **실습**: 추론 + 후처리
    ```python
    from road_topology.segmentation.postprocess import LanePostProcessor

    model.eval()
    with torch.no_grad():
        outputs = model(image)

    semantic_pred = outputs["logits"].argmax(dim=1)[0].cpu().numpy()
    embeddings = outputs["embeddings"][0].cpu().numpy()  # (32, H, W)

    postprocessor = LanePostProcessor()
    result = postprocessor.process(semantic_pred, embeddings)

    print(f"Detected {result['lane_count']} lanes")
    print(f"Lane centers: {result['lane_centers']}")
    ```

### Phase 7: 종합 실습 (전체 파이프라인)

12. **Train → Evaluate → Infer**
    ```bash
    # 1. 학습
    road-topology train lane \
        --config configs/lane_config.yaml \
        --data data/my_dataset \
        --epochs 50

    # 2. 평가
    road-topology evaluate lane \
        --model results/best_model.pth \
        --data data/my_dataset/val

    # 3. 추론
    road-topology infer lane \
        --model results/best_model.pth \
        --input test_image.jpg \
        --output result.png \
        --visualize
    ```

### 추가 자료

- **논문 읽기**:
  - [SegFormer](https://arxiv.org/abs/2105.15203): 전체 아키텍처
  - [Discriminative Loss](https://arxiv.org/abs/1708.02551): Instance segmentation
  - [LaneNet](https://arxiv.org/abs/1807.01726): Lane detection baseline

- **디버깅 팁**:
  - Visualize 활용: 예측 결과를 시각화해서 어디서 실패하는지 파악
  - Loss 곡선 모니터링: 손실이 떨어지지 않으면? → 학습률, 배치크기, 데이터 확인
  - Gradient 확인: `torch.nn.utils.clip_grad_norm_()` 후 norm 값 모니터링

---

## 요약

| 개념 | 핵심 포인트 |
|-----|-----------|
| **아키텍처** | SegFormer-B5: 전역 맥락 + 효율성 |
| **손실 함수** | Semantic (CE+Dice) + Instance (Discriminative) 결합 |
| **학습 기법** | Mixed Precision + Gradient Clipping + Differential LR |
| **데이터** | Semantic + Instance 마스크 동시 로드 |
| **후처리** | Embedding 클러스터링 → Instance ID + Left-to-right 정렬 |
| **직관** | Embedding 공간에서 같은 차선은 모으고, 다른 차선은 밀어냄 |

---

**문서 작성**: 2026-02-01
**대상 독자**: Road Topology 프로젝트에 기여하고자 하는 ML 엔지니어
