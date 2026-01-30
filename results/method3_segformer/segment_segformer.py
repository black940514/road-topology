import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
import torchvision
import torchvision.models.segmentation as segmentation
from tqdm import tqdm

def segment_with_deeplabv3(image_path, output_dir, model, device):
    """DeepLabV3 기반 세그멘테이션 (SegFormer 대체)"""
    # 이미지 로드
    image = Image.open(image_path).convert("RGB")
    img_np = np.array(image)

    # 전처리
    preprocess = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0).to(device)

    # 추론
    with torch.no_grad():
        output = model(input_batch)['out'][0]

    # 후처리
    output_predictions = output.argmax(0).cpu().numpy()

    # PASCAL VOC 클래스 매핑
    # 클래스 0: background
    # 클래스 7: car
    # 클래스 15: person
    # Approximating road as non-person, non-car, non-background areas
    # For better results, we'll use inverse of high-confidence non-road areas

    # 도로로 추정되는 영역 (낮은 클래스 ID)
    road_mask = np.logical_or(
        output_predictions == 0,  # background often includes road
        np.logical_and(output_predictions > 0, output_predictions < 7)
    ).astype(np.uint8) * 255

    # 시각화
    overlay = img_np.copy()

    # 세그멘테이션 결과를 색상으로 표시
    # 클래스 0-6을 회색(도로)으로 표시
    road_color = [128, 128, 128]
    overlay[road_mask == 255] = road_color

    # 다른 객체들은 다른 색상으로
    vehicle_mask = np.isin(output_predictions, [7]).astype(bool)
    person_mask = np.isin(output_predictions, [15]).astype(bool)

    overlay[vehicle_mask] = [0, 0, 255]  # 빨강 - 차량
    overlay[person_mask] = [0, 255, 0]   # 초록 - 사람

    blended = cv2.addWeighted(img_np, 0.6, overlay, 0.4, 0)

    # 저장
    output_path = output_dir / f"{image_path.stem}_segformer.png"
    cv2.imwrite(str(output_path), cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))

    mask_path = output_dir / f"{image_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), road_mask)

    return output_path, output_predictions

# 모델 로드 (DeepLabV3 - SegFormer 대체)
# Use CPU due to CUDA compatibility issues
device = "cpu"
print(f"Using device: {device}")
print("Note: Using DeepLabV3 instead of SegFormer due to compatibility issues with PyTorch 1.8.0")
print("Note: Using CPU due to CUDA version mismatch (GPU requires newer PyTorch)")

# DeepLabV3 with ResNet101 backbone
model = segmentation.deeplabv3_resnet101(pretrained=True).to(device)
model.eval()

# 실행
input_dir = Path("output_dir3")
output_dir = Path("results/method3_segformer")
output_dir.mkdir(exist_ok=True)

images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
print(f"Processing {len(images)} images with DeepLabV3...")

for img_path in tqdm(images[:10]):  # 샘플 10개
    try:
        segment_with_deeplabv3(img_path, output_dir, model, device)
    except Exception as e:
        print(f"Error processing {img_path.name}: {e}")

print("Segmentation complete!")
