import cv2
import numpy as np
from pathlib import Path
from PIL import Image
import torch
from tqdm import tqdm

# SAM availability check
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("SAM not available, using alternative approach")

def segment_with_sam_alternative(image_path, output_dir):
    """SAM 대안: GrabCut + 색상 기반 세그멘테이션"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load {image_path}")
        return None

    height, width = img.shape[:2]

    # 초기 마스크 (하단 60%를 도로로 가정)
    mask = np.zeros((height, width), np.uint8)

    # GrabCut용 초기 영역
    rect = (0, int(height*0.4), width, height)

    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # GrabCut 실행
    cv2.grabCut(img, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # 전경/배경 분리
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    # 결과 시각화
    result = img.copy()
    result[mask2 == 1] = result[mask2 == 1] * 0.5 + np.array([0, 128, 128]) * 0.5

    # 저장
    output_path = output_dir / f"{image_path.stem}_sam.png"
    cv2.imwrite(str(output_path), result.astype(np.uint8))

    mask_path = output_dir / f"{image_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), mask2 * 255)

    return output_path

def segment_with_color_clustering(image_path, output_dir):
    """K-means 클러스터링 기반 도로 세그멘테이션"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load {image_path}")
        return None

    height, width = img.shape[:2]

    # 하단 영역만 처리 (ROI)
    roi = img[int(height*0.4):, :]

    # K-means 클러스터링
    pixels = roi.reshape(-1, 3).astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 5
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_PP_CENTERS)

    # 도로 클러스터 선택 (가장 어두운 회색 톤)
    gray_centers = np.mean(centers, axis=1)
    road_cluster = np.argmin(np.abs(gray_centers - 100))  # 회색에 가까운 클러스터

    # 마스크 생성
    labels_reshaped = labels.reshape(roi.shape[:2])
    road_mask = (labels_reshaped == road_cluster).astype(np.uint8) * 255

    # 전체 이미지 마스크
    full_mask = np.zeros((height, width), np.uint8)
    full_mask[int(height*0.4):, :] = road_mask

    # 시각화
    result = img.copy()
    result[full_mask > 0] = result[full_mask > 0] * 0.6 + np.array([0, 100, 200]) * 0.4

    output_path = output_dir / f"{image_path.stem}_sam.png"
    cv2.imwrite(str(output_path), result.astype(np.uint8))

    mask_path = output_dir / f"{image_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), full_mask)

    return output_path

def advanced_road_segmentation(image_path, output_dir):
    """고급 도로 세그멘테이션: 색상 + 질감 + 형태학적 연산"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load {image_path}")
        return None

    height, width = img.shape[:2]

    # 1. 색상 기반 도로 후보 추출 (회색/검정 계열)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 도로 색상 범위 (낮은 채도, 중간 밝기)
    lower_gray = np.array([0, 0, 30])
    upper_gray = np.array([180, 50, 150])
    color_mask = cv2.inRange(hsv, lower_gray, upper_gray)

    # 2. 하단 영역에 가중치 부여
    position_weight = np.zeros((height, width), dtype=np.float32)
    position_weight[int(height*0.3):, :] = 1.0
    position_weight[:int(height*0.3), :] = np.linspace(0, 1, int(height*0.3))[:, np.newaxis]

    # 3. 질감 분석 (엣지 밀도가 낮은 영역 = 도로)
    edges = cv2.Canny(gray, 30, 100)
    edge_density = cv2.GaussianBlur(edges, (21, 21), 0)
    texture_mask = 255 - edge_density  # 엣지가 적은 곳 = 도로

    # 4. 결합 및 정제
    combined = cv2.addWeighted(color_mask.astype(np.float32), 0.5,
                               texture_mask.astype(np.float32), 0.3, 0)
    combined = (combined * position_weight).astype(np.uint8)

    # 5. 형태학적 연산으로 노이즈 제거
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN, kernel)

    # 6. 이진화
    _, final_mask = cv2.threshold(combined, 50, 255, cv2.THRESH_BINARY)

    # 7. 최대 연결 컴포넌트만 유지 (가장 큰 도로 영역)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(final_mask, connectivity=8)
    if num_labels > 1:
        # 배경(0) 제외하고 가장 큰 컴포넌트
        largest_component = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        final_mask = (labels == largest_component).astype(np.uint8) * 255

    # 8. 시각화
    result = img.copy()
    overlay = result.copy()
    overlay[final_mask > 0] = [0, 150, 255]  # 주황색 오버레이
    result = cv2.addWeighted(result, 0.6, overlay, 0.4, 0)

    # 마스크 경계선 그리기
    contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    # 저장
    output_path = output_dir / f"{image_path.stem}_sam.png"
    cv2.imwrite(str(output_path), result)

    mask_path = output_dir / f"{image_path.stem}_mask.png"
    cv2.imwrite(str(mask_path), final_mask)

    return output_path

# 실행
if __name__ == "__main__":
    input_dir = Path("output_dir3")
    output_dir = Path("results/method4_sam")
    output_dir.mkdir(exist_ok=True, parents=True)

    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    images = [img for img in images if not img.name.startswith('.')]

    print(f"Processing {len(images)} images with advanced segmentation...")

    success_count = 0
    for img_path in tqdm(images[:15]):  # 샘플 15개
        try:
            result = advanced_road_segmentation(img_path, output_dir)
            if result:
                success_count += 1
        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    print(f"\nSegmentation complete! {success_count}/{min(15, len(images))} images processed successfully.")
    print(f"Results saved to: {output_dir}")
