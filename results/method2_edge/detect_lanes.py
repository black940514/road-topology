import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def detect_lanes_edges(image_path, output_dir):
    """엣지 기반 차선 검출"""
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load image: {image_path}")
        return None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 가우시안 블러
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 엣지 검출
    edges = cv2.Canny(blur, 50, 150)

    # ROI 설정 (하단 60%)
    height, width = edges.shape
    roi_vertices = np.array([[(0, height), (0, int(height*0.4)),
                              (width, int(height*0.4)), (width, height)]], dtype=np.int32)
    mask = np.zeros_like(edges)
    cv2.fillPoly(mask, roi_vertices, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    # Hough 변환으로 직선 검출
    lines = cv2.HoughLinesP(masked_edges, 1, np.pi/180, 50,
                            minLineLength=50, maxLineGap=100)

    # 결과 시각화
    result = img.copy()
    lane_count = 0
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 기울기로 좌/우 차선 구분
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            if abs(slope) > 0.3:  # 수평선 제외
                color = (0, 255, 0) if slope > 0 else (255, 0, 0)
                cv2.line(result, (x1, y1), (x2, y2), color, 3)
                lane_count += 1

    # 횡단보도 검출 (수평 패턴)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 1))
    horizontal_lines = cv2.morphologyEx(masked_edges, cv2.MORPH_OPEN, horizontal_kernel)

    # 횡단보도 영역 표시
    crosswalk_contours, _ = cv2.findContours(horizontal_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in crosswalk_contours:
        if cv2.contourArea(contour) > 500:  # 최소 면적 필터
            x, y, w, h = cv2.boundingRect(contour)
            cv2.rectangle(result, (x, y), (x+w, y+h), (255, 255, 0), 2)

    # 저장
    output_path = output_dir / f"{image_path.stem}_lanes.png"
    cv2.imwrite(str(output_path), result)

    edge_path = output_dir / f"{image_path.stem}_edges.png"
    cv2.imwrite(str(edge_path), masked_edges)

    return output_path, lane_count

# 실행
if __name__ == "__main__":
    input_dir = Path("output_dir3")
    output_dir = Path("results/method2_edge")
    output_dir.mkdir(exist_ok=True, parents=True)

    images = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
    print(f"Processing {len(images)} images with edge detection...")

    if len(images) == 0:
        print("No images found in output_dir3!")
    else:
        total_lanes = 0
        processed = 0
        for img_path in tqdm(images[:10]):  # 샘플 10개
            result = detect_lanes_edges(img_path, output_dir)
            if result:
                _, lane_count = result
                total_lanes += lane_count
                processed += 1

        print(f"\nEdge-based lane detection complete!")
        print(f"Processed: {processed} images")
        print(f"Total lanes detected: {total_lanes}")
        print(f"Average lanes per image: {total_lanes/processed if processed > 0 else 0:.2f}")
