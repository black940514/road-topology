import cv2
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def create_comparison_image(image_name, output_dir):
    """5가지 방법 비교 이미지 생성"""
    methods = {
        'Original': f'output_dir3/{image_name}',
        'Color (HSV)': f'results/method1_color/{image_name.replace(".jpg", "_color.png").replace(".png", "_color.png")}',
        'Edge (Hough)': f'results/method2_edge/{image_name.replace(".jpg", "_lanes.png").replace(".png", "_lanes.png")}',
        'SegFormer': f'results/method3_segformer/{image_name.replace(".jpg", "_segformer.png").replace(".png", "_segformer.png")}',
        'SAM/GrabCut': f'results/method4_sam/{image_name.replace(".jpg", "_sam.png").replace(".png", "_sam.png")}',
        'DeepLabV3': f'results/method5_deeplab/{image_name.replace(".jpg", "_deeplab.png").replace(".png", "_deeplab.png")}',
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()

    for idx, (name, path) in enumerate(methods.items()):
        if Path(path).exists():
            img = cv2.imread(path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                axes[idx].imshow(img)
            else:
                axes[idx].text(0.5, 0.5, f'Error Loading\n{path}', ha='center', va='center', fontsize=10)
        else:
            axes[idx].text(0.5, 0.5, f'Not Available\n{path}', ha='center', va='center', fontsize=10, wrap=True)
        axes[idx].set_title(name, fontsize=14, fontweight='bold')
        axes[idx].axis('off')

    plt.suptitle(f'Road Segmentation Comparison: {image_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()

    output_path = output_dir / f'comparison_{image_name.replace(".jpg", ".png").replace(".png", ".png")}'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()

    return output_path

def main():
    # 실행
    output_dir = Path('results/comparison')
    output_dir.mkdir(exist_ok=True, parents=True)

    # 샘플 이미지들로 비교 생성
    sample_images = ['01CT000000164.jpg', '02CT000000181.jpg', '03CT000000034.jpg',
                     '04CT000001344.jpg', '05CT000000473.jpg']

    created_count = 0
    for img_name in sample_images:
        if Path(f'output_dir3/{img_name}').exists():
            try:
                output_path = create_comparison_image(img_name, output_dir)
                print(f'✓ Created comparison for {img_name}: {output_path}')
                created_count += 1
            except Exception as e:
                print(f'✗ Error creating comparison for {img_name}: {e}')
        else:
            print(f'✗ Original image not found: {img_name}')

    print(f'\n=== Summary ===')
    print(f'Total comparisons created: {created_count}/{len(sample_images)}')
    print(f'Output directory: {output_dir.absolute()}')

if __name__ == '__main__':
    main()
