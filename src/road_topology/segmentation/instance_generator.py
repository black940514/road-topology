"""
Instance Mask Generator for Lane Detection

This module provides functionality to generate lane instance masks from semantic
segmentation using connected components analysis.
"""

import numpy as np
import cv2
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm


class InstanceMaskGenerator:
    """
    Generates lane instance masks from semantic segmentation masks.

    Uses connected components to separate individual lane markings and orders
    them left-to-right based on x-centroid position.

    Attributes:
        lane_class_id: Semantic class ID for lane markings (default: 2)
        min_component_size: Minimum pixels for valid lane component (default: 100)
        connectivity: Connectivity for component analysis (4 or 8, default: 8)

    Examples:
        >>> generator = InstanceMaskGenerator(lane_class_id=2, min_component_size=100)
        >>> semantic_mask = np.array([[0, 0, 2, 2], [0, 2, 2, 0]])
        >>> instance_mask = generator.generate_instance_mask(semantic_mask)
        >>> print(instance_mask.shape)
        (2, 4)

        >>> # Process entire dataset
        >>> stats = generator.process_dataset(
        ...     semantic_masks_dir="data/semantic",
        ...     output_dir="data/instance",
        ...     pattern="*.png"
        ... )
        >>> print(f"Processed {stats['total_images']} images")
    """

    def __init__(
        self,
        lane_class_id: int = 2,
        min_component_size: int = 100,
        connectivity: int = 8
    ):
        """
        Initialize InstanceMaskGenerator.

        Args:
            lane_class_id: Semantic class ID for lane markings
            min_component_size: Minimum number of pixels for a valid lane component
            connectivity: Connectivity type (4 or 8) for connected components
        """
        if connectivity not in [4, 8]:
            raise ValueError("connectivity must be 4 or 8")

        self.lane_class_id = lane_class_id
        self.min_component_size = min_component_size
        self.connectivity = connectivity

    def generate_instance_mask(self, semantic_mask: np.ndarray) -> np.ndarray:
        """
        Generate instance mask from semantic segmentation mask.

        Process:
        1. Extract lane marking pixels (semantic_mask == lane_class_id)
        2. Apply connected components analysis
        3. Filter out small components (noise)
        4. Order components left-to-right by x-centroid
        5. Reassign instance IDs based on left-to-right order

        Args:
            semantic_mask: 2D numpy array of semantic segmentation (H, W)

        Returns:
            instance_mask: 2D numpy array with instance IDs (H, W)
                          0 = background, 1-N = lane instances ordered left-to-right

        Examples:
            >>> generator = InstanceMaskGenerator()
            >>> semantic = np.zeros((100, 200), dtype=np.uint8)
            >>> semantic[40:60, 50:70] = 2  # Left lane
            >>> semantic[40:60, 130:150] = 2  # Right lane
            >>> instance = generator.generate_instance_mask(semantic)
            >>> assert instance[50, 60] == 1  # Left lane is ID 1
            >>> assert instance[50, 140] == 2  # Right lane is ID 2
        """
        # Validate input
        if semantic_mask.ndim != 2:
            raise ValueError(f"Expected 2D semantic mask, got shape {semantic_mask.shape}")

        # Extract lane marking pixels
        lane_binary = (semantic_mask == self.lane_class_id).astype(np.uint8)

        # Apply connected components
        num_labels, labels = cv2.connectedComponents(
            lane_binary,
            connectivity=self.connectivity
        )

        # Filter small components and compute centroids
        valid_components = []

        for label_id in range(1, num_labels):  # Skip 0 (background)
            component_mask = (labels == label_id)
            component_size = np.sum(component_mask)

            # Filter by size
            if component_size < self.min_component_size:
                continue

            # Compute x-centroid for ordering
            y_coords, x_coords = np.where(component_mask)
            x_centroid = np.mean(x_coords)

            valid_components.append({
                'label_id': label_id,
                'x_centroid': x_centroid,
                'size': component_size
            })

        # Sort components left-to-right by x-centroid
        valid_components.sort(key=lambda c: c['x_centroid'])

        # Create instance mask with ordered IDs
        instance_mask = np.zeros_like(semantic_mask, dtype=np.uint8)

        for new_id, component in enumerate(valid_components, start=1):
            old_label = component['label_id']
            instance_mask[labels == old_label] = new_id

        return instance_mask

    def process_dataset(
        self,
        semantic_masks_dir: str,
        output_dir: str,
        pattern: str = "*.png"
    ) -> Dict[str, any]:
        """
        Process entire dataset of semantic masks to generate instance masks.

        Args:
            semantic_masks_dir: Directory containing semantic segmentation masks
            output_dir: Directory to save generated instance masks
            pattern: File pattern to match (default: "*.png")

        Returns:
            stats: Dictionary containing processing statistics
                - total_images: Total number of images processed
                - total_lanes: Total number of lane instances detected
                - avg_lanes_per_image: Average lanes per image
                - lane_count_distribution: Dict mapping lane count to frequency
                - failed_images: List of filenames that failed processing

        Examples:
            >>> generator = InstanceMaskGenerator(min_component_size=50)
            >>> stats = generator.process_dataset(
            ...     semantic_masks_dir="data/semantic_masks",
            ...     output_dir="data/instance_masks",
            ...     pattern="*.png"
            ... )
            >>> print(f"Average lanes per image: {stats['avg_lanes_per_image']:.2f}")
        """
        semantic_dir = Path(semantic_masks_dir)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Collect all semantic mask files
        mask_files = sorted(semantic_dir.glob(pattern))

        if not mask_files:
            raise ValueError(f"No files matching pattern '{pattern}' found in {semantic_dir}")

        # Initialize statistics
        stats = {
            'total_images': 0,
            'total_lanes': 0,
            'lane_count_distribution': {},
            'failed_images': []
        }

        # Process each semantic mask
        for mask_file in tqdm(mask_files, desc="Generating instance masks"):
            try:
                # Load semantic mask
                semantic_mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)

                if semantic_mask is None:
                    stats['failed_images'].append(mask_file.name)
                    continue

                # Generate instance mask
                instance_mask = self.generate_instance_mask(semantic_mask)

                # Count lanes in this image
                num_lanes = instance_mask.max()
                stats['total_lanes'] += num_lanes
                stats['total_images'] += 1

                # Update distribution
                stats['lane_count_distribution'][num_lanes] = \
                    stats['lane_count_distribution'].get(num_lanes, 0) + 1

                # Save instance mask
                output_file = output_path / mask_file.name
                cv2.imwrite(str(output_file), instance_mask)

            except Exception as e:
                stats['failed_images'].append(f"{mask_file.name}: {str(e)}")

        # Compute average
        if stats['total_images'] > 0:
            stats['avg_lanes_per_image'] = stats['total_lanes'] / stats['total_images']
        else:
            stats['avg_lanes_per_image'] = 0.0

        return stats

    def get_component_stats(self, instance_mask: np.ndarray) -> Dict[int, Dict[str, any]]:
        """
        Get detailed statistics for each lane instance.

        Args:
            instance_mask: Instance mask with lane IDs

        Returns:
            Dictionary mapping instance_id to statistics dict containing:
                - size: Number of pixels
                - x_centroid: X-coordinate of centroid
                - y_centroid: Y-coordinate of centroid
                - bbox: Bounding box as (x_min, y_min, x_max, y_max)

        Examples:
            >>> generator = InstanceMaskGenerator()
            >>> instance_mask = np.array([[1, 1, 0, 2], [1, 0, 0, 2]])
            >>> stats = generator.get_component_stats(instance_mask)
            >>> print(stats[1]['size'])
            3
        """
        num_instances = instance_mask.max()
        component_stats = {}

        for instance_id in range(1, num_instances + 1):
            mask = (instance_mask == instance_id)
            y_coords, x_coords = np.where(mask)

            if len(y_coords) == 0:
                continue

            component_stats[instance_id] = {
                'size': len(y_coords),
                'x_centroid': float(np.mean(x_coords)),
                'y_centroid': float(np.mean(y_coords)),
                'bbox': (
                    int(x_coords.min()),
                    int(y_coords.min()),
                    int(x_coords.max()),
                    int(y_coords.max())
                )
            }

        return component_stats


if __name__ == "__main__":
    # Example usage
    print("InstanceMaskGenerator - Example Usage")
    print("=" * 50)

    # Create synthetic semantic mask with 3 lanes
    semantic_mask = np.zeros((100, 300), dtype=np.uint8)

    # Left lane
    semantic_mask[40:60, 50:70] = 2

    # Center lane
    semantic_mask[40:60, 140:160] = 2

    # Right lane
    semantic_mask[40:60, 230:250] = 2

    # Add some noise (small components)
    semantic_mask[10:12, 10:12] = 2

    # Generate instance mask
    generator = InstanceMaskGenerator(min_component_size=100)
    instance_mask = generator.generate_instance_mask(semantic_mask)

    print(f"\nSemantic mask shape: {semantic_mask.shape}")
    print(f"Instance mask shape: {instance_mask.shape}")
    print(f"Number of lane instances: {instance_mask.max()}")

    # Get component statistics
    stats = generator.get_component_stats(instance_mask)
    print("\nLane Instance Statistics:")
    for instance_id, stat in stats.items():
        print(f"  Lane {instance_id}: size={stat['size']}, "
              f"x_centroid={stat['x_centroid']:.1f}, "
              f"bbox={stat['bbox']}")
