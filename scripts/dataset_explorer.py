"""
Dataset Explorer module for YOLO training template.
Provides functionality to load, visualize, and analyze YOLO datasets.
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_dataset(dataset_path: str) -> Dict[str, List[str]]:
    """
    Load dataset structure and return image paths per split.

    Args:
        dataset_path: Path to dataset root directory

    Returns:
        Dictionary with keys like 'train', 'val', 'test' mapping to lists of image paths
    """
    splits = {}
    subdirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    logging.info(f"Detected subdirs in dataset: {subdirs}")

    # Case-insensitive mapping for common folder names
    subdirs_lower = {d.lower(): d for d in subdirs}
    split_names = ['train', 'valid', 'test']
    split_variants = {
        'train': ['train', 'training'],
        'valid': ['valid', 'validation', 'val'],
        'test': ['test', 'testing']
    }

    for split_key, variants in split_variants.items():
        key_split = 'val' if split_key == 'valid' else split_key
        for variant in variants:
            if variant in subdirs_lower:
                split_dir = os.path.join(dataset_path, subdirs_lower[variant])
                # Look for images folder (case-insensitive)
                images_variants = ['images', 'image', 'imgs', 'img']
                images_dir = None
                for img_var in images_variants:
                    if img_var in [d.lower() for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]:
                        images_dir = os.path.join(split_dir, [d for d in os.listdir(split_dir) if d.lower() == img_var][0])
                        break
                if images_dir and os.path.exists(images_dir):
                    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
                    image_files = []
                    for ext in image_extensions:
                        image_files.extend(Path(images_dir).glob(f'*{ext}'))
                        image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
                    if image_files:  # Only add if there are images
                        splits[key_split] = sorted([str(f) for f in image_files])
                    break

    # Fallback: check if images/labels are directly in root (case-insensitive)
    if not splits:
        images_variants = ['images', 'image', 'imgs', 'img']
        labels_variants = ['labels', 'label', 'anns', 'annotations']
        images_dir = None
        labels_dir = None
        for img_var in images_variants:
            if img_var in subdirs_lower:
                images_dir = os.path.join(dataset_path, subdirs_lower[img_var])
                break
        for lbl_var in labels_variants:
            if lbl_var in subdirs_lower:
                labels_dir = os.path.join(dataset_path, subdirs_lower[lbl_var])
                break

        if images_dir and os.path.exists(images_dir):
            image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(images_dir).glob(f'*{ext}'))
                image_files.extend(Path(images_dir).glob(f'*{ext.upper()}'))
            if image_files:
                splits['train'] = sorted([str(f) for f in image_files])

    return splits

def parse_yolo_labels(label_path: str, img_width: int, img_height: int) -> List[Dict[str, Any]]:
    """
    Parse YOLO format labels and return bbox data in pixel coordinates.

    Args:
        label_path: Path to .txt label file
        img_width: Image width in pixels
        img_height: Image height in pixels

    Returns:
        List of dicts with 'class_id', 'bbox' (x1,y1,x2,y2), 'confidence'
    """
    bboxes = []
    if not os.path.exists(label_path):
        return bboxes

    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                class_id = int(parts[0])
                x_center = float(parts[1]) * img_width
                y_center = float(parts[2]) * img_height
                width = float(parts[3]) * img_width
                height = float(parts[4]) * img_height

                x1 = int(x_center - width / 2)
                y1 = int(y_center - height / 2)
                x2 = int(x_center + width / 2)
                y2 = int(y_center + height / 2)

                confidence = float(parts[5]) if len(parts) > 5 else 1.0

                bboxes.append({
                    'class_id': class_id,
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence
                })
    return bboxes

def draw_annotations(image: np.ndarray, bboxes: List[Dict[str, Any]], class_names: Optional[List[str]] = None, conf_threshold: float = 0.0) -> np.ndarray:
    """
    Draw bounding boxes on image.

    Args:
        image: Input image as numpy array
        bboxes: List of bbox dicts
        class_names: Optional list of class names
        conf_threshold: Minimum confidence to display

    Returns:
        Image with annotations drawn
    """
    img = image.copy()
    colors = plt.cm.tab10(np.linspace(0, 1, 10))  # Use matplotlib colormap for colors

    for bbox_data in bboxes:
        if bbox_data['confidence'] < conf_threshold:
            continue

        class_id = bbox_data['class_id']
        x1, y1, x2, y2 = bbox_data['bbox']
        conf = bbox_data['confidence']

        color = tuple(int(c * 255) for c in colors[class_id % 10][:3])  # RGB to BGR for OpenCV

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        label = f"Class {class_id}"
        if class_names and class_id < len(class_names):
            label = class_names[class_id]
        if conf < 1.0:
            label += f" {conf:.2f}"

        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return img

def compute_dataset_stats(dataset_splits: Dict[str, List[str]], labels_dir: Optional[str] = None, class_names: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Compute statistics for the dataset.

    Args:
        dataset_splits: Dictionary of split names to image paths
        labels_dir: Directory containing labels (if not in same structure)
        class_names: Optional class names list

    Returns:
        Dictionary with stats: total_images, class_counts, etc.
    """
    stats = {
        'total_images': 0,
        'split_counts': {},
        'class_counts': defaultdict(int),
        'bbox_stats': {'total_bboxes': 0, 'avg_bbox_per_image': 0.0}
    }

    total_images = 0
    total_bboxes = 0

    for split_name, image_paths in dataset_splits.items():
        stats['split_counts'][split_name] = len(image_paths)
        total_images += len(image_paths)

        for img_path in image_paths:
            img = cv2.imread(img_path)
            if img is None:
                continue

            h, w = img.shape[:2]
            label_path = img_path.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt').replace('.bmp', '.txt')
            if labels_dir:
                label_path = os.path.join(labels_dir, os.path.basename(label_path))

            bboxes = parse_yolo_labels(label_path, w, h)
            for bbox in bboxes:
                stats['class_counts'][bbox['class_id']] += 1
                total_bboxes += 1

    stats['total_images'] = total_images
    stats['bbox_stats']['total_bboxes'] = total_bboxes
    if total_images > 0:
        stats['bbox_stats']['avg_bbox_per_image'] = total_bboxes / total_images

    # Convert defaultdict to dict
    stats['class_counts'] = dict(stats['class_counts'])

    return stats

def get_sample_images(dataset_splits: Dict[str, List[str]], sample_size: int = 10, labels_dir: Optional[str] = None, class_names: Optional[List[str]] = None, conf_threshold: float = 0.0, filter_classes: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """
    Get sample images with annotations for display.

    Args:
        dataset_splits: Dataset splits
        sample_size: Number of images to sample
        labels_dir: Labels directory
        class_names: Class names
        conf_threshold: Confidence threshold

    Returns:
        List of dicts with 'image', 'path', 'bboxes'
    """
    all_images = []
    for split, paths in dataset_splits.items():
        all_images.extend(paths)

    # Sample randomly
    np.random.shuffle(all_images)
    sample_paths = all_images[:min(sample_size, len(all_images))]

    samples = []
    for img_path in sample_paths:
        img = cv2.imread(img_path)
        if img is None:
            continue

        h, w = img.shape[:2]
        label_path = img_path.replace('.jpg', '.txt').replace('.png', '.txt').replace('.jpeg', '.txt').replace('.bmp', '.txt')
        if labels_dir:
            label_path = os.path.join(labels_dir, os.path.basename(label_path))

        bboxes = parse_yolo_labels(label_path, w, h)
        # Filter by classes if specified
        if filter_classes is not None:
            bboxes = [b for b in bboxes if b['class_id'] in filter_classes]
        annotated_img = draw_annotations(img, bboxes, class_names, conf_threshold)

        samples.append({
            'image': cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB),  # Convert to RGB for PIL
            'path': img_path,
            'bboxes': bboxes
        })

    return samples