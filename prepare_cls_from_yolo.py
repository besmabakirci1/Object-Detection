#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
from typing import List, Tuple
import yaml
import cv2
from tqdm import tqdm


def load_names_from_yaml(yaml_path: Path) -> List[str]:
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    names = data.get('names')
    if isinstance(names, dict):
        # sometimes names is {0: 'classA', 1: 'classB', ...}
        names = [names[k] for k in sorted(names.keys())]
    if not isinstance(names, list):
        raise ValueError('names not found or not a list in data.yaml')
    return names


def parse_yolo_label_line(line: str) -> Tuple[int, float, float, float, float]:
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError('Invalid YOLO label line')
    cls_id = int(float(parts[0]))
    x, y, w, h = map(float, parts[1:5])
    return cls_id, x, y, w, h


def yolo_to_xyxy(x: float, y: float, w: float, h: float, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    cx = x * img_w
    cy = y * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = int(round(cx - bw / 2))
    y1 = int(round(cy - bh / 2))
    x2 = int(round(cx + bw / 2))
    y2 = int(round(cy + bh / 2))
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_w - 1, x2)
    y2 = min(img_h - 1, y2)
    return x1, y1, x2, y2


def find_split_dirs(root: Path) -> List[Tuple[Path, Path]]:
    candidates = []
    # common layouts:
    # 1) root/images/train, root/labels/train ...
    for split in ['train', 'valid', 'val', 'test']:
        img_dir = root / 'images' / split
        lbl_dir = root / 'labels' / split
        if img_dir.exists() and lbl_dir.exists():
            candidates.append((img_dir, lbl_dir))
    # 2) root/train/images, root/train/labels ...
    for split in ['train', 'valid', 'val', 'test']:
        split_dir = root / split
        img_dir = split_dir / 'images'
        lbl_dir = split_dir / 'labels'
        if img_dir.exists() and lbl_dir.exists():
            candidates.append((img_dir, lbl_dir))
    return candidates


def convert(root: Path, out_dir: Path, data_yaml: Path, min_size: int = 16) -> None:
    names = load_names_from_yaml(data_yaml)
    out_dir.mkdir(parents=True, exist_ok=True)
    # make class dirs
    class_dirs = []
    for name in names:
        safe = name.replace('/', '_').replace(' ', '_')
        d = out_dir / safe
        d.mkdir(parents=True, exist_ok=True)
        class_dirs.append(d)

    splits = find_split_dirs(root)
    if not splits:
        raise RuntimeError('Could not find images/labels split directories under dataset root')

    for img_dir, lbl_dir in splits:
        img_paths = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.png'))
        for img_path in tqdm(img_paths, desc=f'Processing {img_dir.relative_to(root)}'):
            lbl_path = lbl_dir / (img_path.stem + '.txt')
            if not lbl_path.exists():
                continue
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            h, w = img.shape[:2]
            with open(lbl_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        cls_id, x, y, bw, bh = parse_yolo_label_line(line)
                    except Exception:
                        continue
                    x1, y1, x2, y2 = yolo_to_xyxy(x, y, bw, bh, w, h)
                    if x2 <= x1 or y2 <= y1:
                        continue
                    if (x2 - x1) < min_size or (y2 - y1) < min_size:
                        continue
                    crop = img[y1:y2, x1:x2]
                    cls_dir = class_dirs[cls_id]
                    # filename
                    idx = len(list(cls_dir.glob('*.jpg')))
                    out_path = cls_dir / f'{img_path.stem}_{idx:06d}.jpg'
                    cv2.imwrite(str(out_path), crop)


def main():
    parser = argparse.ArgumentParser(description='Convert YOLO/Roboflow detection dataset to classification ImageFolder by cropping boxes')
    parser.add_argument('--dataset-root', type=str, required=True, help='Root of YOLO dataset (contains images/ and labels/ or split subfolders)')
    parser.add_argument('--data-yaml', type=str, required=True, help='Path to data.yaml with names list')
    parser.add_argument('--out-dir', type=str, default='models/car_cls_from_roboflow', help='Output ImageFolder directory')
    parser.add_argument('--min-size', type=int, default=16, help='Minimum crop size (pixels) to keep')
    args = parser.parse_args()

    convert(Path(args.dataset_root), Path(args.out_dir), Path(args.data_yaml), args.min_size)
    print('Done. ImageFolder created at:', args.out_dir)


if __name__ == '__main__':
    main()

