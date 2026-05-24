
!pip install -q groundingdino-py supervision ultralytics
!pip install -q segment-anything torchvision
!wget -q https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth

from google.colab import drive
drive.mount('/content/drive')

from pathlib import Path

#All paths in one place 
IMAGE_FRESH  = Path("/content/drive/MyDrive/foto/assets/images/fresh")
LABEL_FRESH  = Path("/content/drive/MyDrive/foto/assets/labels/fresh")
IMAGE_TRAIN  = Path("/content/drive/MyDrive/foto/assets/images/train")
IMAGE_VAL    = Path("/content/drive/MyDrive/foto/assets/images/val")
LABEL_TRAIN  = Path("/content/drive/MyDrive/foto/assets/labels/train")
LABEL_VAL    = Path("/content/drive/MyDrive/foto/assets/labels/val")
DATASET_BASE = Path("/content/drive/MyDrive/foto/assets")
RUNS_DIR     = Path("/content/drive/MyDrive/foto/runs")

# Create all folders
for p in [LABEL_FRESH, IMAGE_TRAIN, IMAGE_VAL, LABEL_TRAIN, LABEL_VAL, RUNS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Collect all images
image_paths = [
    f for f in IMAGE_FRESH.iterdir()
    if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}
]
print(f"Total images found: {len(image_paths)}")


def classify_image(img_path):
    img = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    strip_hues = []
    for i in range(8):
        x1, x2 = i * w // 8, (i+1) * w // 8
        strip = img_hsv[h//4 : 3*h//4, x1:x2]
        strip_hues.append(float(np.median(strip[:, :, 0])))
    hue_std = np.std(strip_hues)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    vert_edge_ratio = np.abs(sobel_x).mean() / (np.abs(sobel_y).mean() + 1e-5)

    mid_row = img_hsv[h//2, :, 0].astype(float)
    diffs_smooth = np.convolve(np.abs(np.diff(mid_row)), np.ones(15)/15, mode='same')
    color_transitions = int((diffs_smooth > 15).sum())
  
    votes_group = sum([
        hue_std > 12,
        vert_edge_ratio > 1.15,
        color_transitions > w * 0.08
    ])
    return 'group' if votes_group >= 2 else 'single'
  
import random
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np

#Classify images into singles and groups
singles = []
groups = []

for img_path in image_paths:
    if classify_image(img_path) == 'single':
        singles.append(img_path)
    else:
        groups.append(img_path)

print(f"Classified {len(singles)} single images and {len(groups)} group images.")

# Random sample every run 
random.seed()   
n_show  = 12  
n_cols  = 6
n_rows  = 2

sample_singles = random.sample(singles, min(n_cols, len(singles)))
sample_groups  = random.sample(groups,  min(n_cols, len(groups)))

fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 3, n_rows * 3.5))

for ax, p in zip(axes[0], sample_singles):
    ax.imshow(Image.open(p))
    ax.set_title("single", fontsize=8, color="royalblue", fontweight='bold')
    ax.axis('off')

for ax, p in zip(axes[1], sample_groups):
    ax.imshow(Image.open(p))
    ax.set_title("group", fontsize=8, color="green", fontweight='bold')
    ax.axis('off')

for ax in axes[0][len(sample_singles):]:
    ax.axis('off')
for ax in axes[1][len(sample_groups):]:
    ax.axis('off')

plt.suptitle(
    f"Top: SINGLE ({len(singles)} total)  |  "
    f"Bottom: GROUP ({len(groups)} total)\n"
    f"Run cell again for a new random sample",
    fontsize=11
)
plt.tight_layout()
plt.savefig(
    '/content/drive/MyDrive/foto/classification_check.png',
    dpi=100, bbox_inches='tight'
)
plt.show()
print("Run this cell again to see a different random sample")

def label_single_image(img_path, label_dir, padding=0.04):
    """One box covering full image minus small padding"""
    label_path = Path(label_dir) / (img_path.stem + ".txt")
    p = padding
    with open(label_path, 'w') as f:
        f.write(f"0 0.500000 0.500000 {1-(2*p):.6f} {1-(2*p):.6f}\n")

for p in singles:
    label_single_image(p, LABEL_FRESH)

print(f"Labeled {len(singles)} single images")

import torch
import os
import sys

!pip install --upgrade pip setuptools

!apt-get update -qq > /dev/null
!apt-get install -y build-essential python3-dev > /dev/null


!pip install supervision==0.6.0 groundingdino-py==0.4.0 transformers==4.38.2 ultralytics

if not os.path.exists("groundingdino_repo"):
    !git clone https://github.com/IDEA-Research/GroundingDINO.git groundingdino_repo

!mkdir -p weights
!wget -q -O weights/groundingdino_swint_ogc.pth \
    https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth

BERTWARPER_FILE = "groundingdino_repo/groundingdino/models/GroundingDINO/bertwarper.py"

if os.path.exists(BERTWARPER_FILE):
    src = open(BERTWARPER_FILE).read()
    sentinel = "self.bert.get_head_mask = lambda"

    if sentinel not in src:
        patch = (
            "\n        # --- compatibility patch ---\n"
            "        if not hasattr(self.bert, 'get_head_mask'):\n"
            "            self.bert.get_head_mask = lambda head_mask, num_hidden_layers: [None] * num_hidden_layers\n"
            "        # --- end patch ---\n"
        )
        anchor = "self.config = bert_model.config"
        src = src.replace(anchor, anchor + patch, 1)  
        open(BERTWARPER_FILE, "w").write(src)
        print(f"Patched {BERTWARPER_FILE}")
    else:
        print(f"Patch already applied to {BERTWARPER_FILE}")
else:
    print(f"Warning: {BERTWARPER_FILE} not found. Patch skipped. This may cause an AttributeError later.")


from groundingdino.util.inference import load_model, load_image, predict
import torchvision.ops as ops

dino_model = load_model(
    "groundingdino_repo/groundingdino/config/GroundingDINO_SwinT_OGC.py",
    "weights/groundingdino_swint_ogc.pth"
)
print("Grounding DINO loaded")


def grid_based_labeling(img_path, expected_cols=None):
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 30, 100)

    h_proj = edges.sum(axis=1).astype(float) / edges.sum(axis=1).max()
    v_proj = edges.sum(axis=0).astype(float) / edges.sum(axis=0).max()

    def find_splits_from_peaks(proj, min_gap):
        """Find object BOUNDARIES as peaks in edge density"""
        smooth = np.convolve(proj, np.ones(30)/30, mode='same')
        threshold = smooth.mean() * 0.6
        splits = [0]
        for i in range(1, len(smooth) - 1):
            if (smooth[i] > smooth[i-1] and smooth[i] > smooth[i+1]
                    and smooth[i] > threshold
                    and i - splits[-1] > min_gap):
                splits.append(i)
        splits.append(len(proj))
        return splits

    row_splits = find_splits_from_peaks(h_proj, min_gap=h//8)
    col_splits = find_splits_from_peaks(v_proj, min_gap=w//8)

    # Fallback: if peaks find nothing, use a fixed adaptive grid
    if len(row_splits) <= 2 and len(col_splits) <= 2:
        aspect = w / h
        if aspect > 1.5:
            n_rows, n_cols = 2, 4
        elif aspect < 0.7:
            n_rows, n_cols = 4, 2
        else:
            n_rows, n_cols = 3, 3
        row_splits = [i * h // n_rows for i in range(n_rows+1)]
        col_splits = [i * w // n_cols for i in range(n_cols+1)]
        print(f"  Adaptive fixed grid: {n_rows}×{n_cols}")

    boxes = []
    pad = 5
    for r in range(len(row_splits) - 1):
        for c in range(len(col_splits) - 1):
            x1, y1 = col_splits[c]+pad, row_splits[r]+pad
            x2, y2 = col_splits[c+1]-pad, row_splits[r+1]-pad
            if (x2-x1)*(y2-y1) > w*h*0.01:
                boxes.append([x1, y1, x2, y2])

    print(f"  Peak-based grid: {len(row_splits)-1}r × {len(col_splits)-1}c = {len(boxes)} boxes")
    return boxes, h, w

!pip install -q segment-anything
!wget -q -nc https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth -O /content/sam_vit_b_01ec64.pth
import torch
import numpy as np
import cv2
import shutil
from pathlib import Path
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

SAM_CHECKPOINT = '/content/sam_vit_b_01ec64.pth'
sam = sam_model_registry["vit_b"](checkpoint=SAM_CHECKPOINT)
sam.to(device='cuda' if torch.cuda.is_available() else 'cpu')

mask_generator = SamAutomaticMaskGenerator(
    model                  = sam,
    points_per_side        = 8, 
    points_per_batch       = 32,
    pred_iou_thresh        = 0.86,
    stability_score_thresh = 0.92,
    box_nms_thresh         = 0.5,
    min_mask_region_area   = 500,
)

def nms_boxes(boxes, iou_thresh=0.4):
    if len(boxes) == 0:
        return boxes
    boxes = np.array(boxes)
    x1, y1 = boxes[:,0], boxes[:,1]
    x2, y2 = boxes[:,2], boxes[:,3]
    areas  = (x2-x1) * (y2-y1)
    order  = areas.argsort()[::-1]
    keep   = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return boxes[keep].tolist()


def relabel_with_sam(img_path, min_area_ratio=0.015, max_area_ratio=0.55):
    img_bgr = cv2.imread(str(img_path))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    H, W    = img_rgb.shape[:2]
    total   = H * W

    masks = mask_generator.generate(img_rgb)

    raw_boxes = []
    for m in masks:
        ratio = m['area'] / total
        if min_area_ratio < ratio < max_area_ratio:
            x, y, w, h = m['bbox']
            raw_boxes.append([x, y, x+w, y+h])

    if not raw_boxes:
        return []

    clean_boxes = nms_boxes(raw_boxes, iou_thresh=0.4)

    yolo_lines = []
    for (x1, y1, x2, y2) in clean_boxes:
        cx = ((x1+x2)/2) / W
        cy = ((y1+y2)/2) / H
        nw = (x2-x1) / W
        nh = (y2-y1) / H
        yolo_lines.append(f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}")

    return yolo_lines

BACKUP_DIR = DATASET_BASE / 'labels_backup'
if not BACKUP_DIR.exists():
    shutil.copytree(str(DATASET_BASE / 'labels'), str(BACKUP_DIR))
    print(f"Backed up original labels → {BACKUP_DIR}")
else:
    print(f"Backup already exists → {BACKUP_DIR}")

split_map = {
    'train': (IMAGE_TRAIN, LABEL_TRAIN),
    'val'  : (IMAGE_VAL,   LABEL_VAL),
}

stats = {'relabeled': 0, 'kept': 0, 'sam_empty': 0}

for split, (img_dir, lbl_dir) in split_map.items():
    print(f"\n── Processing {split.upper()} ──────────────────────")

    for lbl_path in sorted(lbl_dir.glob('*.txt')):
        current_lines = [l.strip() for l in
                         lbl_path.read_text().strip().splitlines()
                         if l.strip()]
        n_boxes = len(current_lines)

        if n_boxes > 2:
            stats['kept'] += 1
            continue

        img_path = None
        for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG']:
            candidate = img_dir / (lbl_path.stem + ext)
            if candidate.exists():
                img_path = candidate
                break

        if img_path is None:
            print(f"  Image not found for {lbl_path.name}, skipping")
            continue

        new_lines = relabel_with_sam(img_path)

        if len(new_lines) <= 1:
            stats['sam_empty'] += 1
            print(f"  Single (SAM confirmed): {lbl_path.name} → keeping {n_boxes} box")
            continue
          
        lbl_path.write_text('\n'.join(new_lines))
        stats['relabeled'] += 1
        print(f"  Fixed: {lbl_path.name}  {n_boxes} box → {len(new_lines)} boxes")

#Summary
print(f"\n{'='*50}")
print(f"Relabeled (fixed)  : {stats['relabeled']} images")
print(f"Kept (already ok)  : {stats['kept']} images")
print(f"SAM confirmed single: {stats['sam_empty']} images")

print(f"\n── Label quality check after SAM relabeling ───")
for split, (_, lbl_dir) in split_map.items():
    counts = [
        len([l for l in p.read_text().strip().splitlines() if l.strip()])
        for p in lbl_dir.glob('*.txt')
    ]
    counts = np.array(counts)
    print(f"\n  {split.upper()}")
    print(f"    Total instances : {counts.sum()}")
    print(f"    Avg per image   : {counts.mean():.1f}")
    print(f"    Images 1 box    : {(counts == 1).sum()}")
    print(f"    Images 5+ boxes : {(counts >= 5).sum()}")
