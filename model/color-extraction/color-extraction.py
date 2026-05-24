
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches

MODEL_PATH = str(RUNS_DIR / 'fabric_v1/weights/best.pt')
best_model = YOLO(MODEL_PATH)

def classify_image(img_path):
    """Same classifier used during labeling"""
    img          = cv2.imread(str(img_path))
    h, w         = img.shape[:2]
    aspect       = w / h
    gray         = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges        = cv2.Canny(gray, 50, 150)
    edge_density = edges.mean()
    if edge_density > 18 or aspect > 1.2:
        return 'group'
    return 'single'

def detect_fabrics(image_path, show=True):
    img_type = classify_image(str(image_path))
    img      = Image.open(image_path).convert("RGB")
    img_np   = np.array(img)
    h, w     = img_np.shape[:2]

    print(f"Image type detected: {img_type.upper()}")

    if img_type == 'single':
        conf      = 0.20   
        iou       = 0.50
    else:
        conf      = 0.25    
        iou       = 0.30   

    results = best_model(img_np, conf=conf, iou=iou)[0]
    boxes   = results.boxes.xyxy.cpu().numpy().astype(int)
    confs   = results.boxes.conf.cpu().numpy()
  
    if img_type == 'single' and len(boxes) == 0:
        print("Fallback: no detection → using full image as 1 fabric")
        pad  = int(min(h, w) * 0.03)
        boxes = np.array([[pad, pad, w-pad, h-pad]])
        confs = np.array([1.0])

    #Draw boxes 
    annotated = img_np.copy()
    for i, (box, c) in enumerate(zip(boxes, confs)):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 80), 2)
        cv2.putText(
            annotated, f"#{i+1}  {c:.2f}",
            (x1, max(y1-8, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 2
        )

    if show:
        plt.figure(figsize=(12, 8))
        plt.imshow(annotated)
        plt.title(
            f"Type: {img_type.upper()}  |  "
            f"Fabric count: {len(boxes)}  |  "
            f"conf≥{conf}  iou≤{iou}",
            fontsize=12
        )
        plt.axis('off')
        plt.tight_layout()
        plt.show()

    print(f"\nObject count : {len(boxes)}")
    print(f"Conf scores  : {[round(float(c),2) for c in confs]}")
    return boxes, img_np, img_type

from google.colab import files
from PIL import Image
import numpy as np
import io

#Upload images from local computer 
print("Upload your SINGLE fabric image:")
single_upload = files.upload()
single_image_path = f"/content/{list(single_upload.keys())[0]}"
with open(single_image_path, 'wb') as f:
    f.write(list(single_upload.values())[0])
print(f"Uploaded: {list(single_upload.keys())[0]} ({len(list(single_upload.values())[0])/1024:.1f} KB)")

print("\nUpload your GROUPED fabric image:")
grouped_upload = files.upload()
grouped_image_path = f"/content/{list(grouped_upload.keys())[0]}"
with open(grouped_image_path, 'wb') as f:
    f.write(list(grouped_upload.values())[0])
print(f"Uploaded: {list(grouped_upload.keys())[0]} ({len(list(grouped_upload.values())[0])/1024:.1f} KB)")

print(f"\nSingle image path: {single_image_path}")
print(f"Grouped image path: {grouped_image_path}")

from sklearn.cluster import KMeans
import pandas as pd
import math
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_dominant_color(crop_np):
    pixels = crop_np.reshape(-1, 3).astype(np.float32)
    brightness = pixels.mean(axis=1)
    pixels = pixels[brightness < 230]
    if len(pixels) < 10:
        return (128, 128, 128)
    km = KMeans(n_clusters=1, n_init=5, random_state=42)
    km.fit(pixels)
    return tuple(km.cluster_centers_[0].astype(int))

def rgb_to_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(*rgb)

def extract_colors(image_np, boxes, image_type=''):
    if len(boxes) == 0:
        print("No boxes found.")
        return []

    results = []
    crops, rgbs, hexes, valid_ids = [], [], [], []

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1 = max(0, x1); y1 = max(0, y1)
        x2 = min(image_np.shape[1], x2)
        y2 = min(image_np.shape[0], y2)
        crop = image_np[y1:y2, x1:x2]

        if crop.size == 0 or crop.shape[0] < 5 or crop.shape[1] < 5:
            continue

        rgb = get_dominant_color(crop)
        hex_code = rgb_to_hex(rgb)
        crops.append(crop)
        rgbs.append(rgb)
        hexes.append(hex_code)
        valid_ids.append(i + 1)

        results.append({
            'crop_id': i + 1,
            'image_type': image_type,
            'rgb': tuple(int(val) for val in rgb), # Store as a tuple of native ints
            'hex': hex_code,
            'bbox': box.tolist()
        })
        print(f"Crop {i+1:2d} | RGB={rgb} | HEX={hex_code}")

    n = len(crops)
    if n == 0:
        print("No valid crops.")
        return results

    if n <= 6:
        items_per_row = 3
    elif n <= 12:
        items_per_row = 4
    elif n <= 20:
        items_per_row = 5
    else:
        items_per_row = 6

    n_rows = math.ceil(n / items_per_row)

    fig, axes = plt.subplots(
        n_rows, items_per_row * 2,
        figsize=(items_per_row * 5, n_rows * 3.5),
        squeeze=False
    )

    for i in range(n):
        row = i // items_per_row
        col = (i % items_per_row) * 2

        axes[row][col].imshow(crops[i])
        axes[row][col].set_title(f"Crop {valid_ids[i]}", fontsize=9, pad=3)
        axes[row][col].axis('off')
        swatch = np.ones((60, 80, 3), dtype=np.uint8)
        swatch[:] = rgbs[i]
        axes[row][col+1].imshow(swatch)
        axes[row][col+1].set_title(
            f"{hexes[i]}\n{rgbs[i]}", fontsize=7, pad=3
        )
        axes[row][col+1].axis('off')

    # Hide unused slots
    for j in range(n, n_rows * items_per_row):
        row = j // items_per_row
        col = (j % items_per_row) * 2
        axes[row][col].axis('off')
        axes[row][col+1].axis('off')

    plt.suptitle(
        f"{image_type.upper()}  |  "
        f"{n} fabrics  |  "
        f"grid {items_per_row} per row",
        fontsize=12, y=1.01
    )
    plt.tight_layout()

    save_path = '/content/drive/MyDrive/foto/color_results.png'
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    plt.show()
    print(f"\nGrid: {n_rows} rows × {items_per_row} cols  |  Saved → {save_path}")
    return results
  
all_results = []

#Process the single image
print(f"\n{'─'*50}")
print(f"Processing: {single_image_path}")
boxes_single, img_np_single, img_type_single = detect_fabrics(single_image_path)
print(f"\n--- Color extraction ({img_type_single}) ---")
results_single = extract_colors(img_np_single, boxes_single, img_type_single)
all_results.extend(results_single)

#Process the grouped image
print(f"\n{'─'*50}")
print(f"Processing: {grouped_image_path}")
boxes_grouped, img_np_grouped, img_type_grouped = detect_fabrics(grouped_image_path)
print(f"\n--- Color extraction ({img_type_grouped}) ---")
results_grouped = extract_colors(img_np_grouped, boxes_grouped, img_type_grouped)
all_results.extend(results_grouped)

if all_results:
    import pandas as pd
    df = pd.DataFrame(all_results)
    csv_path = '/content/drive/MyDrive/foto/detection_results.csv'
    df.to_csv(csv_path, index=False)
    print(f"\nCSV saved → {csv_path}")
    print(df[['crop_id', 'image_type', 'hex', 'rgb']].to_string(index=False))
else:
    print("No results to save.")
