import io
import base64
import numpy as np
import cv2
from PIL import Image
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
import torch
from sklearn.cluster import KMeans
import math, ast
import importlib

try:
    classify_image = importlib.import_module("classifier").classify_image
except (ImportError, AttributeError):
    def classify_image(img_np):
        return 'single'

app = FastAPI(title="Fabric Detection API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_methods=["*"],
    allow_headers=["*"],
)

#Load model once at startup 
MODEL_PATH = "best.pt"
_original_torch_load = torch.load


def _torch_load_with_weights_only_disabled(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(*args, **kwargs)


torch.load = _torch_load_with_weights_only_disabled
try:
    model = YOLO(MODEL_PATH)
finally:
    torch.load = _original_torch_load

print(f"Model loaded: {MODEL_PATH}")


# Helpers 
def read_image(upload: UploadFile) -> np.ndarray:
    data = upload.file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")
    return np.array(img)

def np_to_b64(img_np: np.ndarray) -> str:
    _, buf = cv2.imencode(".jpg", cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR))
    return base64.b64encode(buf).decode()

def get_dominant_color(crop_np: np.ndarray):
    pixels = crop_np.reshape(-1, 3).astype(np.float32)
    brightness = pixels.mean(axis=1)
    pixels = pixels[brightness < 230]
    if len(pixels) < 10:
        return (128, 128, 128)
    km = KMeans(n_clusters=1, n_init=5, random_state=42)
    km.fit(pixels)
    return tuple(km.cluster_centers_[0].astype(int).tolist())

def rgb_to_hex(rgb):
    return '#{:02X}{:02X}{:02X}'.format(*[int(x) for x in rgb])

def nms_boxes(boxes, iou_thresh=0.4):
    if len(boxes) == 0:
        return boxes
    boxes = np.array(boxes)
    x1, y1 = boxes[:,0], boxes[:,1]
    x2, y2 = boxes[:,2], boxes[:,3]
    areas = (x2 - x1) * (y2 - y1)
    order = areas.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, xx2-xx1) * np.maximum(0, yy2-yy1)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou < iou_thresh]
    return boxes[keep].tolist()


#Routes
@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_PATH}


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    img_np = read_image(file)
    label = classify_image(img_np)
    return {"image_type": label}


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    img_np = read_image(file)
    H, W = img_np.shape[:2]
    img_type = classify_image(img_np)

    conf = 0.20 if img_type == 'single' else 0.25
    iou  = 0.50 if img_type == 'single' else 0.30

    results = model(img_np, conf=conf, iou=iou)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int).tolist()
    confs = results.boxes.conf.cpu().numpy().tolist()

    # fallback for single with no detection
    if img_type == 'single' and len(boxes) == 0:
        pad = int(min(H, W) * 0.03)
        boxes = [[pad, pad, W - pad, H - pad]]
        confs = [1.0]

    annotated = img_np.copy()
    for i, (box, c) in enumerate(zip(boxes, confs)):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 200, 80), 2)
        cv2.putText(annotated, f"#{i+1} {c:.2f}",
                    (x1, max(y1 - 8, 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 200, 80), 2)

    return {
        "image_type": img_type,
        "count":      len(boxes),
        "boxes":      boxes,
        "confidences": confs,
        "annotated_image_b64": np_to_b64(annotated)
    }


@app.post("/extract-colors")
async def extract_colors(file: UploadFile = File(...)):
    img_np = read_image(file)
    H, W = img_np.shape[:2]
    img_type = classify_image(img_np)

    conf = 0.20 if img_type == 'single' else 0.25
    iou  = 0.50 if img_type == 'single' else 0.30
    results = model(img_np, conf=conf, iou=iou)[0]
    boxes = results.boxes.xyxy.cpu().numpy().astype(int).tolist()
    confs = results.boxes.conf.cpu().numpy().tolist()

    if img_type == 'single' and len(boxes) == 0:
        pad = int(min(H, W) * 0.03)
        boxes = [[int(pad), int(pad), int(W - pad), int(H - pad)]]
        confs = [1.0]

    color_data = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(W, x2), min(H, y2)
        crop = img_np[y1:y2, x1:x2]
        if crop.size == 0:
            continue
        rgb = get_dominant_color(crop)
        hex_code = rgb_to_hex(rgb)
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(crop, cv2.COLOR_RGB2BGR))
        crop_b64 = base64.b64encode(buf).decode()
        color_data.append({
            "crop_id":   i + 1,
            "rgb":       rgb,
            "hex":       hex_code,
            "bbox":      box,
            "crop_b64":  crop_b64,
            "confidence": round(confs[i], 3) if i < len(confs) else 1.0
        })

    return {"image_type": img_type, "count": len(color_data), "colors": color_data}


@app.post("/generate-variations")
async def generate_variations(
    single_file: UploadFile = File(...),
    group_file:  UploadFile = File(...)
):
    single_np = read_image(single_file)
    group_np  = read_image(group_file)
    H_s, W_s  = single_np.shape[:2]

    # Detect single fabric box
    res_s = model(single_np, conf=0.20, iou=0.50)[0]
    boxes_s = res_s.boxes.xyxy.cpu().numpy().astype(int).tolist()
    if not boxes_s:
        pad = int(min(H_s, W_s) * 0.03)
        boxes_s = [[pad, pad, W_s - pad, H_s - pad]]

    x1, y1, x2, y2 = boxes_s[0]
    fabric_crop = single_np[y1:y2, x1:x2]

    # Extract colors from group image
    H_g, W_g = group_np.shape[:2]
    res_g = model(group_np, conf=0.25, iou=0.30)[0]
    boxes_g = res_g.boxes.xyxy.cpu().numpy().astype(int).tolist()

    target_colors = []
    for i, box in enumerate(boxes_g):
        bx1, by1, bx2, by2 = box
        crop = group_np[max(0,by1):min(H_g,by2), max(0,bx1):min(W_g,bx2)]
        if crop.size == 0:
            continue
        rgb = get_dominant_color(crop)
        target_colors.append({"id": i + 1, "rgb": rgb, "hex": rgb_to_hex(rgb)})

    # Generate recolored variations
    variations = []
    for entry in target_colors:
        recolored = recolor_fabric(fabric_crop, entry["rgb"])
        _, buf = cv2.imencode(".jpg", cv2.cvtColor(recolored, cv2.COLOR_RGB2BGR))
        variations.append({
            "id":        entry["id"],
            "rgb":       entry["rgb"],
            "hex":       entry["hex"],
            "image_b64": base64.b64encode(buf).decode()
        })

    _, orig_buf = cv2.imencode(".jpg", cv2.cvtColor(fabric_crop, cv2.COLOR_RGB2BGR))
    return {
        "original_b64": base64.b64encode(orig_buf).decode(),
        "variations":   variations
    }


# Recolor helper 
def boost_color_saturation(rgb, sat_mult=1.6, min_sat=120):
    px = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[0,0,1] = max(min_sat, min(255, hsv[0,0,1] * sat_mult))
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return (int(bgr[0,0,2]), int(bgr[0,0,1]), int(bgr[0,0,0]))

def get_fabric_mask(img_bgr, sat_thresh=40, val_max=242):
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:,:,1] > sat_thresh) & (hsv[:,:,2] < val_max)).astype(np.uint8)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    return mask.astype(bool)

def recolor_fabric(fabric_np, target_rgb):
    fabric_bgr = cv2.cvtColor(fabric_np, cv2.COLOR_RGB2BGR)
    fabric_hsv = cv2.cvtColor(fabric_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    fabric_mask = get_fabric_mask(fabric_bgr)
    gray = cv2.cvtColor(fabric_bgr, cv2.COLOR_BGR2GRAY)
    light_mask = gray > 185
    recolor_mask = fabric_mask & ~light_mask

    if not recolor_mask.any():
        fabric_mask  = get_fabric_mask(fabric_bgr, sat_thresh=20)
        recolor_mask = fabric_mask & ~light_mask

    target_vivid = boost_color_saturation(target_rgb)
    t_px  = np.array([[[target_vivid[2], target_vivid[1], target_vivid[0]]]], dtype=np.uint8)
    t_hsv = cv2.cvtColor(t_px, cv2.COLOR_BGR2HSV)[0,0].astype(np.float32)

    orig_v      = fabric_hsv[:,:,2]
    fabric_vals = orig_v[recolor_mask]
    mean_orig_v = float(fabric_vals.mean()) if len(fabric_vals) > 0 else 128.0
    v_deviation = orig_v.astype(float) - mean_orig_v
    new_v = np.clip(t_hsv[2] + v_deviation * 0.75, 0, 255)

    result_hsv = fabric_hsv.copy()
    result_hsv[:,:,0][recolor_mask] = t_hsv[0]
    result_hsv[:,:,1][recolor_mask] = np.clip(t_hsv[1], 110, 255)
    result_hsv[:,:,2][recolor_mask] = new_v[recolor_mask]

    result_bgr = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
