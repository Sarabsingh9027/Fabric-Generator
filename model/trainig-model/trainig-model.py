
import shutil, random, yaml
all_labeled = [
    f for f in IMAGE_FRESH.iterdir()
    if f.suffix.lower() in {'.jpg','.jpeg','.png'}
    and (LABEL_FRESH / (f.stem + ".txt")).exists()
]

random.seed(42)
random.shuffle(all_labeled)
split = int(len(all_labeled) * 0.8)
train_imgs = all_labeled[:split]
val_imgs   = all_labeled[split:]

print(f"Train: {len(train_imgs)}  |  Val: {len(val_imgs)}")

def copy_split(img_list, split_name):
    img_out = DATASET_BASE / "images" / split_name
    lbl_out = DATASET_BASE / "labels" / split_name
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)
    for img_path in img_list:
        lbl_path = LABEL_FRESH / (img_path.stem + ".txt")
        shutil.copy(img_path, img_out / img_path.name)
        shutil.copy(lbl_path, lbl_out / (img_path.stem + ".txt"))

copy_split(train_imgs, "train")
copy_split(val_imgs,   "val")
print("Split complete!")

#cell2
config = {
    'path': str(DATASET_BASE),
    'train': 'images/train',
    'val': 'images/val',
    'nc': 1,
    'names': ['fabric_roll']
}

CONFIG_PATH = "/content/dataset.yaml"
with open(CONFIG_PATH, 'w') as f:
    yaml.dump(config, f, default_flow_style=False)

with open(CONFIG_PATH) as f:
    print(f.read())


from ultralytics import YOLO
model = YOLO('yolov8s.pt')

results = model.train(
    data          = CONFIG_PATH,
    epochs        = 150,
    imgsz         = 640,
    batch         = 16,       
    device        = 0,
    patience      = 30,
    name          = 'fabric_v2',
    project       = str(RUNS_DIR),
    lr0           = 0.005,
    lrf           = 0.01,
    warmup_epochs = 5,
    weight_decay  = 0.0005,
    mosaic        = 1.0,
    close_mosaic  = 15,        
    fliplr        = 0.5,       
    flipud        = 0.2,       
    degrees       = 5.0,       
    hsv_h         = 0.01,      
    hsv_s         = 0.2,      
    hsv_v         = 0.25,      
)

print("Training complete")

best_model = YOLO(str(RUNS_DIR / 'fabric_v1/weights/best.pt'))
metrics    = best_model.val(data=CONFIG_PATH)

print(f"mAP50     : {metrics.box.map50:.3f}")
print(f"mAP50-95  : {metrics.box.map:.3f}")
print(f"Precision : {metrics.box.mp:.3f}")
print(f"Recall    : {metrics.box.mr:.3f}")

import matplotlib.image as mpimg
img = mpimg.imread(str(RUNS_DIR / 'fabric_v1/results.png'))
plt.figure(figsize=(14, 6))
plt.imshow(img)
plt.axis('off')
plt.show()

#cell5
test_img = list((DATASET_BASE / "images/val").iterdir())[0]
result   = best_model(str(test_img), conf=0.4)[0]
annotated = result.plot()
plt.figure(figsize=(10, 7))
plt.imshow(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB))
plt.title(f"Detected: {len(result.boxes)} fabric rolls")
plt.axis('off')
plt.show()
