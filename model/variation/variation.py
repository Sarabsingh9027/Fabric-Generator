import cv2, math, ast
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

def clean_rgb(rgb):
    if isinstance(rgb, str):
        rgb = ast.literal_eval(rgb)
    return tuple(int(x) for x in rgb)

def get_fabric_mask(img_bgr, sat_thresh=40, val_max=242):
   
    hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask = ((hsv[:,:,1] > sat_thresh) & (hsv[:,:,2] < val_max)).astype(np.uint8)
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))
    k_open  = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,  7))
    mask    = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask    = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k_open)
    return mask.astype(bool)

def boost_color_saturation(rgb, sat_mult=1.6, min_sat=120):
    """
    Ensure target color is vivid enough to produce distinct variations.
    Muted extracted colors (sat < 80) become washed-out results without this.
    """
    px  = np.array([[[rgb[2], rgb[1], rgb[0]]]], dtype=np.uint8)
    hsv = cv2.cvtColor(px, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[0,0,1] = max(min_sat, min(255, hsv[0,0,1] * sat_mult))
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return (int(bgr[0,0,2]), int(bgr[0,0,1]), int(bgr[0,0,0]))  # RGB

def recolor_fabric(fabric_np, target_rgb,
                   light_threshold=185, sat_thresh=40):
    fabric_bgr = cv2.cvtColor(fabric_np, cv2.COLOR_RGB2BGR)
    fabric_hsv = cv2.cvtColor(fabric_bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    fabric_mask = get_fabric_mask(fabric_bgr, sat_thresh=sat_thresh)
    gray        = cv2.cvtColor(fabric_bgr, cv2.COLOR_BGR2GRAY)
    light_mask  = gray > light_threshold
    recolor_mask = fabric_mask & ~light_mask

    if not recolor_mask.any():
        print("  Warning: fabric mask empty, lowering sat_thresh to 20")
        fabric_mask  = get_fabric_mask(fabric_bgr, sat_thresh=20)
        recolor_mask = fabric_mask & ~light_mask
    target_vivid = boost_color_saturation(target_rgb, sat_mult=1.6, min_sat=120)
    t_px  = np.array([[[target_vivid[2], target_vivid[1], target_vivid[0]]]], dtype=np.uint8)
    t_hsv = cv2.cvtColor(t_px, cv2.COLOR_BGR2HSV)[0,0].astype(np.float32)
    orig_v       = fabric_hsv[:,:,2]
    fabric_vals  = orig_v[recolor_mask]
    mean_orig_v  = float(fabric_vals.mean()) if len(fabric_vals) > 0 else 128.0
    v_deviation  = orig_v.astype(float) - mean_orig_v   # texture = deviation from mean
    new_v = np.clip(t_hsv[2] + v_deviation * 0.75, 0, 255)
    result_hsv = fabric_hsv.copy()
    result_hsv[:,:,0][recolor_mask] = t_hsv[0]                      
    result_hsv[:,:,1][recolor_mask] = np.clip(t_hsv[1], 110, 255)   
    result_hsv[:,:,2][recolor_mask] = new_v[recolor_mask]        

    result_bgr = cv2.cvtColor(result_hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    return cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)


# Main-generate and display variations 
def generate_variations(single_img_np, single_boxes, group_results,
                        save_base='/content/drive/MyDrive/foto'):

    # Crop single fabric
    if len(single_boxes) > 0:
        x1, y1, x2, y2 = single_boxes[0]
        pad = 6
        x1 = max(0, x1+pad);  y1 = max(0, y1+pad)
        x2 = min(single_img_np.shape[1], x2-pad)
        y2 = min(single_img_np.shape[0], y2-pad)
        fabric_crop = single_img_np[y1:y2, x1:x2]
    else:
        fabric_crop = single_img_np

    # Parse and clean colors
    entries = []
    for r in group_results:
        rgb = clean_rgb(r['rgb'])   
        entries.append({
            'rgb': rgb,
            'hex': r['hex'],
            'crop_id': r['crop_id']
        })

    print(f"Fabric crop : {fabric_crop.shape}")
    print(f"Variations  : {len(entries)}")

    #Generate variations
    results = []
    for e in entries:
        var = recolor_fabric(fabric_crop, e['rgb'])
        results.append({**e, 'image': var})
        print(f"  Variation {e['crop_id']:2d} | {e['hex']} | RGB={e['rgb']}")
        Image.fromarray(var).save(
            f"{save_base}/variation_{e['crop_id']}_{e['hex'].replace('#','')}.jpg",
            quality=95
        )

    #Visualization 
    n = len(results)
    n_cols = 3
    n_rows = math.ceil((n + 1) / n_cols)  
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 5, n_rows * 5.2),
        squeeze=False
    )

    #Slot 0: Original
    axes[0][0].imshow(fabric_crop)
    axes[0][0].set_title("ORIGINAL", fontsize=11, fontweight='bold', pad=8)
    axes[0][0].axis('off')

    # Variations
    for i, v in enumerate(results):
        slot = i + 1
        r, c = slot // n_cols, slot % n_cols
        ax = axes[r][c]
        ax.imshow(v['image'])
      
        rgb_norm = tuple(x/255 for x in v['rgb'])
        ax.set_title(
            f"Variation {v['crop_id']}   {v['hex']}\n"
            f"RGB = {v['rgb']}",
            fontsize=8, pad=4
        )
        inset = ax.inset_axes([0.72, 0.02, 0.26, 0.12])
        inset.set_facecolor(rgb_norm)
        inset.set_xticks([]); inset.set_yticks([])
        for spine in inset.spines.values():
            spine.set_edgecolor('white'); spine.set_linewidth(1.5)
        ax.axis('off')
    for j in range(n + 1, n_rows * n_cols):
        axes[j // n_cols][j % n_cols].axis('off')

    plt.suptitle(
        f"Single fabric  ·  {n} color variations  ·  colors from group image",
        fontsize=12, y=1.01
    )
    plt.tight_layout(pad=1.5)
    grid_path = f'{save_base}/fabric_variations_v2.png'
    plt.savefig(grid_path, dpi=130, bbox_inches='tight')
    plt.show()
    print(f"\nSaved → {grid_path}")
    return results
                          
# RUN 
variations = generate_variations(
    single_img_np = img_np_single,
    single_boxes  = boxes_single,
    group_results = results_grouped,
    save_base     = '/content/drive/MyDrive/foto'
)
