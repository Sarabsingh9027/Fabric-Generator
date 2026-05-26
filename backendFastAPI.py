import cv2
import numpy as np

def classify_image(img_np: np.ndarray) -> str:
    """Accepts RGB numpy array, returns 'single' or 'group'."""
    h, w = img_np.shape[:2]
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    strip_hues = []
    for i in range(8):
        x1, x2 = i * w // 8, (i + 1) * w // 8
        strip = img_hsv[h // 4: 3 * h // 4, x1:x2]
        strip_hues.append(float(np.median(strip[:, :, 0])))
    hue_std = np.std(strip_hues)

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    sobel_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
    sobel_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
    vert_edge_ratio = np.abs(sobel_x).mean() / (np.abs(sobel_y).mean() + 1e-5)

    mid_row = img_hsv[h // 2, :, 0].astype(float)
    diffs_smooth = np.convolve(np.abs(np.diff(mid_row)), np.ones(15) / 15, mode='same')
    color_transitions = int((diffs_smooth > 15).sum())

    votes_group = sum([
        hue_std > 12,
        vert_edge_ratio > 1.15,
        color_transitions > w * 0.08
    ])
    return 'group' if votes_group >= 2 else 'single'