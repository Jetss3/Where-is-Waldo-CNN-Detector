import numpy as np
import matplotlib.pyplot as plt


def int_rnd(num):
    return np.round(num).astype(int)


def sample_scale():
    return np.random.uniform(0.9, 1.1)


# Crop full image around a Wally's bbox
def crop_positive(img, bbox, sub_w, sub_h, severe_jitter=True):

    h, w = img.shape[:2]

    x1, y1, x2, y2 = bbox

    cx = int((x1 + x2) / 2)
    cy = int((y1 + y2) / 2)


    if severe_jitter:
        jitter_x = np.random.randint(-sub_w // 3, sub_w // 3)
        jitter_y = np.random.randint(-sub_h // 3, sub_h // 3)
    else:
        jitter_x = np.random.randint(-sub_w // 8, sub_w // 8)
        jitter_y = np.random.randint(-sub_h // 8, sub_h // 8)

    cx += jitter_x
    cy += jitter_y

    x1 = np.clip(cx - sub_w // 2, 0, w - sub_w)
    y1 = np.clip(cy - sub_h // 2, 0, h - sub_h)

    return x1, y1, x1 + sub_w, y1 + sub_h


# Crop full image randomly
def random_crop(img, sub_w, sub_h):

    h, w = img.shape[:2]

    if w <= sub_w or h <= sub_h:
        raise ValueError("Image smaller than crop size")

    x1 = np.random.randint(0, w - sub_w)
    y1 = np.random.randint(0, h - sub_h)

    return x1, y1, x1 + sub_w, y1 + sub_h


# Compute bbox overlap in pascal_voc
def visibility_ratio(crop, bbox):

    cx1, cy1, cx2, cy2 = crop
    bx1, by1, bx2, by2 = bbox

    # intersection rectangle
    ix1 = max(cx1, bx1)
    iy1 = max(cy1, by1)
    ix2 = min(cx2, bx2)
    iy2 = min(cy2, by2)

    # no overlap
    if ix2 <= ix1 or iy2 <= iy1:
        return 0.0

    intersection_area = (ix2 - ix1) * (iy2 - iy1)
    bbox_area = (bx2 - bx1) * (by2 - by1)

    if bbox_area <= 0:
        return 0.0

    return intersection_area / bbox_area

def plot_history(history):
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    plt.figure(figsize=(18, 5))

    # LOSS
    plt.subplot(1, 3, 1)
    plt.plot(epochs, hist["loss"], label="Train Loss")
    plt.plot(epochs, hist["val_loss"], label="Val Loss")
    plt.title("Loss Batch Ranking Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    # MAE
    if "mae" in hist:
        plt.subplot(1, 3, 2)
        plt.plot(epochs, hist["mae"], label="Train MAE")
        plt.plot(epochs, hist["val_mae"], label="Val MAE")
        plt.title("Visibility MAE")
        plt.xlabel("Epoch")
        plt.ylabel("MAE")
        plt.legend()

    # TOP-1 SOFT
    if "top1_soft" in hist:
        plt.subplot(1, 3, 3)
        plt.plot(epochs, hist["top1_soft"], label="Train Top-1")
        plt.plot(epochs, hist["val_top1_soft"], label="Val Top-1")
        plt.title("Top-1 Soft Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()

    plt.tight_layout()
    plt.show()