import numpy as np
import albumentations as A
import cv2
from SupportFunctions import sample_scale, crop_positive, random_crop, visibility_ratio, int_rnd

# Generate batches of 64
def image_batch_generator(maybe_wally, sub_w=128, sub_h=128, num_crops=64):

    data = maybe_wally

    # Random transform for physical picture differences
    transform = A.Compose([
        A.RandomBrightnessContrast(0.3, 0.3, p=0.6),
        A.HueSaturationValue(20, 30, 20, p=0.5),
        A.GaussNoise(p=0.3),
        A.GaussianBlur(blur_limit=3, p=0.2),
    ])

    while True:
        # Take a random sample from the full images data
        sample = data[np.random.randint(len(data))]

        img = sample["image"]
        boxes = sample["bboxes"]

        crops = []
        labels = []

        max_vis = 0.

        for _ in range(num_crops):

            scale = sample_scale()
            crop_w = int_rnd(sub_w * scale)
            crop_h = int_rnd(sub_h * scale)

            vis = 0.

            # Split positives (positive crop) and negatives (random crop) at 3-7 ratio
            if np.random.rand() < 0.3:
                box = boxes[np.random.randint(len(boxes))]
                x1, y1, x2, y2 = crop_positive(img, box, crop_w, crop_h)
                vis = visibility_ratio([x1, y1, x2, y2], box)

            else:
                x1, y1, x2, y2 = random_crop(img, crop_w, crop_h)
                for box in boxes:
                    vis = max(visibility_ratio([x1, y1, x2, y2], box), vis)

            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (sub_w, sub_h))

            crop = transform(image=crop)["image"]

            crop = crop.astype(np.float32)

            # Sigmoid function to deter images with wally with low vis aka no distinguishable features
            vis = 1 / (1 + np.exp(-8 * (vis - 0.6)))

            max_vis = max(max_vis, vis)

            crops.append(crop.astype(np.float32))
            labels.append(vis)

        if max_vis < 0.95:
            scale = sample_scale()
            crop_w = int_rnd(sub_w * scale)
            crop_h = int_rnd(sub_h * scale)

            box = boxes[np.random.randint(len(boxes))]
            x1, y1, x2, y2 = crop_positive(img, box, crop_w, crop_h, severe_jitter=False)
            vis = visibility_ratio([x1, y1, x2, y2], box)

            crop = img[y1:y2, x1:x2]
            crop = cv2.resize(crop, (sub_w, sub_h))

            crop = transform(image=crop)["image"]

            crop = crop.astype(np.float32)

            # your shaping
            vis = 1 / (1 + np.exp(-12 * (vis - 0.5)))

            random_index = int(np.random.randint(num_crops))

            crops[random_index] = crop.astype(np.float32)
            labels[random_index] = vis

        # Shuffle batch to maximize randomness

        # Random order
        indices = np.random.permutation(len(crops))

        # Apply order to images and labels
        crops = np.array(crops)[indices]
        labels = np.array(labels, dtype=np.float32)[indices]

        # Yield results and continue on next call
        yield np.stack(crops), np.array(labels, dtype=np.float32)