import cv2
import numpy as np
import os
import json

from pydantic.v1 import PathNotExistsError


def read_data(path="../s11_bboxes", images_path="../s11_images"):
    data_dicts = []

    # Ensure paths exist
    if not os.path.exists(path):
        raise PathNotExistsError(path=path)
    if not os.path.exists(images_path):
        raise PathNotExistsError(path=images_path)

    print("Reading data ...")

    for filename in os.listdir(path):

        # Ensure .json type
        if not filename.endswith(".json"):
            print(f"Unknown file format found: {filename}")
            continue

        # Open and read JSON file
        with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
            data = json.load(file)

            img = cv2.imread(os.path.join(images_path, f"bboxes_img_{data['image']}.png"))

            if img is None:
                print(f"No image found in {filename}")
                break

            h, w = img.shape[:2]

            # Verify all bboxes before being stored
            for bbox in data['wallies']:
                if bbox[0] >= bbox[2]:
                    raise ValueError(f"Max x wrong in {filename}")
                if bbox[1] >= bbox[3]:
                    raise ValueError(f"Max y wrong in {filename}")
                if len(bbox) > 4:
                    raise ValueError(f"Too many values in {filename}")
                if bbox[0] > w or bbox[1] > h or bbox[2] > w or bbox[3] > h:
                    raise ValueError(f"bbox out of bounds in {filename}")

            # Add dict with image and all bboxes, now formatted correctly
            data_dicts.append({'image': img, 'bboxes': np.array(data['wallies'], dtype=np.float32)})

    print('Data acquired')

    # Return all data
    return data_dicts

if __name__ == '__main__':
    read_data(path="../s11_bboxes")