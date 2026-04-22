import json
import os
from math import ceil
import cv2
import numpy as np
import math

from pydantic.v1 import PathNotExistsError


def format_data(path, out_bboxes="../s11_bboxes", out_img="../s11_images", scale_to=32):
    img_num = 0

    if not os.path.exists(path):
        raise PathNotExistsError(path=path)

    # Create folder for data storage if they don't exist
    if not os.path.exists(out_bboxes):
        os.makedirs(out_bboxes)
    if not os.path.exists(out_img):
        os.makedirs(out_img)

    # Check all files
    for filename in os.listdir(path):

        # Find all .json files per Where's Wally page
        if filename.endswith(".json"):
            img_num += 1
            image_path = ""
            types = (".png", ".jpeg", ".jpg")

            # Find the connected image of the .json file
            for ext in types:
                image_path = os.path.join(path, str(filename[:-5]) + ext)
                if os.path.isfile(image_path):
                    print(f"Found {image_path}. With json {filename}")
                    break

            # If no connected image could be found, disregard .json
            if image_path == "":
                print(f"File {filename} not found with extensions {types}")
                break

            # Read image data
            image = cv2.imread(os.path.join(path, image_path))
            if image is None:
                print(f"Could not open {filename} as image")
                break

            img_height, img_width = image.shape[:2]

            # Open and read JSON file
            with open(os.path.join(path, filename), 'r', encoding='utf-8') as file:
                # Read the .json
                data = json.load(file)

                # Collect all points from original saved format
                points = np.array([shape['points'] for shape in data['shapes']])

                # All widths
                widths = points[:, 1, 0] - points[:, 0, 0]

                # Size threshold filtering out the exceptionally large Wally's which aren't meant to be found
                threshold = widths.min() * 1.8

                # Create a mask for all Wally's that are smaller than the threshold
                mask = widths < threshold

                # Filter out the expections
                filtered = points[mask]
                filtered_widths = widths[mask]

                # Get Wally's average size in the scene
                wally_size = math.ceil(filtered_widths.mean())

                # Find the scaler to the set scale for Wally
                scaler = scale_to / wally_size

                # Resize image so Wally is at scale
                new_size = (ceil(img_width * scaler), ceil(img_height * scaler))
                resized = cv2.resize(image, new_size)

                wallies = []

                # Format the data into the wallies array
                for i, shape in enumerate(filtered):
                    # Use standard [4,] form
                    bbox = shape.flatten()

                    # Scale the points data
                    bbox_scaled = (bbox * scaler).tolist()

                    # Add scaled bbox to the maybe_wally dict in pascal_voc
                    wallies.append(bbox_scaled)

                # Save the resized images
                cv2.imwrite(os.path.join(out_img, f"bboxes_img_{img_num}.png"), resized)

                # Write all bbox data with image path to json file
                json_str = json.dumps({'image': img_num, 'wallies': wallies}, indent=4)
                with open(os.path.join(out_bboxes, f"bboxes_{img_num}.json"), "w") as f:
                    f.write(json_str)


# Run to save images
# Is not run automatically in Model
if __name__ == "__main__":
    format_data("../Images")



