import cv2
import numpy as np
import os
import time
import threading
from datetime import datetime
from pathlib import Path
from math import ceil
import tensorflow as tf


TILE_SIZE = 128
DEFAULT_MODEL_PATH = "Keras Models/prototype11k.keras"
OUTPUT_FOLDER = "processed_images"


def top1_soft_2(y_true, y_pred, tol=0.05):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    pred_idx = tf.argmax(y_pred)
    pred_true = tf.gather(y_true, pred_idx)
    max_true = tf.reduce_max(y_true)
    return tf.cast(pred_true >= (max_true - tol), tf.float32)


def combined_loss(y_true, y_pred):
    return tf.keras.losses.Huber(delta=0.1)(y_true, y_pred)


class ImageProcessor:
    def __init__(self, on_status=None, on_new_image=None):
        self.model = None
        self.model_loaded = False
        self.model_path = DEFAULT_MODEL_PATH

        self.scale_to = 32
        self.rotate_cw = False
        self.rotate_ccw = False

        self.watch_folder = ""
        self.output_folder = OUTPUT_FOLDER
        self.processed_files = set()
        self.folder_monitoring = False
        self.monitor_thread = None

        self._on_status = on_status or (lambda text, color: None)
        self._on_new_image = on_new_image or (lambda image, path: None)

    def _status(self, text, color="black"):
        self._on_status(text, color)

    def load_model(self, path=None):
        if path:
            self.model_path = path
        try:
            if not os.path.exists(self.model_path):
                self._status(f"Model not found: {self.model_path}", "orange")
                return False

            self._status("Loading model...", "blue")
            self.model = tf.keras.models.load_model(
                self.model_path,
                custom_objects={
                    "top1_soft_2": top1_soft_2,
                    "combined_loss": combined_loss,
                },
            )
            self.model_loaded = True
            self._status("Model loaded successfully", "green")
            return True
        except Exception as e:
            self._status("Model loading failed", "red")
            raise RuntimeError(f"Failed to load model: {e}") from e

    def set_rotation(self, state):
        if state == "cw":
            self.rotate_cw = not self.rotate_cw
            if self.rotate_cw:
                self.rotate_ccw = False
        elif state == "ccw":
            self.rotate_ccw = not self.rotate_ccw
            if self.rotate_ccw:
                self.rotate_cw = False

    def apply_rotation(self, img):
        if self.rotate_cw:
            return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        if self.rotate_ccw:
            return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return img

    def display_to_original_coords(self, disp_x, disp_y, original_image, display_width, display_height):
        img = self.apply_rotation(original_image)
        rot_h, rot_w = img.shape[:2]

        rot_x = max(0.0, min(disp_x * rot_w / display_width, rot_w - 1))
        rot_y = max(0.0, min(disp_y * rot_h / display_height, rot_h - 1))

        if self.rotate_cw:
            orig_x, orig_y = rot_y, rot_h - 1 - rot_x
        elif self.rotate_ccw:
            orig_x, orig_y = rot_w - 1 - rot_y, rot_x
        else:
            orig_x, orig_y = rot_x, rot_y

        orig_x = max(0, min(int(orig_x), original_image.shape[1] - 1))
        orig_y = max(0, min(int(orig_y), original_image.shape[0] - 1))
        return orig_x, orig_y

    def original_to_display_coords(self, orig_x, orig_y, original_image, display_width, display_height):
        img = self.apply_rotation(original_image)
        rot_h, rot_w = img.shape[:2]

        if self.rotate_cw:
            rot_x, rot_y = orig_y, rot_h - 1 - orig_x
        elif self.rotate_ccw:
            rot_x, rot_y = rot_w - 1 - orig_y, orig_x
        else:
            rot_x, rot_y = orig_x, orig_y

        return int(rot_x * (display_width / rot_w)), int(rot_y * (display_height / rot_h))

    def perspective_correction(self, image, pts):
        def order_points(pts):
            rect = np.zeros((4, 2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            return rect

        rect = order_points(pts)
        (tl, tr, br, bl) = rect

        widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
        widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
        maxWidth = max(int(widthA), int(widthB))

        heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
        heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
        maxHeight = max(int(heightA), int(heightB))

        dst = np.array(
            [[0, 0], [maxWidth - 1, 0], [maxWidth - 1, maxHeight - 1], [0, maxHeight - 1]],
            dtype="float32",
        )
        M = cv2.getPerspectiveTransform(rect, dst)
        img = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
        img = self.apply_rotation(img)

        cv2.namedWindow("BoundAverageCharacter", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("BoundAverageCharacter", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        height, width, _ = img.shape
        state = {"bbox": [0, 0, 0, 0], "drawing": 0}

        def draw_rectangle_with_drag(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                state["bbox"] = [x, y, x, y]
                state["drawing"] = 1
            elif event == cv2.EVENT_MOUSEMOVE and state["drawing"] == 1:
                state["bbox"][2:] = x, y
            elif event == cv2.EVENT_LBUTTONUP:
                state["drawing"] = 2 if state["bbox"][0] != state["bbox"][2] else 0

        cv2.setMouseCallback("BoundAverageCharacter", draw_rectangle_with_drag)

        while state["drawing"] < 2:
            display = img.copy()
            if state["drawing"] > 0:
                cv2.rectangle(display, state["bbox"][:2], state["bbox"][2:], (0, 255, 0), 2)
            cv2.imshow("BoundAverageCharacter", display)
            cv2.waitKey(1)

        cv2.destroyAllWindows()

        wally_size = abs(state["bbox"][2] - state["bbox"][0]) / 1.3
        scaler = self.scale_to / wally_size
        print(f"scaler {scaler} with wally_size {wally_size} and person {state['bbox'][2] - state['bbox'][0]}")
        return cv2.resize(img, (ceil(width * scaler), ceil(height * scaler)))

    def process_with_model(self, image, show_results_callback=None):
        if not self.model_loaded:
            if not self.load_model():
                return None, None, None

        H, W = image.shape[:2]
        patches, coords = [], []

        for y in range(0, H - 127, 64):
            for x in range(0, W - 127, 64):
                patches.append(image[y:y + 128, x:x + 128])
                coords.append((x, y))

        patches = np.array(patches, dtype=np.float32)
        scores = self.model.predict(patches, verbose=0).squeeze()

        if scores is None or len(scores) == 0:
            return None, None, None

        order = np.argsort(scores)[::-1]
        ranked_scores = scores[order]
        ranked_coords = [coords[i] for i in order]

        max_index = np.argmax(ranked_scores)
        max_prob = ranked_scores[max_index]
        best_x, best_y = ranked_coords[max_index]
        best_crop = image[best_y:best_y + 128, best_x:best_x + 128]

        if show_results_callback:
            show_results_callback(image, best_crop, max_prob, ranked_scores, ranked_coords)

        return best_crop, max_prob, (ranked_scores, ranked_coords)

    def save_corrected_image(self, corrected, original_path):
        os.makedirs(self.output_folder, exist_ok=True)
        original_name = Path(original_path).stem
        output_path = os.path.join(self.output_folder, f"{original_name}_corrected.png")
        cv2.imwrite(output_path, corrected)
        return output_path

    def save_best_crop(self, crop_image):
        os.makedirs(self.output_folder, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(self.output_folder, f"best_match_{timestamp}.png")
        cv2.imwrite(output_path, crop_image)
        return output_path

    def start_monitoring(self, folder):
        if self.folder_monitoring:
            return False

        self.watch_folder = folder
        self.folder_monitoring = True
        self.processed_files.clear()

        self.monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self.monitor_thread.start()
        return True

    def stop_monitoring(self):
        self.folder_monitoring = False

    def _monitor_loop(self):
        image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}

        while self.folder_monitoring:
            try:
                new_images = [
                    (os.path.join(self.watch_folder, f),
                     os.path.getmtime(os.path.join(self.watch_folder, f)))
                    for f in os.listdir(self.watch_folder)
                    if (os.path.isfile(os.path.join(self.watch_folder, f))
                        and Path(f).suffix.lower() in image_extensions
                        and os.path.join(self.watch_folder, f) not in self.processed_files)
                ]

                if new_images:
                    new_images.sort(key=lambda x: x[1], reverse=True)
                    latest_path = new_images[0][0]

                    time.sleep(0.5)
                    image = cv2.imread(latest_path)

                    if image is not None:
                        self.processed_files.add(latest_path)
                        self._on_new_image(image, latest_path)

                time.sleep(1)

            except Exception as e:
                print(f"Error: {e}")
                time.sleep(5)
