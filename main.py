from tkinter import *
from tkinter import filedialog, messagebox
import cv2
from PIL import Image, ImageTk
import numpy as np
import os
import threading

from image_processing_functions import ImageProcessor

DISPLAY_WIDTH = 400
DISPLAY_HEIGHT = 300

class waldoGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Where is Waldo?")
        self.root.geometry("800x900")
        self.root.protocol("WM_DELETE_WINDOW", self._on_closing)

        self.current_image = None
        self.current_image_path = None
        self.points = []
        self.selecting_point = False
        self.current_point = None

        self.processor = ImageProcessor(
            on_status=self._set_status,
            on_new_image=self._on_new_image,
        )

        self._build_ui()
        self.root.after(100, self._auto_load_model)

    def _build_ui(self):
        self.label_widget = Label(self.root, bg="white")
        self.label_widget.pack(pady=10)
        self.label_widget.bind("<Button-1>", self._start_selection)
        self.label_widget.bind("<B1-Motion>", self._update_selection)
        self.label_widget.bind("<ButtonRelease-1>", self._end_selection)

        control_frame = Frame(self.root)
        control_frame.pack(pady=5)

        monitor_frame = Frame(control_frame)
        monitor_frame.pack(pady=5)

        self.start_btn = Button(
            monitor_frame, text="Start Monitoring", command=self._start_monitoring,
            bg="lightgreen", width=15, font=("Arial", 10),
        )
        self.start_btn.pack(side=LEFT, padx=5)

        self.stop_btn = Button(
            monitor_frame, text="Stop Monitoring", command=self._stop_monitoring,
            bg="lightcoral", width=15, font=("Arial", 10), state=DISABLED,
        )
        self.stop_btn.pack(side=LEFT, padx=5)

        process_frame = Frame(control_frame)
        process_frame.pack(pady=5)

        Button(process_frame, text="Process Image", command=self._process_current_image,
               bg="lightblue", width=15, font=("Arial", 10)).pack(side=LEFT, padx=5)
        Button(process_frame, text="Reset Points", command=self._reset_selection,
               bg="yellow", width=15, font=("Arial", 10)).pack(side=LEFT, padx=5)
        Button(process_frame, text="Rotate 90 degrees\n CW",
               command=lambda: self._rotate_image_90("cw"),
               bg="lightpink", width=15, font=("Arial", 10)).pack(side=LEFT, padx=5)
        Button(process_frame, text="Rotate 90 degrees\n CCW",
               command=lambda: self._rotate_image_90("ccw"),
               bg="lightpink", width=15, font=("Arial", 10)).pack(side=LEFT, padx=5)

        model_frame = Frame(control_frame)
        model_frame.pack(pady=5)

        Button(model_frame, text="Load Model", command=self._load_model_manually,
               bg="lightcyan", width=15, font=("Arial", 10)).pack(side=LEFT, padx=5)

        instructions = (
            "\nInstructions:\n"
            "1. Load a Keras model using \"Load Model\" button\n"
            "2. Click \"Start Monitoring\" and select a folder to watch\n"
            "3. Add NEW images to that folder (existing images are ignored)\n"
            "4. When a new image appears, click 4 points in clockwise order\n"
            "5. Click \"Process Image\" to apply perspective correction\n"
            "6. After correction, you can process with the model to find objects\n"
            "7. Enable \"Auto-Process\" to automatically run model after correction\n"
            "8. Corrected images are saved in the \"processed_images\" folder\n"
        )
        Label(self.root, text=instructions, justify=LEFT, fg="gray",
              font=("Arial", 9), bg="white", relief=SUNKEN, padx=10, pady=10
              ).pack(pady=10, fill=BOTH, padx=20)

        status_frame = Frame(self.root)
        status_frame.pack(fill=X, padx=10, pady=5)

        self.status_label = Label(
            status_frame, text="Ready - Load model to enable detection",
            fg="blue", font=("Arial", 9),
        )
        self.status_label.pack()

    def _set_status(self, text, color="black"):
        self.status_label.config(text=text, fg=color)

    def _auto_load_model(self):
        try:
            self.processor.load_model()
        except RuntimeError as e:
            messagebox.showerror("Model Error", str(e))

    def _load_model_manually(self):
        file_path = filedialog.askopenfilename(
            title="Select Keras Model",
            filetypes=[("Keras Model", "*.keras"), ("H5 Model", "*.h5"), ("All Files", "*.*")],
        )
        if file_path:
            try:
                self.processor.load_model(path=file_path)
                messagebox.showinfo("Success", f"Model loaded from:\n{file_path}")
            except RuntimeError as e:
                messagebox.showerror("Model Error", str(e))

    def _start_monitoring(self):
        folder = filedialog.askdirectory(title="Select Folder to Monitor for New Images")
        if not folder:
            return

        started = self.processor.start_monitoring(folder)
        if not started:
            messagebox.showinfo("Already Running", "Folder monitoring is already active")
            return

        self.current_image = None
        self._reset_selection()

        self._set_status(f"Monitoring: {folder}", "green")
        self.start_btn.config(state=DISABLED)
        self.stop_btn.config(state=NORMAL)

        messagebox.showinfo(
            "Monitoring Started",
            f"Monitoring folder:\n{folder}\n\nnew stuff will be processed.\n"
            "When a new image appears you'll be able to select 4 points and correct its perspective.",
        )

    def _stop_monitoring(self):
        self.processor.stop_monitoring()
        self._set_status("Monitoring stopped", "red")
        self.start_btn.config(state=NORMAL)
        self.stop_btn.config(state=DISABLED)
        print("Folder monitoring stopped")

    def _on_new_image(self, image, path):
        self.current_image = image
        self.current_image_path = path
        self.root.after(0, self._reset_selection)
        self.root.after(0, lambda: self._set_status(
            f"New image: {os.path.basename(path)} - Select 4 points", "green"
        ))
        self.root.after(0, self._display_image_with_overlay)

    def _display_image_with_overlay(self):
        if self.current_image is None:
            return

        img = self.processor.apply_rotation(self.current_image.copy())
        display_frame = cv2.resize(img, (DISPLAY_WIDTH, DISPLAY_HEIGHT))

        if self.points:
            display_points = [
                self.processor.original_to_display_coords(
                    ox, oy, self.current_image, DISPLAY_WIDTH, DISPLAY_HEIGHT
                )
                for (ox, oy) in self.points
            ]

            for i, (dx, dy) in enumerate(display_points):
                cv2.circle(display_frame, (dx, dy), 5, (0, 255, 0), -1)
                cv2.putText(display_frame, str(i + 1), (dx + 5, dy - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                if i < len(display_points) - 1:
                    cv2.line(display_frame, display_points[i], display_points[i + 1], (0, 255, 0), 2)

            if len(display_points) == 4:
                cv2.line(display_frame, display_points[3], display_points[0], (0, 255, 0), 2)
                overlay = display_frame.copy()
                pts = np.array(display_points, np.int32).reshape((-1, 1, 2))
                cv2.fillPoly(overlay, [pts], (0, 255, 0))
                cv2.addWeighted(overlay, 0.3, display_frame, 0.7, 0, display_frame)

        if self.selecting_point and self.current_point:
            cv2.circle(display_frame, self.current_point, 5, (255, 0, 0), -1)

        opencv_image = cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA)
        photo_image = ImageTk.PhotoImage(image=Image.fromarray(opencv_image))
        self.label_widget.photo_image = photo_image
        self.label_widget.configure(image=photo_image)

        if self.processor.folder_monitoring:
            self.label_widget.after(50, self._display_image_with_overlay)

    def _start_selection(self, event):
        if len(self.points) < 4 and self.current_image is not None:
            self.selecting_point = True
            self.current_point = (event.x, event.y)

    def _update_selection(self, event):
        if self.selecting_point:
            self.current_point = (event.x, event.y)

    def _end_selection(self, event):
        if not self.selecting_point or self.current_image is None or len(self.points) >= 4:
            return

        orig_x, orig_y = self.processor.display_to_original_coords(
            event.x, event.y, self.current_image, DISPLAY_WIDTH, DISPLAY_HEIGHT
        )

        self.points.append((orig_x, orig_y))
        self._set_status(f"Point {len(self.points)} added", "green")

        self.selecting_point = False
        self.current_point = None

        if len(self.points) == 4:
            self._set_status("Box complete - click 'Process Image'", "orange")

        self._display_image_with_overlay()

    def _reset_selection(self):
        self.points = []
        self._set_status("Selection reset", "blue")
        print("Selection reset")

    def _rotate_image_90(self, state):
        self.processor.set_rotation(state)
        print(f"cw: ", {self.processor.rotate_cw}, " ccw: ", {self.processor.rotate_ccw})

    def _process_current_image(self):
        if len(self.points) != 4:
            messagebox.showwarning(
                "Incomplete",
                f"Please select exactly 4 points, only {len(self.points)} points",
            )
            return

        if self.current_image is None:
            messagebox.showerror("Error", "No image to process")
            return

        corrected = self.processor.perspective_correction(
            self.current_image, np.array(self.points, dtype="float32")
        )

        if corrected is None:
            messagebox.showerror("Error", "Failed to process image")
            return

        output_path = self.processor.save_corrected_image(corrected, self.current_image_path)
        self._show_corrected_result(corrected, output_path)
        self._reset_selection()

    def _show_corrected_result(self, corrected, output_path):
        display_image = cv2.cvtColor(corrected, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(display_image)
        pil_image.thumbnail((600, 600), Image.Resampling.LANCZOS)

        result_window = Toplevel(self.root)
        result_window.title("Corrected Image")

        img_tk = ImageTk.PhotoImage(pil_image)
        Label(result_window, image=img_tk).pack(padx=10, pady=10)
        img_tk.image = img_tk

        Label(
            result_window,
            text=f"Saved: {os.path.basename(output_path)}\nResolution: {corrected.shape[1]}x{corrected.shape[0]}",
            font=("Arial", 9),
        ).pack(pady=5)

        btn_frame = Frame(result_window)
        btn_frame.pack(pady=10)

        def process_and_show():
            result_window.destroy()
            threading.Thread(
                target=lambda: self.processor.process_with_model(
                    corrected,
                    show_results_callback=lambda *args: self.root.after(
                        0, lambda: self._show_model_results(*args)
                    ),
                ),
                daemon=True,
            ).start()

        Button(btn_frame, text="Process with Model", command=process_and_show,
               bg="lightblue", width=15).pack(side=LEFT, padx=5)
        Button(btn_frame, text="Close", command=result_window.destroy,
               width=15).pack(side=LEFT, padx=5)

        messagebox.showinfo("Success", f"Image saved to:\n{output_path}")

    def _show_model_results(self, original_image, best_crop, max_prob, probabilities, crops):
        result_window = Toplevel(self.root)
        result_window.title("Model Detection Results")
        result_window.geometry("1000x600")

        main_frame = Frame(result_window)
        main_frame.pack(fill=BOTH, expand=True, padx=10, pady=10)

        left_frame = Frame(main_frame)
        left_frame.pack(side=LEFT, fill=BOTH, expand=True)

        Label(left_frame, text=f"Best Match (Probability: {max_prob:.4f})",
              font=("Arial", 12, "bold")).pack(pady=5)

        if best_crop is not None:
            best_rgb = cv2.cvtColor(best_crop, cv2.COLOR_BGR2RGB)
            best_pil = Image.fromarray(best_rgb)
            best_pil.thumbnail((300, 300), Image.Resampling.LANCZOS)
            best_tk = ImageTk.PhotoImage(best_pil)
            Label(left_frame, image=best_tk).pack()
            best_tk.image = best_tk

            def _save():
                path = self.processor.save_best_crop(best_crop)
                messagebox.showinfo("Saved", f"Best crop saved to:\n{path}")

            Button(left_frame, text="Save Best Crop", command=_save,
                   bg="lightgreen", width=15).pack(pady=5)

        right_frame = Frame(main_frame)
        right_frame.pack(side=RIGHT, fill=BOTH, expand=True)

        Label(right_frame, text="Detection Location",
              font=("Arial", 12, "bold")).pack(pady=5)

        if len(probabilities) > 0:
            max_index = int(np.argmax(probabilities))
            display_img = original_image.copy()
            best_x, best_y = crops[max_index]
            cv2.rectangle(display_img,
                          (best_x, best_y), (best_x + 128, best_y + 128),
                          (0, 255, 0), 10)
            display_rgb = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
            display_pil = Image.fromarray(display_rgb)
            display_pil.thumbnail((800, 800), Image.Resampling.LANCZOS)
            display_tk = ImageTk.PhotoImage(display_pil)
            Label(right_frame, image=display_tk).pack()
            display_tk.image = display_tk

        bottom_frame = Frame(result_window)
        bottom_frame.pack(fill=X, padx=10, pady=10)

        probs_list = list(probabilities)
        crops_list = list(crops)

        def find_next_best():
            if len(probs_list) <= 1:
                return
            cur_max = int(np.argmax(probs_list))
            del probs_list[cur_max]
            del crops_list[cur_max]
            new_max = int(np.argmax(probs_list))
            new_prob = probs_list[new_max]
            nx, ny = crops_list[new_max]
            new_crop = original_image[ny:ny + 128, nx:nx + 128]
            result_window.destroy()
            self._show_model_results(original_image, new_crop, new_prob, probs_list, crops_list)

        Button(bottom_frame, text="Next Best Match", command=find_next_best,
               bg="lightblue", width=15).pack(side=LEFT, padx=5)
        Button(bottom_frame, text="Close", command=result_window.destroy,
               width=15).pack(side=RIGHT, padx=5)

    def _on_closing(self):
        self.processor.stop_monitoring()
        self.root.destroy()


if __name__ == "__main__":
    root = Tk()
    app = waldoGUI(root)
    root.mainloop()
