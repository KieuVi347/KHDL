# =====================================================
# APP - SIGN LANGUAGE TO TEXT (MobileNetV2)
# Thay thế ensemble 4 models cũ bằng MobileNetV2
# =====================================================

import os
import json
from pathlib import Path

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from spellchecker import SpellChecker

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf


# =====================================================
# PATH
# =====================================================

BASE_DIR   = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"

MODEL_PATH        = MODELS_DIR / "mobilenet_model.h5"
CLASS_INDICES_PATH = MODELS_DIR / "class_indices.json"


# =====================================================
# LOAD MODEL & CLASS INDICES
# =====================================================

def load_model_and_classes():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model tại {MODEL_PATH}\n"
            "Hãy chạy train_mobilenet.py trước!"
        )

    print("Đang load MobileNetV2 model...")
    model = tf.keras.models.load_model(str(MODEL_PATH))
    print("Load model thành công!")

    with open(str(CLASS_INDICES_PATH), "r") as f:
        idx_to_class = json.load(f)

    # Key là string khi load từ JSON, chuyển về int
    idx_to_class = {int(k): v for k, v in idx_to_class.items()}

    return model, idx_to_class


# =====================================================
# APPLICATION
# =====================================================

class Application:

    def __init__(self):
        self.spell = SpellChecker(language="en")

        # Load model
        self.model, self.idx_to_class = load_model_and_classes()

        # Preprocess function giống lúc train
        self.preprocess = tf.keras.applications.mobilenet_v2.preprocess_input

        # Camera
        self.vs = cv2.VideoCapture(0)

        if not self.vs.isOpened():
            print("Không mở được camera 0. Đang thử camera 1...")
            self.vs = cv2.VideoCapture(1)

        if not self.vs.isOpened():
            print("Không mở được camera. Kiểm tra quyền camera Windows.")

        self.current_image  = None
        self.current_image2 = None

        # =====================================================
        # COUNTER (giữ logic gốc)
        # =====================================================

        self.ct = {}
        self.ct["blank"] = 0
        self.blank_flag = 0

        all_labels = list(self.idx_to_class.values())

        for label in all_labels:
            self.ct[label] = 0

        if "blank" not in self.ct:
            self.ct["blank"] = 0

        # =====================================================
        # GUI
        # =====================================================

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion (MobileNetV2)")
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)
        self.root.geometry("1000x900")

        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)

        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=400, y=65, width=275, height=275)

        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(
            text="Sign Language To Text (MobileNetV2)",
            font=("Courier", 24, "bold")
        )

        self.panel3 = tk.Label(self.root)
        self.panel3.place(x=500, y=540)

        self.T1 = tk.Label(self.root)
        self.T1.place(x=10, y=540)
        self.T1.config(text="Character :", font=("Courier", 30, "bold"))

        self.panel4 = tk.Label(self.root)
        self.panel4.place(x=220, y=595)

        self.T2 = tk.Label(self.root)
        self.T2.place(x=10, y=595)
        self.T2.config(text="Word :", font=("Courier", 30, "bold"))

        self.panel5 = tk.Label(self.root)
        self.panel5.place(x=350, y=645)

        self.T3 = tk.Label(self.root)
        self.T3.place(x=10, y=645)
        self.T3.config(text="Sentence :", font=("Courier", 30, "bold"))

        self.T4 = tk.Label(self.root)
        self.T4.place(x=250, y=690)
        self.T4.config(
            text="Suggestions :",
            fg="red",
            font=("Courier", 30, "bold")
        )

        # Label hiển thị confidence score
        self.T_conf = tk.Label(self.root)
        self.T_conf.place(x=10, y=500)
        self.T_conf.config(text="Confidence: -", font=("Courier", 16), fg="green")

        self.bt1 = tk.Button(self.root, command=self.action1, height=1, width=12)
        self.bt1.place(x=26, y=745)

        self.bt2 = tk.Button(self.root, command=self.action2, height=1, width=12)
        self.bt2.place(x=220, y=745)

        self.bt3 = tk.Button(self.root, command=self.action3, height=1, width=12)
        self.bt3.place(x=414, y=745)

        self.bt4 = tk.Button(self.root, command=self.action4, height=1, width=12)
        self.bt4.place(x=608, y=745)

        self.bt5 = tk.Button(self.root, command=self.action5, height=1, width=12)
        self.bt5.place(x=802, y=745)

        self.str            = ""
        self.word           = ""
        self.current_symbol = "Empty"
        self.current_conf   = 0.0
        self.photo          = "Empty"

        self.video_loop()

    # =====================================================
    # SUGGESTIONS
    # =====================================================

    def get_suggestions(self):
        word = self.word.strip().lower()

        if len(word) == 0:
            return []

        try:
            candidates = self.spell.candidates(word)

            if candidates is None:
                return []

            candidates = sorted(
                list(candidates),
                key=lambda x: (abs(len(x) - len(word)), x)
            )

            return [w.upper() for w in candidates[:5]]

        except Exception:
            return []

    # =====================================================
    # VIDEO LOOP
    # =====================================================

    def video_loop(self):
        ok, frame = self.vs.read()

        if ok:
            frame = cv2.flip(frame, 1)

            frame_h, frame_w = frame.shape[:2]

            x1 = int(0.5 * frame_w)
            y1 = 10
            x2 = frame_w - 10
            y2 = min(y1 + 300, frame_h - 10)

            cv2.rectangle(frame, (x1 - 1, y1 - 1), (x2 + 1, y2 + 1), (255, 0, 0), 2)

            # Hiển thị camera
            display_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(display_image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            # ROI -> threshold (giữ giống pipeline gốc)
            roi  = frame[y1:y2, x1:x2]
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5, 5), 2)

            th3 = cv2.adaptiveThreshold(
                blur, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11, 2
            )

            _, res = cv2.threshold(
                th3, 70, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )

            self.predict(res)

            # Hiển thị ảnh threshold
            self.current_image2 = Image.fromarray(res)
            imgtk2 = ImageTk.PhotoImage(image=self.current_image2)
            self.panel2.imgtk = imgtk2
            self.panel2.config(image=imgtk2)

            self.panel3.config(text=self.current_symbol, font=("Courier", 30))
            self.panel4.config(text=self.word, font=("Courier", 30))
            self.panel5.config(text=self.str, font=("Courier", 30))
            self.T_conf.config(
                text=f"Confidence: {self.current_conf * 100:.1f}%",
                fg="green" if self.current_conf > 0.7 else "orange"
            )

            # Suggestions
            predicts = self.get_suggestions()
            buttons  = [self.bt1, self.bt2, self.bt3, self.bt4, self.bt5]

            for i, btn in enumerate(buttons):
                btn.config(
                    text=predicts[i] if i < len(predicts) else "",
                    font=("Courier", 18)
                )

        self.root.after(5, self.video_loop)

    # =====================================================
    # PREDICT - dùng MobileNetV2
    # =====================================================

    def predict(self, test_image):
        # Resize về 128x128
        img = cv2.resize(test_image, (128, 128))

        # Grayscale -> RGB (lặp 3 kênh)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # Preprocess giống lúc train
        img_input = img_rgb.astype("float32")
        img_input = self.preprocess(img_input)
        img_input = img_input.reshape(1, 128, 128, 3)

        # Predict
        result = self.model.predict(img_input, verbose=0)[0]

        pred_idx  = int(np.argmax(result))
        pred_conf = float(result[pred_idx])

        pred_label = self.idx_to_class.get(pred_idx, "blank")

        self.current_conf = pred_conf

        # =====================================================
        # COUNTER LOGIC (giữ nguyên logic gốc)
        # =====================================================

        if pred_label == "blank" or pred_label == "0":
            for label in list(self.ct.keys()):
                if label != "blank":
                    self.ct[label] = 0

        self.current_symbol = pred_label
        self.ct[pred_label] = self.ct.get(pred_label, 0) + 1

        if self.ct[pred_label] > 60:
            # Kiểm tra xem có ký tự nào khác quá gần không
            for label, count in self.ct.items():
                if label == pred_label:
                    continue

                if abs(self.ct[pred_label] - count) <= 20:
                    self.ct = {k: 0 for k in self.ct}
                    return

            # Reset counter
            self.ct = {k: 0 for k in self.ct}

            if pred_label in ["blank", "0"]:
                if self.blank_flag == 0:
                    self.blank_flag = 1

                    if len(self.str) > 0:
                        self.str += " "

                    self.str += self.word
                    self.word = ""
            else:
                if len(self.str) > 16:
                    self.str = ""

                self.blank_flag = 0
                self.word += pred_label

    # =====================================================
    # SUGGESTIONS ACTIONS
    # =====================================================

    def use_suggestion(self, index):
        predicts = self.get_suggestions()

        if len(predicts) > index:
            self.word = ""

            if len(self.str) > 0:
                self.str += " "

            self.str += predicts[index]

    def action1(self): self.use_suggestion(0)
    def action2(self): self.use_suggestion(1)
    def action3(self): self.use_suggestion(2)
    def action4(self): self.use_suggestion(3)
    def action5(self): self.use_suggestion(4)

    # =====================================================
    # CLOSE
    # =====================================================

    def destructor(self):
        print("Closing Application...")

        try:
            self.root.destroy()
        except Exception:
            pass

        try:
            self.vs.release()
        except Exception:
            pass

        cv2.destroyAllWindows()


# =====================================================
# RUN
# =====================================================

print("Starting Application (MobileNetV2)...")
app = Application()
app.root.mainloop()
