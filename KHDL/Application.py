# Importing Libraries

import os
import json
import operator
from pathlib import Path
from string import ascii_uppercase

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk

from spellchecker import SpellChecker

# Giảm log TensorFlow
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    InputLayer,
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)


# =====================================================
# PATH
# =====================================================

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "Models"


# =====================================================
# BUILD MODEL TỪ FILE JSON CŨ
# Không dùng model_from_json vì Keras mới dễ lỗi Python 3.13
# =====================================================

def build_model_from_old_json(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        model_config = json.load(f)

    model_name = model_config.get("config", {}).get("name", "sequential")
    layers_config = model_config["config"]["layers"]

    model = Sequential(name=model_name)

    input_added = False

    for layer_info in layers_config:
        class_name = layer_info["class_name"]
        config = layer_info["config"]

        if class_name == "InputLayer":
            batch_shape = config.get("batch_input_shape") or config.get("batch_shape")

            if batch_shape is not None:
                input_shape = tuple(batch_shape[1:])
            else:
                input_shape = tuple(config.get("shape"))

            if not input_added:
                model.add(InputLayer(
                    shape=input_shape,
                    name=config.get("name"),
                    dtype=config.get("dtype", "float32")
                ))
                input_added = True

        elif class_name == "Conv2D":
            # Nếu JSON không có InputLayer riêng thì lấy input_shape từ Conv2D đầu tiên
            if not input_added and "batch_input_shape" in config:
                input_shape = tuple(config["batch_input_shape"][1:])
                model.add(InputLayer(shape=input_shape))
                input_added = True

            model.add(Conv2D(
                filters=config["filters"],
                kernel_size=tuple(config["kernel_size"]),
                strides=tuple(config.get("strides", (1, 1))),
                padding=config.get("padding", "valid"),
                activation=config.get("activation", None),
                use_bias=config.get("use_bias", True),
                dilation_rate=tuple(config.get("dilation_rate", (1, 1))),
                name=config.get("name")
            ))

        elif class_name == "MaxPooling2D":
            pool_size = tuple(config.get("pool_size", (2, 2)))
            strides = config.get("strides")

            if strides is not None:
                strides = tuple(strides)
            else:
                strides = pool_size

            model.add(MaxPooling2D(
                pool_size=pool_size,
                strides=strides,
                padding=config.get("padding", "valid"),
                name=config.get("name")
            ))

        elif class_name == "Flatten":
            model.add(Flatten(
                name=config.get("name")
            ))

        elif class_name == "Dense":
            model.add(Dense(
                units=config["units"],
                activation=config.get("activation", None),
                use_bias=config.get("use_bias", True),
                name=config.get("name")
            ))

        elif class_name == "Dropout":
            model.add(Dropout(
                rate=config["rate"],
                name=config.get("name")
            ))

        else:
            raise ValueError(f"Layer chưa hỗ trợ: {class_name}")

    return model

def load_old_model(json_file, h5_file, model_name):
    json_path = MODELS_DIR / json_file
    h5_path = MODELS_DIR / h5_file

    model = build_model_from_old_json(json_path)
    model.load_weights(str(h5_path))

    print(f"Loaded {model_name}")

    return model


# =====================================================
# APPLICATION
# =====================================================

class Application:

    def __init__(self):
        # Suggestions mới, thay cho Hunspell
        self.spell = SpellChecker(language="en")

        # Camera
        self.vs = cv2.VideoCapture(0)

        if not self.vs.isOpened():
            print("Không mở được camera 0. Đang thử camera 1...")
            self.vs = cv2.VideoCapture(1)

        if not self.vs.isOpened():
            print("Không mở được camera. Kiểm tra quyền camera Windows.")

        self.current_image = None
        self.current_image2 = None

        # =====================================================
        # LOAD MODELS
        # =====================================================

        print("Loading models...")

        self.loaded_model = load_old_model(
            "model_new.json",
            "model_new.h5",
            "Main model"
        )

        self.loaded_model_dru = load_old_model(
            "model-bw_dru.json",
            "model-bw_dru.h5",
            "DRU model"
        )

        self.loaded_model_tkdi = load_old_model(
            "model-bw_tkdi.json",
            "model-bw_tkdi.h5",
            "TKDI model"
        )

        self.loaded_model_smn = load_old_model(
            "model-bw_smn.json",
            "model-bw_smn.h5",
            "SMN model"
        )

        print("Loaded model from disk")

        # =====================================================
        # COUNTER
        # =====================================================

        self.ct = {}
        self.ct["blank"] = 0
        self.blank_flag = 0

        for i in ascii_uppercase:
            self.ct[i] = 0

        # =====================================================
        # GUI
        # =====================================================

        self.root = tk.Tk()
        self.root.title("Sign Language To Text Conversion")
        self.root.protocol("WM_DELETE_WINDOW", self.destructor)
        self.root.geometry("1000x900")

        self.panel = tk.Label(self.root)
        self.panel.place(x=100, y=10, width=580, height=580)

        self.panel2 = tk.Label(self.root)
        self.panel2.place(x=400, y=65, width=275, height=275)

        self.T = tk.Label(self.root)
        self.T.place(x=60, y=5)
        self.T.config(
            text="Sign Language To Text Conversion",
            font=("Courier", 30, "bold")
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

        self.str = ""
        self.word = ""
        self.current_symbol = "Empty"
        self.photo = "Empty"

        self.video_loop()

    # =====================================================
    # GET SUGGESTIONS
    # =====================================================

    def get_suggestions(self):
        word = self.word.strip().lower()

        if len(word) == 0:
            return []

        try:
            candidates = self.spell.candidates(word)

            if candidates is None:
                return []

            candidates = list(candidates)

            # Ưu tiên từ gần độ dài với từ hiện tại
            candidates = sorted(
                candidates,
                key=lambda x: (abs(len(x) - len(word)), x)
            )

            candidates = [w.upper() for w in candidates]

            return candidates[:5]

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

            # Vùng nhận diện tay ROI
            x1 = int(0.5 * frame_w)
            y1 = 10
            x2 = frame_w - 10
            y2 = min(y1 + 300, frame_h - 10)

            cv2.rectangle(
                frame,
                (x1 - 1, y1 - 1),
                (x2 + 1, y2 + 1),
                (255, 0, 0),
                2
            )

            # Hiển thị ảnh camera
            display_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(display_image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)

            self.panel.imgtk = imgtk
            self.panel.config(image=imgtk)

            # Cắt ROI
            roi = frame[y1:y2, x1:x2]

            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

            blur = cv2.GaussianBlur(gray, (5, 5), 2)

            th3 = cv2.adaptiveThreshold(
                blur,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                11,
                2
            )

            _, res = cv2.threshold(
                th3,
                70,
                255,
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

            # Suggestions
            predicts = self.get_suggestions()

            if len(predicts) > 0:
                self.bt1.config(text=predicts[0], font=("Courier", 18))
            else:
                self.bt1.config(text="")

            if len(predicts) > 1:
                self.bt2.config(text=predicts[1], font=("Courier", 18))
            else:
                self.bt2.config(text="")

            if len(predicts) > 2:
                self.bt3.config(text=predicts[2], font=("Courier", 18))
            else:
                self.bt3.config(text="")

            if len(predicts) > 3:
                self.bt4.config(text=predicts[3], font=("Courier", 18))
            else:
                self.bt4.config(text="")

            if len(predicts) > 4:
                self.bt5.config(text=predicts[4], font=("Courier", 18))
            else:
                self.bt5.config(text="")

        self.root.after(5, self.video_loop)

    # =====================================================
    # PREDICT
    # =====================================================

    def predict(self, test_image):
        test_image = cv2.resize(test_image, (128, 128))
        test_image = test_image.astype("float32")
        test_image = test_image.reshape(1, 128, 128, 1)

        result = self.loaded_model.predict(test_image, verbose=0)

        result_dru = self.loaded_model_dru.predict(test_image, verbose=0)
        result_tkdi = self.loaded_model_tkdi.predict(test_image, verbose=0)
        result_smn = self.loaded_model_smn.predict(test_image, verbose=0)

        prediction = {}
        prediction["blank"] = result[0][0]

        inde = 1

        for i in ascii_uppercase:
            prediction[i] = result[0][inde]
            inde += 1

        # LAYER 1
        prediction = sorted(
            prediction.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        self.current_symbol = prediction[0][0]

        # LAYER 2: D, R, U
        if self.current_symbol in ["D", "R", "U"]:
            prediction_dru = {}
            prediction_dru["D"] = result_dru[0][0]
            prediction_dru["R"] = result_dru[0][1]
            prediction_dru["U"] = result_dru[0][2]

            prediction_dru = sorted(
                prediction_dru.items(),
                key=operator.itemgetter(1),
                reverse=True
            )

            self.current_symbol = prediction_dru[0][0]

        # LAYER 2: D, I, K, T
        if self.current_symbol in ["D", "I", "K", "T"]:
            prediction_tkdi = {}
            prediction_tkdi["D"] = result_tkdi[0][0]
            prediction_tkdi["I"] = result_tkdi[0][1]
            prediction_tkdi["K"] = result_tkdi[0][2]
            prediction_tkdi["T"] = result_tkdi[0][3]

            prediction_tkdi = sorted(
                prediction_tkdi.items(),
                key=operator.itemgetter(1),
                reverse=True
            )

            self.current_symbol = prediction_tkdi[0][0]

        # LAYER 2: M, N, S
        if self.current_symbol in ["M", "N", "S"]:
            prediction_smn = {}
            prediction_smn["M"] = result_smn[0][0]
            prediction_smn["N"] = result_smn[0][1]
            prediction_smn["S"] = result_smn[0][2]

            prediction_smn = sorted(
                prediction_smn.items(),
                key=operator.itemgetter(1),
                reverse=True
            )

            self.current_symbol = prediction_smn[0][0]

        # Nếu blank thì reset bộ đếm chữ cái
        if self.current_symbol == "blank":
            for i in ascii_uppercase:
                self.ct[i] = 0

        self.ct[self.current_symbol] += 1

        if self.ct[self.current_symbol] > 60:

            for i in ascii_uppercase:
                if i == self.current_symbol:
                    continue

                tmp = abs(self.ct[self.current_symbol] - self.ct[i])

                if tmp <= 20:
                    self.ct["blank"] = 0

                    for j in ascii_uppercase:
                        self.ct[j] = 0

                    return

            self.ct["blank"] = 0

            for i in ascii_uppercase:
                self.ct[i] = 0

            if self.current_symbol == "blank":

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
                self.word += self.current_symbol

    # =====================================================
    # ACTION SUGGESTION
    # =====================================================

    def use_suggestion(self, index):
        predicts = self.get_suggestions()

        if len(predicts) > index:
            selected_word = predicts[index]

            self.word = ""

            if len(self.str) > 0:
                self.str += " "

            self.str += selected_word

    def action1(self):
        self.use_suggestion(0)

    def action2(self):
        self.use_suggestion(1)

    def action3(self):
        self.use_suggestion(2)

    def action4(self):
        self.use_suggestion(3)

    def action5(self):
        self.use_suggestion(4)

    # =====================================================
    # CLOSE APP
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
# RUN APP
# =====================================================

print("Starting Application...")

app = Application()
app.root.mainloop()