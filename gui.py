import os
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.efficientnet import preprocess_input


CNN_MODEL_PATH = r"D:\PREDICTION OF DYSLEXIA\Backend\cnn_model.h5"
LSTM_MODEL_PATH = r"D:\PREDICTION OF DYSLEXIA\Backend\lstm_model.keras"
HYBRID_MODEL_PATH = r"D:\PREDICTION OF DYSLEXIA\Backend\hybrid.keras"

IMG_SIZE = (224, 224)

MAX_TIMESTEPS = 100
SEQ_FEATURE_COLS_LSTM = ["fix_x", "fix_y", "duration_ms", "saccade_len"]
SEQ_FEATURE_COLS_HYBRID = ["fix_x", "fix_y", "duration_ms"]

MILD_THRESHOLD = 0.4
SEVERE_THRESHOLD = 0.7


_cnn_model = None
_lstm_model = None
_hybrid_model = None


def get_cnn_model():
    global _cnn_model
    if _cnn_model is None:
        if not os.path.exists(CNN_MODEL_PATH):
            raise FileNotFoundError(f"CNN model not found: {CNN_MODEL_PATH}")
        _cnn_model = load_model(CNN_MODEL_PATH)
    return _cnn_model


def get_lstm_model():
    global _lstm_model
    if _lstm_model is None:
        if not os.path.exists(LSTM_MODEL_PATH):
            raise FileNotFoundError(f"LSTM model not found: {LSTM_MODEL_PATH}")
        _lstm_model = load_model(LSTM_MODEL_PATH)
    return _lstm_model


def get_hybrid_model():
    global _hybrid_model
    if _hybrid_model is None:
        if not os.path.exists(HYBRID_MODEL_PATH):
            raise FileNotFoundError(f"Hybrid model not found: {HYBRID_MODEL_PATH}")
        _hybrid_model = load_model(HYBRID_MODEL_PATH)
    return _hybrid_model



def preprocess_image_for_efficientnet(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_arr = img_to_array(img)
    img_arr = preprocess_input(img_arr)  # same as your hybrid code
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr


def preprocess_csv_for_lstm(csv_path):
    df = pd.read_csv(csv_path)

    base_cols = ["fix_x", "fix_y", "duration_ms"]
    for col in base_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV.")

    df = df[base_cols].dropna()
    if df.empty:
        raise ValueError("CSV file has no valid fixation rows after dropping NaNs.")

    if len(df) > 1:
        saccades = np.sqrt(np.diff(df["fix_x"]) ** 2 + np.diff(df["fix_y"]) ** 2)
        saccades = np.concatenate([[0], saccades])
    else:
        saccades = np.array([0] * len(df))

    df["saccade_len"] = saccades

    q99 = df["duration_ms"].quantile(0.99)
    df["duration_ms"] = np.clip(df["duration_ms"], 0, q99)

    df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

    arr = df[SEQ_FEATURE_COLS_LSTM].values.astype(np.float32)

    mean = arr.mean(axis=0, keepdims=True)
    std = arr.std(axis=0, keepdims=True)
    std[std == 0] = 1.0
    arr = (arr - mean) / std

    if arr.shape[0] > MAX_TIMESTEPS:
        arr = arr[:MAX_TIMESTEPS]
    else:
        pad_len = MAX_TIMESTEPS - arr.shape[0]
        pad_block = np.zeros((pad_len, arr.shape[1]), dtype=np.float32)
        arr = np.vstack([arr, pad_block])

    arr = np.expand_dims(arr, axis=0)
    return arr


def preprocess_csv_for_hybrid(csv_path):
    df = pd.read_csv(csv_path)

    for col in SEQ_FEATURE_COLS_HYBRID:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in CSV.")

    seq = df[SEQ_FEATURE_COLS_HYBRID].values.astype(np.float32)

    if seq.shape[0] > MAX_TIMESTEPS:
        seq = seq[:MAX_TIMESTEPS]
    else:
        pad_len = MAX_TIMESTEPS - seq.shape[0]
        pad_block = np.zeros((pad_len, seq.shape[1]), dtype=np.float32)
        seq = np.vstack([seq, pad_block])

    seq = np.expand_dims(seq, axis=0)
    return seq


def map_probability_to_stage(prob):
    if prob < MILD_THRESHOLD:
        return "Non-dyslexic", "Stage: No Stages"
    elif prob < SEVERE_THRESHOLD:
        return "Mild / Moderate dyslexia risk", "Stage: Mild / Early stage dyslexia"
    else:
        return "High dyslexia risk", "Stage: Severe / Advanced dyslexia"



class DyslexiaApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dyslexia Prediction - Eye Gaze (CNN / LSTM / Hybrid)")
        self.root.geometry("1050x650")
        self.root.configure(bg="#121212")
        self.root.resizable(False, False)

        self.image_path = None
        self.csv_path = None
        self.image_preview = None

        self._build_ui()

    def _build_ui(self):
        card = tk.Frame(self.root, bg="#1E1E1E", bd=0, highlightthickness=0)
        card.place(relx=0.5, rely=0.5, anchor="center", width=950, height=550)

        title = tk.Label(
            card,
            text="Dyslexia Prediction from Eye Gaze",
            fg="#FFFFFF",
            bg="#1E1E1E",
            font=("Segoe UI", 20, "bold")
        )
        title.pack(pady=(20, 5))

        subtitle = tk.Label(
            card,
            text="Upload fixation image (CNN), CSV gaze file (LSTM), or both (Hybrid CNN-LSTM)",
            fg="#BBBBBB",
            bg="#1E1E1E",
            font=("Segoe UI", 10)
        )
        subtitle.pack(pady=(0, 15))

        content_frame = tk.Frame(card, bg="#1E1E1E")
        content_frame.pack(fill="both", expand=True, padx=20, pady=10)

        left_frame = tk.Frame(content_frame, bg="#1E1E1E")
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 10))

        img_label_title = tk.Label(
            left_frame,
            text="Image Preview (for CNN / Hybrid)",
            fg="#FFFFFF",
            bg="#1E1E1E",
            font=("Segoe UI", 11, "bold")
        )
        img_label_title.pack(anchor="w")

        self.img_preview_label = tk.Label(
            left_frame,
            bg="#252525",
            fg="#AAAAAA",
            text="No image selected",
            width=40,
            height=12,
            relief="flat"
        )
        self.img_preview_label.pack(pady=(5, 15), fill="both", expand=False)

        csv_label_title = tk.Label(
            left_frame,
            text="CSV Preview (for LSTM / Hybrid)",
            fg="#FFFFFF",
            bg="#1E1E1E",
            font=("Segoe UI", 11, "bold")
        )
        csv_label_title.pack(anchor="w")

        csv_preview_frame = tk.Frame(left_frame, bg="#252525")
        csv_preview_frame.pack(pady=(5, 0), fill="both", expand=True)

        columns = ["index"] + list(dict.fromkeys(SEQ_FEATURE_COLS_LSTM + SEQ_FEATURE_COLS_HYBRID))
        self.csv_tree = ttk.Treeview(
            csv_preview_frame,
            columns=columns,
            show="headings",
            height=7
        )
        for col in columns:
            self.csv_tree.heading(col, text=col)
            self.csv_tree.column(col, width=80, anchor="center")

        vsb = ttk.Scrollbar(csv_preview_frame, orient="vertical", command=self.csv_tree.yview)
        self.csv_tree.configure(yscroll=vsb.set)

        self.csv_tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        style = ttk.Style()
        style.theme_use("default")
        style.configure(
            "Treeview",
            background="#252525",
            foreground="#FFFFFF",
            fieldbackground="#252525",
            rowheight=22,
            borderwidth=0
        )
        style.map("Treeview", background=[("selected", "#3A3A3A")])

        right_frame = tk.Frame(content_frame, bg="#1E1E1E")
        right_frame.pack(side="right", fill="y", padx=(10, 0))

        self.image_info_label = tk.Label(
            right_frame,
            text="Image: None",
            fg="#DDDDDD",
            bg="#1E1E1E",
            font=("Segoe UI", 9)
        )
        self.image_info_label.pack(anchor="w", pady=(10, 3))

        self.csv_info_label = tk.Label(
            right_frame,
            text="CSV: None",
            fg="#DDDDDD",
            bg="#1E1E1E",
            font=("Segoe UI", 9)
        )
        self.csv_info_label.pack(anchor="w", pady=(0, 15))

        btn_frame = tk.Frame(right_frame, bg="#1E1E1E")
        btn_frame.pack(pady=5, fill="x")

        btn_style = {
            "bg": "#2D8CFF",
            "fg": "#FFFFFF",
            "activebackground": "#1B5FBF",
            "activeforeground": "#FFFFFF",
            "font": ("Segoe UI", 10, "bold"),
            "bd": 0,
            "relief": "flat",
            "cursor": "hand2",
            "padx": 10,
            "pady": 6,
            "height": 1,
            "width": 18
        }

        img_btn = tk.Button(
            btn_frame,
            text="Select Image",
            command=self.select_image,
            **btn_style
        )
        img_btn.grid(row=0, column=0, padx=(0, 10), pady=5)

        csv_btn = tk.Button(
            btn_frame,
            text="Select CSV",
            command=self.select_csv,
            **btn_style
        )
        csv_btn.grid(row=0, column=1, padx=(0, 0), pady=5)

        clear_btn = tk.Button(
            btn_frame,
            text="Clear",
            command=self.clear_all,
            bg="#444444",
            activebackground="#333333"
        )
        clear_btn.grid(row=1, column=0, columnspan=2, pady=(10, 5), sticky="ew")

        predict_btn = tk.Button(
            right_frame,
            text="Predict",
            command=self.predict,
            **btn_style
        )
        predict_btn.pack(pady=(20, 10), fill="x")

        result_frame = tk.Frame(right_frame, bg="#252525")
        result_frame.pack(fill="both", expand=True, pady=(10, 10))

        result_title = tk.Label(
            result_frame,
            text="Prediction Result",
            fg="#FFFFFF",
            bg="#252525",
            font=("Segoe UI", 11, "bold")
        )
        result_title.pack(anchor="w", padx=10, pady=(10, 5))

        self.model_used_label = tk.Label(
            result_frame,
            text="Model: -",
            fg="#CCCCCC",
            bg="#252525",
            font=("Segoe UI", 9)
        )
        self.model_used_label.pack(anchor="w", padx=10, pady=(0, 5))

        self.pred_label = tk.Label(
            result_frame,
            text="Label: -",
            fg="#FFFFFF",
            bg="#252525",
            font=("Segoe UI", 11)
        )
        self.pred_label.pack(anchor="w", padx=10, pady=(0, 3))

        self.stage_label = tk.Label(
            result_frame,
            text="Stage: -",
            fg="#AAAAAA",
            bg="#252525",
            font=("Segoe UI", 10)
        )
        self.stage_label.pack(anchor="w", padx=10, pady=(0, 3))

        self.prob_label = tk.Label(
            result_frame,
            text="Probability: -",
            fg="#AAAAAA",
            bg="#252525",
            font=("Segoe UI", 10)
        )
        self.prob_label.pack(anchor="w", padx=10, pady=(0, 10))

        self.status_label = tk.Label(
            card,
            text="Ready",
            fg="#888888",
            bg="#1E1E1E",
            anchor="w",
            font=("Segoe UI", 9)
        )
        self.status_label.pack(side="bottom", fill="x", padx=10, pady=(0, 8))


    def select_image(self):
        path = filedialog.askopenfilename(
            title="Select fixation image",
            filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp"), ("All files", "*.*")]
        )
        if not path:
            return
        self.image_path = path
        self.image_info_label.config(text=f"Image: {os.path.basename(path)}")
        self.update_image_preview()
        self._set_status("Image selected.")

    def select_csv(self):
        path = filedialog.askopenfilename(
            title="Select CSV gaze file",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        self.csv_path = path
        self.csv_info_label.config(text=f"CSV: {os.path.basename(path)}")
        self.update_csv_preview()
        self._set_status("CSV file selected.")

    def clear_all(self):
        self.image_path = None
        self.csv_path = None
        self.image_info_label.config(text="Image: None")
        self.csv_info_label.config(text="CSV: None")
        self.img_preview_label.config(image="", text="No image selected")
        self.image_preview = None

        for row in self.csv_tree.get_children():
            self.csv_tree.delete(row)

        self.model_used_label.config(text="Model: -")
        self.pred_label.config(text="Label: -")
        self.stage_label.config(text="Stage: -")
        self.prob_label.config(text="Probability: -")
        self._set_status("Cleared all selections.")

    def update_image_preview(self):
        if not self.image_path or not os.path.exists(self.image_path):
            self.img_preview_label.config(image="", text="No image selected")
            self.image_preview = None
            return

        img = Image.open(self.image_path).convert("RGB")
        img.thumbnail((350, 250))
        self.image_preview = ImageTk.PhotoImage(img)
        self.img_preview_label.config(image=self.image_preview, text="")

    def update_csv_preview(self, max_rows=8):
        for row in self.csv_tree.get_children():
            self.csv_tree.delete(row)

        if not self.csv_path or not os.path.exists(self.csv_path):
            return

        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to read CSV file:\n{e}")
            return

        preview_cols = self.csv_tree["columns"][1:]  # skip index
        for i, row in df.head(max_rows).iterrows():
            values = [i]
            for col in preview_cols:
                values.append(row[col] if col in df.columns else "-")
            self.csv_tree.insert("", "end", values=values)

    def _set_status(self, text):
        self.status_label.config(text=text)


    def predict(self):
        if not self.image_path and not self.csv_path:
            messagebox.showwarning("No Input", "Please select an image, a CSV file, or both.")
            return

        try:
            prob = None
            model_name = ""

            if self.image_path and not self.csv_path:
                # CNN only
                model = get_cnn_model()
                img_arr = preprocess_image_for_efficientnet(self.image_path)
                pred = model.predict(img_arr)
                prob = float(pred.squeeze())
                model_name = "CNN (Image only)"

            elif self.csv_path and not self.image_path:
                # LSTM only
                model = get_lstm_model()
                seq_arr = preprocess_csv_for_lstm(self.csv_path)
                pred = model.predict(seq_arr)
                prob = float(pred.squeeze())
                model_name = "LSTM (CSV only)"

            else:
                model = get_hybrid_model()
                img_arr = preprocess_image_for_efficientnet(self.image_path)
                seq_arr = preprocess_csv_for_hybrid(self.csv_path)
                pred = model.predict([img_arr, seq_arr])
                prob = float(pred.squeeze())
                model_name = "Hybrid CNN-LSTM (Image + CSV)"

            label_text, stage_text = map_probability_to_stage(prob)

            self.model_used_label.config(text=f"Model: {model_name}")
            self.pred_label.config(text=f"Label: {label_text}")
            self.stage_label.config(text=stage_text)
            self.prob_label.config(text=f"Probability: {prob:.3f}")

            self._set_status("Prediction completed successfully.")

        except FileNotFoundError as e:
            messagebox.showerror("Model Not Found", str(e))
            self._set_status("Error: model file missing.")
        except ValueError as e:
            messagebox.showerror("Preprocessing Error", str(e))
            self._set_status("Error during preprocessing.")
        except Exception as e:
            messagebox.showerror("Prediction Error", f"An error occurred during prediction:\n{e}")
            self._set_status("Prediction failed due to an error.")



if __name__ == "__main__":
    root = tk.Tk()
    app = DyslexiaApp(root)
    root.mainloop()
