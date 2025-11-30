import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)

import tensorflow as tf
tf.random.set_seed(42)

from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras import layers, models, Input, Model
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, roc_curve, roc_auc_score
)

BASE_FOLDER = r"D:\PREDICTION OF DYSLEXIA\DATASETS\13332134"
IMG_FOLDER  = os.path.join(BASE_FOLDER, "fixation_images", "MeaningfulText")
FIX_FOLDER  = os.path.join(BASE_FOLDER, "final-data", "data")
LABEL_CSV   = os.path.join(BASE_FOLDER, "dyslexia_class_label.csv")

labels_df = pd.read_csv(LABEL_CSV)

def img_path(subject_id):
    return os.path.join(IMG_FOLDER, f"Subject_{subject_id}_MeaningfulText.png")

def fix_path(subject_id):
    return os.path.join(FIX_FOLDER, f"Subject_{subject_id}_T4_Meaningful_Text_fixations.csv")

labels_df["img_path"] = labels_df["subject_id"].astype(str).apply(img_path)
labels_df["fix_path"] = labels_df["subject_id"].astype(str).apply(fix_path)

labels_df = labels_df[
    labels_df["img_path"].apply(os.path.exists)
    & labels_df["fix_path"].apply(os.path.exists)
].copy()

labels_df["label_int"] = labels_df["label"].map({"non-dyslexic": 0, "dyslexic": 1}).astype(int)

print("Total usable samples:", len(labels_df))
print(labels_df[["subject_id", "label", "img_path", "fix_path"]].head())

IMG_SIZE = (224, 224)
MAX_TIMESTEPS = 100
SEQ_FEATURES = ["fix_x", "fix_y", "duration_ms"]

X_img = []
X_seq = []
y = []

for _, row in labels_df.iterrows():
    img = load_img(row["img_path"], target_size=IMG_SIZE)
    img_arr = img_to_array(img)
    img_arr = preprocess_input(img_arr)
    X_img.append(img_arr)

    df_fix = pd.read_csv(row["fix_path"])
    seq = df_fix[SEQ_FEATURES].values.astype(np.float32)

    if len(seq) > MAX_TIMESTEPS:
        seq = seq[:MAX_TIMESTEPS]
    else:
        pad_len = MAX_TIMESTEPS - len(seq)
        pad_block = np.zeros((pad_len, len(SEQ_FEATURES)), dtype=np.float32)
        seq = np.vstack([seq, pad_block])

    X_seq.append(seq)
    y.append(row["label_int"])

X_img = np.array(X_img, dtype=np.float32)
X_seq = np.array(X_seq, dtype=np.float32)
y = np.array(y, dtype=np.int64)

print("X_img shape:", X_img.shape)
print("X_seq shape:", X_seq.shape)
print("y shape:", y.shape)

X_img_train, X_img_test, X_seq_train, X_seq_test, y_train, y_test = train_test_split(
    X_img, X_seq, y, test_size=0.3, random_state=42, stratify=y
)

print("Train samples:", len(y_train), "Test samples:", len(y_test))

img_input = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="img_input")

base_cnn = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_cnn.trainable = False

x = base_cnn(img_input, training=False)
x = layers.GlobalAveragePooling2D(name="gap")(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu', name="cnn_dense_128")(x)
x = layers.Dropout(0.3)(x)
cnn_features = x
seq_input = Input(shape=(MAX_TIMESTEPS, len(SEQ_FEATURES)), name="seq_input")

s = layers.Masking(mask_value=0.0)(seq_input)
s = layers.Bidirectional(
        layers.LSTM(64, return_sequences=False, dropout=0.25, recurrent_dropout=0.15),
        name="bilstm_64"
    )(s)
s = layers.Dense(32, activation='relu', name="lstm_dense_32")(s)
s = layers.Dropout(0.3)(s)
lstm_features = s

combined = layers.Concatenate(name="fusion")([cnn_features, lstm_features])
z = layers.Dense(32, activation='relu', name="fusion_dense_32")(combined)
z = layers.Dropout(0.3)(z)
output = layers.Dense(1, activation='sigmoid', name="output")(z)

hybrid_model = Model(inputs=[img_input, seq_input], outputs=output)

hybrid_model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name="auc")]
)

hybrid_model.summary()

early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = hybrid_model.fit(
    [X_img_train, X_seq_train], y_train,
    validation_data=([X_img_test, X_seq_test], y_test),
    epochs=25,
    batch_size=8,
    callbacks=[early_stop],
    verbose=2
)

print("\n=== HYBRID CNN–LSTM FINAL RESULTS ===")
y_probs = hybrid_model.predict([X_img_test, X_seq_test]).ravel()
y_pred = (y_probs > 0.5).astype(int)

acc = accuracy_score(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_probs)

print(f"Test Accuracy: {acc:.4f}")
print(f"Test AUC:      {auc_score:.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Non-Dyslexic', 'Dyslexic']))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 5))
sns.heatmap(
    cm, annot=True, fmt='d', cmap='Blues',
    xticklabels=['Non-Dyslexic', 'Dyslexic'],
    yticklabels=['Non-Dyslexic', 'Dyslexic']
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix - Hybrid CNN–LSTM ")
plt.tight_layout()
plt.show()

fpr, tpr, _ = roc_curve(y_test, y_probs)
plt.figure(figsize=(5, 5))
plt.plot(fpr, tpr, label=f"AUC = {auc_score:.4f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Hybrid CNN–LSTM ")
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

hybrid_model.save("hybrid.keras")
print("Hybrid model saved as hybrid.keras")
