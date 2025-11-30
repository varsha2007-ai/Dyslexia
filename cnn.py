import os
import pandas as pd
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report, accuracy_score,
    precision_score, recall_score, f1_score, roc_curve, roc_auc_score
)
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import random

random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)

images_folder = r"D:\PREDICTION OF DYSLEXIA\DATASETS\13332134\fixation_images\MeaningfulText"
labels_csv = r"D:\PREDICTION OF DYSLEXIA\DATASETS\13332134\dyslexia_class_label.csv"

df = pd.read_csv(labels_csv)

def get_image_path(subject_id):
    return os.path.join(images_folder, f"Subject_{str(subject_id).strip()}_MeaningfulText.png")

df['filepath'] = df['subject_id'].apply(get_image_path)
df = df[df['filepath'].apply(os.path.exists)]

df['label_enc'] = df['label'].map({'non-dyslexic': 0, 'dyslexic': 1})

train_df, val_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    stratify=df['label_enc']
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 8

train_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    horizontal_flip=True,
    rotation_range=10,
    width_shift_range=0.05,
    height_shift_range=0.05,
    zoom_range=0.1
)

val_gen = ImageDataGenerator(
    preprocessing_function=preprocess_input
)

train_data = train_gen.flow_from_dataframe(
    train_df,
    x_col='filepath',
    y_col='label_enc',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw'
)

val_data = val_gen.flow_from_dataframe(
    val_df,
    x_col='filepath',
    y_col='label_enc',
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='raw',
    shuffle=False
)

base_model = EfficientNetB0(
    include_top=False,
    weights='imagenet',
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)

base_model.trainable = False

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base_model(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_data,
    epochs=15,
    validation_data=val_data
)

y_true = val_data.labels
y_pred_probs = model.predict(val_data).flatten()
y_pred = (y_pred_probs > 0.5).astype(int)

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5,5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=['Non-dyslexic', 'Dyslexic'],
    yticklabels=['Non-dyslexic', 'Dyslexic']
)
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print("\nClassification Report:\n",
      classification_report(y_true, y_pred,
                            target_names=['Non-dyslexic', 'Dyslexic'],
                            zero_division=0))

fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
auc = roc_auc_score(y_true, y_pred_probs)
plt.figure()
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
print(f"AUC (Area Under Curve): {auc:.4f}")

model.save("cnn_model.h5")
