import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, roc_auc_score, accuracy_score
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout, Bidirectional, Masking
from tensorflow.keras.callbacks import EarlyStopping
import os
import glob
import warnings
import random

warnings.filterwarnings('ignore')

np.random.seed(42)
os.environ['PYTHONHASHSEED'] = '33'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(33)
tf.random.set_seed(33)


class DyslexiaClassifier:
    def __init__(self, data_folder_path, max_timesteps=100):
        self.data_folder_path = data_folder_path
        self.features_df = None
        self.model = None
        self.scaler = StandardScaler()
        self.seq_data = None
        self.seq_labels = None
        self.seq_subjects = None
        self.seq_feature_cols = None
        self.lstm_model = None
        self.history = None
        self.max_timesteps = max_timesteps

    def load_subject_labels(self, labels_file):
        labels_df = pd.read_csv(labels_file)
        return dict(zip(labels_df['subject_id'], labels_df['class_id']))
    def prepare_lstm_sequences(self, labels_file, max_timesteps=None):
        if max_timesteps is None:
            max_timesteps = self.max_timesteps
        else:
            self.max_timesteps = max_timesteps

        subject_labels = self.load_subject_labels(labels_file)
        fixation_files = glob.glob(os.path.join(self.data_folder_path, "*_fixations.csv"))

        seq_data = []
        seq_labels = []
        seq_subjects = []

        for file_path in fixation_files:
            filename = os.path.basename(file_path)
            subject_id = int(filename.split('_')[1])

            if subject_id not in subject_labels:
                continue

            df = pd.read_csv(file_path)

            df = df[['fix_x', 'fix_y', 'duration_ms']].dropna()

            if df.empty:
                continue

            if len(df) > 1:
                saccades = np.sqrt(np.diff(df['fix_x']) ** 2 + np.diff(df['fix_y']) ** 2)
                saccades = np.concatenate([[0], saccades])  # first fixation has 0 saccade
            else:
                saccades = np.array([0] * len(df))

            df['saccade_len'] = saccades

            df['duration_ms'] = np.clip(df['duration_ms'], 0, df['duration_ms'].quantile(0.99))

            df = df.replace([np.inf, -np.inf], np.nan).fillna(0)

            seq_feature_cols = ['fix_x', 'fix_y', 'duration_ms', 'saccade_len']
            arr = df[seq_feature_cols].values.astype(np.float32)

            if len(arr) > max_timesteps:
                arr = arr[:max_timesteps]
            else:
                pad_len = max_timesteps - len(arr)
                pad_block = np.zeros((pad_len, len(seq_feature_cols)), dtype=np.float32)
                arr = np.vstack([arr, pad_block])

            seq_data.append(arr)
            seq_labels.append(subject_labels[subject_id])
            seq_subjects.append(subject_id)

        self.seq_data = np.array(seq_data, dtype=np.float32)
        self.seq_labels = np.array(seq_labels, dtype=np.int64)
        self.seq_subjects = np.array(seq_subjects, dtype=np.int32)
        self.seq_feature_cols = seq_feature_cols

        print(f"LSTM Dataset shapes: X={self.seq_data.shape}, y={self.seq_labels.shape}")
        print(f"Sequence features used: {self.seq_feature_cols}")

    def build_lstm_model(self, input_shape):
        inp = Input(shape=input_shape, name="seq_input")
        x = Masking(mask_value=0.0, name="masking")(inp)
        x = Bidirectional(
            LSTM(
                64,
                return_sequences=False,
                dropout=0.25,
                recurrent_dropout=0.15
            ),
            name="bilstm_64"
        )(x)
        x = Dense(32, activation='relu', name="dense_32")(x)
        x = Dropout(0.3, name="dropout_0_3")(x)
        out = Dense(1, activation='sigmoid', name="output")(x)

        model = Model(inputs=inp, outputs=out, name="LSTM_Sequence_Model")
        model.compile(
            optimizer='adam',
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name="auc")]
        )
        return model

    def train_lstm(self, test_size=0.2, epochs=30, batch_size=16):
        if self.seq_data is None:
            raise ValueError("Sequence data not prepared. Call prepare_lstm_sequences() first.")

        X = self.seq_data
        y = self.seq_labels
        max_timesteps = self.max_timesteps
        n_features = X.shape[2]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=42,
            stratify=y
        )

        n_train, t, f = X_train.shape
        n_test = X_test.shape[0]

        X_train_flat = X_train.reshape(-1, f)
        X_test_flat = X_test.reshape(-1, f)

        self.scaler.fit(X_train_flat)
        X_train_scaled = self.scaler.transform(X_train_flat)
        X_test_scaled = self.scaler.transform(X_test_flat)

        X_train_scaled = X_train_scaled.reshape(n_train, t, f)
        X_test_scaled = X_test_scaled.reshape(n_test, t, f)

        model = self.build_lstm_model(input_shape=(max_timesteps, n_features))

        early_stop = EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        )

        self.history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_test_scaled, y_test),
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=2
        )

        print("\n=== LSTM FINAL RESULTS ===")
        test_probs = model.predict(X_test_scaled).ravel()
        preds = (test_probs > 0.5).astype(int)

        auc = roc_auc_score(y_test, test_probs)
        acc = accuracy_score(y_test, preds)
        print(f"LSTM Test Accuracy: {acc:.4f}")
        print(f"LSTM Test AUC:      {auc:.4f}")
        print("\nLSTM Classification Report:")
        print(classification_report(y_test, preds, target_names=['Non-Dyslexic', 'Dyslexic']))

        cm = confusion_matrix(y_test, preds)
        print("\nConfusion Matrix (rows = true, cols = predicted):")
        print("          Pred 0   Pred 1")
        print(f"True 0    {cm[0, 0]:6d} {cm[0, 1]:7d}")
        print(f"True 1    {cm[1, 0]:6d} {cm[1, 1]:7d}")

        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non-Dyslexic', 'Dyslexic'],
            yticklabels=['Non-Dyslexic', 'Dyslexic']
        )
        plt.title("Confusion Matrix - LSTM")
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.tight_layout()
        plt.show()

        fpr, tpr, thresholds = roc_curve(y_test, test_probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.4f}")
        plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve - LSTM")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show()

        self.lstm_model = model
        return model


def main():
    classifier = DyslexiaClassifier(
        data_folder_path=r"D:\PREDICTION OF DYSLEXIA\DATASETS\13332134\final-data\data",
        max_timesteps=100
    )
    try:
        classifier.prepare_lstm_sequences(
            r"D:\PREDICTION OF DYSLEXIA\Backend\expanded_file.csv",
            max_timesteps=100
        )
        lstm_model = classifier.train_lstm(epochs=25, batch_size=16)
    except Exception as e:
        print("LSTM training failed:", e)
        lstm_model = None

    if lstm_model is not None:
        classifier.lstm_model.save('lstm_model.keras')
        print("LSTM model saved as lstm_model.keras")


if __name__ == "__main__":
    main()
