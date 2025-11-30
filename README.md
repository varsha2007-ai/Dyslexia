# Dyslexia
Deep-learning system for predicting dyslexia and its severity using eye-gaze data from the ZTDD70 dataset. Includes LSTM for gaze sequences, CNN for fixation images, and a hybrid CNN-LSTM model, with a GUI that auto-selects the best model based on the input.

About This Project

This work focuses on identifying dyslexia and estimating its severity by analysing eye-gaze behaviour during reading. The approach uses the ZTDD70 eye-tracking dataset, which contains fixation coordinates, durations, and gaze-path images captured from dyslexic and non-dyslexic readers. These behavioural patterns reveal how readers process text and provide meaningful indicators for classification.

To learn these patterns effectively, three deep-learning models were developed:

LSTM to capture temporal reading sequences from fixation CSV files

CNN to extract spatial fixation features from gaze-map images

Hybrid CNN-LSTM to fuse both sequence and spatial information for improved prediction

Because the dataset is small, synthetic gaze sequences were generated using controlled perturbation methods to preserve natural eye-movement behaviour while increasing training stability.

A Tkinter-based graphical interface is included, allowing users to upload a fixation image, gaze CSV, or both. The system automatically chooses the appropriate model and displays the predicted class and severity level.

Overall, this project demonstrates a complete pipeline—from data preparation and feature engineering to model training, fusion, evaluation, and GUI deployment—for building an interpretable and efficient dyslexia-screening tool.
