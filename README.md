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

1. DATA COLLECTION
This project utilizes the ZTDD70 eye-tracking dataset, which provides fixation coordinates, fixation durations, and gaze-path images collected from dyslexic and non-dyslexic individuals during reading tasks.
Given the limited number of subjects, additional samples were created through a controlled synthetic augmentation method. Fixation coordinates and durations were perturbed slightly while maintaining the natural structure of reading patterns. This approach increased dataset size, improved training stability, and reduced the risk of overfitting.

2. MODELS USED:
The system incorporates three deep-learning models to analyze both temporal and spatial characteristics of eye movements.

LSTM Model

Designed to capture temporal reading behaviour from fixation CSV files. Preprocessing includes:

Removal of invalid or empty sequences

Computation of saccade length for each fixation transition

Padding or truncating all sequences to 100 time steps

Standardization of feature values

CNN Model

Used to interpret spatial fixation patterns from gaze-map images. Preprocessing steps include image resizing, normalization, and extraction of spatial features through a pretrained EfficientNetB0 backbone.

Hybrid CNN–LSTM Model

Combines the outputs of the CNN and LSTM models to learn both spatial and sequential dependencies. Feature vectors from both pathways are fused and passed through dense layers for final classification. This integrated approach enhances robustness by leveraging complementary information sources.

3. GRAPHICAL USER INTERFACE(GUI):

A Tkinter-based GUI was developed to enable accessible real-time prediction. The interface supports three modes of operation:

Image input only: Routes through the CNN model

CSV sequence only: Processes through the LSTM model

Image and CSV input together: Utilizes the hybrid model

The GUI displays the uploaded inputs, predicted class label (dyslexic or non-dyslexic), and the estimated severity level based on fixation-related thresholds.

CONCLUSION:

This project presents a complete end-to-end workflow for dyslexia identification using eye-gaze behaviour. Through dataset enhancement, rigorous preprocessing, and the integration of LSTM, CNN, and hybrid architectures, the system provides a reliable and interpretable method for dyslexia screening. The accompanying GUI further supports practical deployment and ease of use.
