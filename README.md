# Gender Classification

This project aims to develop a machine learning model that can predict gender based on a person's photo or camera. Using features extracted from facial images, a model is created that distinguishes between male and female classes. The project combines classical machine learning techniques (SVM) and image processing methods to provide a fast and resource-efficient solution.

## Libraries Used

- **OpenCV**: A library for image processing and face detection.
- **scikit-learn**: A library for training and evaluating a Support Vector Machines (SVM) model.
- **joblib**: A library for saving and loading the trained model.
- **NumPy**: A library for numerical calculations and array operations.

## Data Set

In the project, an open source dataset **https://www.kaggle.com/datasets/cashutosh/gender-classification-dataset** was used. This dataset contains labeled facial images of tens of thousands of different people. The dataset is divided into two separate folders for training and validation:

- **Training**: Training data (Male and Female folders).
- **Validation**: Validation data (Male and Female folders).

## Working Logic of the Project

1. **Data Loading and Processing**: Images are converted to black and white format and resized to 64x64 pixels.
2. **Feature Extraction**: Images are vectorized and matched with labels.
3. **Model Training**: The model is trained using Support Vector Machines (SVM).
4. **Validation**: The trained model is tested on the validation dataset and performance metrics are calculated.
5. **Model Save**: The trained model is saved using `joblib`.
6. **Real Time Prediction**: Faces are detected in webcam images and gender is estimated.
