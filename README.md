# MNIST Digit Recognition with Neural Networks

## Overview

This project demonstrates how to build and train a neural network to recognize handwritten digits from the MNIST dataset. The code is implemented in a Google Colab notebook and utilizes popular Python libraries such as TensorFlow, Keras, NumPy, Matplotlib, and Seaborn for data manipulation, visualization, and model building. The project also demonstrates how to load an external image and use it as an input for the trained model.

## Dataset

The MNIST dataset is used for training and testing the model. It consists of 70,000 images of handwritten digits (0-9), each with a size of 28x28 pixels. The dataset is split into a training set (60,000 images) and a testing set (10,000 images).

## Project Structure

The project is structured as follows:

1.  **Import Libraries:** Import necessary Python libraries, including `tensorflow`, `keras`, `numpy`, `matplotlib`, `seaborn`, and `cv2`.
2.  **Load MNIST Dataset:** The MNIST dataset is loaded using `keras.datasets.mnist.load_data()`.
3.  **Data Exploration:**
    *   The shape of the training and testing data is printed.
    *   The raw pixel values of images are printed.
    *   The label associated with each image is printed.
    *   An example image is displayed using `matplotlib.pyplot.imshow()`.
    *   The unique values in the target variable are printed.
4.  **Data Preprocessing:**
    *   Pixel values are scaled down to the range \[0, 1] by dividing by 255.
5.  **Model Building:**
    *   A sequential neural network model is created using `keras.Sequential()`.
    *   The model consists of an input flattening layer, two hidden dense layers with ReLU activation, and an output dense layer with sigmoid activation.
6.  **Model Compilation:**
    *   The model is compiled with the Adam optimizer, sparse categorical cross-entropy loss function, and accuracy as the evaluation metric.
7.  **Model Training:**
    *   The model is trained using the training data for 10 epochs.
8.  **Model Evaluation:**
    *   The model is evaluated using the test data, and the test accuracy is printed.
9.  **Prediction on Test Data:**
    *   The model predicts the labels for the test data using `model.predict()`.
    *   The prediction probabilities are converted to class labels using `np.argmax()`.
10. **Confusion Matrix:**
    *   A confusion matrix is created using `tensorflow.math.confusion_matrix()` to visualize the model's performance.
    *   The confusion matrix is displayed as a heatmap using `seaborn.heatmap()`.
11. **Prediction on a New Image:**
    * An external image is loaded from the path `'/content/MNIST_digit.png'` using `cv2.imread()`.
    * The image is converted to grayscale using `cv2.cvtColor()`.
    * The image is resized to 28x28 pixels using `cv2.resize()`.
    * The image pixel values are scaled down to the range \[0, 1] by dividing by 255.
    * The image is reshaped to have the required dimensions for the model.
    * The model predicts the digit of the image.
    * The predicted digit is printed.

## Libraries Used

*   **TensorFlow:** For building and training the neural network. Version 2.15.0 is used.
*   **Keras:** A high-level API for building and training deep learning models (integrated with TensorFlow). Version 2.15.0 is used.
*   **NumPy:** For numerical operations and array handling. Version 1.23.5 is used.
*   **Matplotlib:** For creating static, interactive, and animated visualizations. Version 3.7.1 is used.
*   **Seaborn:** For statistical data visualization. Version 0.12.2 is used.
*   **OpenCV (cv2):** For image processing. Version 4.8.0 is used.

## How to Run the Code

1.  The code is developed to be run in a Google Colab environment.
2.  Open the Colab notebook file.
3.  Run each cell sequentially.
4. You must provide an image called `MNIST_digit.png` in the folder `/content/` in order to run the code that predicts the class of this image.

## Key Concepts Demonstrated

*   **Neural Networks:** Building and training a simple neural network.
*   **MNIST Dataset:** Working with a common dataset for image recognition.
*   **Data Preprocessing:** Scaling pixel values.
*   **Model Evaluation:** Calculating and understanding accuracy.
*   **Confusion Matrix:** Using a confusion matrix to evaluate model performance.
*   **Image Loading and Preprocessing**: Loading an external image, resizing it, converting it to grayscale, scaling pixel values, and preparing it to be an input of the model.
* **Making predictions**: Using the model to predict the class of the image.

## Author

[Your Name]
