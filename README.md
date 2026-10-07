# CIFAKE Deepfake Image Detector

## 1. Project Overview

CIFAKE Deepfake Image Detector is a deep-learning based image classification system that identifies whether an input image is REAL or FAKE.

The project uses the CIFAKE dataset and an EfficientNetB0 convolutional neural network for binary image classification. The trained model is integrated with a Flask web application, allowing users to upload an image and receive a prediction through a simple web interface.

---

## 2. Objectives

- To classify images as REAL or FAKE using deep learning.
- To train an EfficientNetB0 model using the CIFAKE dataset.
- To evaluate the trained model using validation and test images.
- To deploy the trained model through a Flask web application.
- To provide a simple interface for uploading images and viewing predictions.

---

## 3. Technologies Used

- Python
- TensorFlow
- Keras
- EfficientNetB0
- Flask
- HTML
- CSS
- Pillow
- NumPy
- VS Code

---

## 4. Dataset

The project uses the CIFAKE dataset.

The dataset contains two classes:

- REAL
- FAKE

The project uses the following structure:

    train/
    ├── REAL/
    └── FAKE/

    test/
    ├── REAL/
    └── FAKE/

The training dataset contains 100,000 images and the test dataset contains 20,000 images in the current project setup.

During training:

- 90,000 images were used for training.
- 10,000 images were used for validation.
- 20,000 images were used for testing.

---

## 5. Model

The project uses EfficientNetB0 as the image classification model.

Input image size:

    128 × 128 pixels

The model performs binary classification:

    REAL = 0
    FAKE = 1

The trained model is stored as:

    models/best_model.keras

---

## 6. Training

The model was trained using TensorFlow and Keras.

Training configuration:

- Model: EfficientNetB0
- Image size: 128 × 128
- Batch size: 32
- Maximum epochs: 5
- Number of classes: 2

Results recorded from the completed training run:

- Accuracy: 88.68%
- Precision: 88.00%
- Recall: 89.57%
- Loss: 0.2744

These values correspond to the evaluation output produced during the completed training run.

---

## 7. Web Application

The trained model is deployed using Flask.

The web application allows the user to:

1. Select an image.
2. Preview the uploaded image.
3. Click the Analyze Image button.
4. Send the image to the Flask backend.
5. Process the image using the trained model.
6. Display the predicted class and confidence.

---

## 8. Project Structure

    CIFAKE_FINAL_VSCODE_PY313/
    │
    ├── app.py
    ├── train_model.py
    ├── requirements.txt
    ├── README.md
    │
    ├── models/
    │   ├── best_model.keras
    │   └── model_metadata.json
    │
    ├── templates/
    │   └── index.html
    │
    ├── static/
    │   ├── style.css
    │   └── uploads/
    │
    ├── train/
    │   ├── REAL/
    │   └── FAKE/
    │
    └── test/
        ├── REAL/
        └── FAKE/

---

## 9. Installation

Create and activate the virtual environment.

On Windows:

    .venv\Scripts\Activate.ps1

Install the required packages:

    pip install -r requirements.txt

---

## 10. Running the Application

Make sure the virtual environment is activated.

Run:

    python app.py

Then open the following address in a web browser:

    http://127.0.0.1:5000

---

## 11. Model Training

Training is only required when creating or updating the model.

Run:

    python train_model.py

The trained model is saved to:

    models/best_model.keras

After the model has already been trained, the web application can be started directly using:

    python app.py

---

## 12. Prediction

The system accepts supported image files through the web interface.

The prediction process is:

    Uploaded Image
          ↓
    Image Preprocessing
          ↓
    EfficientNetB0
          ↓
    Binary Classification
          ↓
    REAL / FAKE
          ↓
    Confidence Display

---

## 13. Testing

The application was tested using images from:

- Training REAL dataset
- Training FAKE dataset
- Test REAL dataset
- Test FAKE dataset

The web application successfully accepted uploaded images and generated predictions.

---

## 14. Limitations

- The prediction depends on the characteristics of the training dataset.
- Images that are very different from the training data may be classified incorrectly.
- A single prediction should not be considered absolute proof that an image is real or fake.
- The current model is designed for image classification rather than forensic verification.

---

## 15. Future Scope

Possible improvements include:

- Larger and more diverse datasets.
- More advanced deep-learning architectures.
- Explainable AI for showing important image regions.
- Improved detection of manipulated images.
- Video deepfake detection.
- Real-time camera-based detection.
- Model optimization for faster inference.
- Cloud deployment.

---

## 16. Conclusion

The CIFAKE Deepfake Image Detector demonstrates how deep learning can be used to classify images as REAL or FAKE. EfficientNetB0 was trained using the CIFAKE dataset and successfully integrated with a Flask web application.

The final system provides a simple interface where users can upload an image and obtain a model-based prediction with confidence.