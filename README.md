# 🌿 Plant Disease Classification using Deep Learning

A deep learning-based system for automatic plant disease detection from leaf images.
The project leverages Convolutional Neural Networks (CNNs) and advanced architectures to classify plant leaves into healthy or diseased categories, supporting early diagnosis and smart agriculture solutions.

---

## 🚀 Features

* 🌱 Classification of plant diseases across **38 classes**
* 🧠 Comparison of multiple models:

  * Custom CNN
  * ResNet
  * EfficientNet-B3 (best performance ~99.9%)
* 🔄 Transfer Learning using pre-trained models
* 📈 Data augmentation and preprocessing pipeline
* 🛡️ Overfitting reduction (Dropout, Early Stopping, hyperparameter tuning)
* 📊 Model evaluation using Accuracy, Precision, Recall, F1-score, and Confusion Matrix
* 🚨 Anomaly detection module using **MSP + GOAD**
* ⚡ GPU-accelerated training (Kaggle environment)
* 🐳 Docker support for deployment
* 🔧 Version control with Git

---

## 🧠 Model Architecture

The system uses **EfficientNet-B3** as the main backbone for feature extraction, combined with:

* MSP (Maximum Softmax Probability) for confidence-based detection
* GOAD (Geometric Transformation-based Anomaly Detection) for detecting out-of-distribution samples

---

## 📂 Dataset

* PlantVillage Dataset
* ~70,000+ training images
* 38 plant disease classes

---

## 📊 Results

* ✅ Accuracy: ~99.9%
* ✅ High Precision, Recall, F1-score
* ✅ Strong generalization with minimal overfitting

---

## 🛠️ Tech Stack

* Python
* PyTorch
* NumPy / Pandas
* Matplotlib / Seaborn
* Docker

---

## 📌 Use Cases

* Early plant disease detection
* Smart agriculture systems
* AI-based crop monitoring

---

## 🔗 Repository

👉 https://github.com/moad-cod/-Plant-Disease-Classification-using-CNN

---

## 📬 Author

**Mouad Elbaz**
MLOps & AI Developer
