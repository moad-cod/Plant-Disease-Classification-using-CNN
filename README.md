# 🌿 Plant Disease Classification using Deep Learning

> **An AI-powered web application for automatic plant disease diagnosis using state-of-the-art deep learning models.**

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red?logo=pytorch)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit)
![Docker](https://img.shields.io/badge/Docker-Deployment-2496ED?logo=docker)
![License](https://img.shields.io/badge/License-MIT-green)

---

## 🚀 Live Demo

🌐 **Try the application**

**https://plant-disease-classification-using-cnn-vtz7rssrkv7nupz7b3gpyb.streamlit.app/**

---

## 📖 Overview

Plant diseases significantly impact agricultural productivity and food security. Early and accurate diagnosis enables farmers to take timely action, reducing crop losses and improving yield.

This project leverages **Convolutional Neural Networks (CNNs)** and **Transfer Learning** to classify plant leaf images into **38 different healthy and diseased categories** with exceptional accuracy.

The application provides an intuitive SaaS-inspired interface where users can upload a leaf image and receive instant AI-powered predictions along with confidence scores.

---

# ✨ Features

* 🌿 Classifies **38 plant disease classes**
* ⚡ Real-time prediction through an interactive Streamlit web application
* 🧠 Multiple deep learning architectures

  * Custom CNN
  * ResNet
  * **EfficientNet-B3 (Best Performing Model)**
* 🔄 Transfer Learning with pretrained ImageNet weights
* 📈 Advanced preprocessing & data augmentation
* 🎯 Confidence score visualization
* 🚨 Out-of-distribution detection using:

  * Maximum Softmax Probability (MSP)
  * GOAD (Geometric Transformation-based Anomaly Detection)
* 🛡️ Overfitting prevention

  * Early Stopping
  * Dropout
  * Learning Rate Scheduling
  * Hyperparameter Optimization
* 🐳 Docker support
* 📱 Modern responsive SaaS-style UI
* 🚀 GPU-accelerated model training

---

# 🖥️ Application Preview

The web application provides:

* Professional dashboard UI
* Drag & Drop image upload
* Instant disease prediction
* Prediction confidence
* Disease information
* Responsive design
* Smooth animations

---

# 🧠 Model Architecture

The final deployed model is based on **EfficientNet-B3**, chosen for its excellent balance between accuracy and computational efficiency.

### Pipeline

Leaf Image

⬇️

Image Preprocessing

⬇️

EfficientNet-B3 Feature Extraction

⬇️

Classification Head

⬇️

Disease Prediction

⬇️

Confidence Estimation (MSP)

⬇️

Anomaly Detection (GOAD)

---

# 📂 Dataset

### PlantVillage Dataset

* 📸 70,000+ leaf images
* 🌱 38 disease categories
* 🍅 Multiple crop species
* ✅ Healthy & infected leaves

The dataset includes diseases affecting:

* Apple
* Corn
* Tomato
* Potato
* Pepper
* Cherry
* Grape
* Peach
* Soybean
* Strawberry
* Squash
* Orange
* Raspberry
* Blueberry

---

# 📊 Performance

| Metric    |     Score |
| --------- | --------: |
| Accuracy  | **99.9%** |
| Precision | **99.9%** |
| Recall    | **99.9%** |
| F1-Score  | **99.9%** |

The EfficientNet-B3 model achieved outstanding performance while maintaining excellent generalization on unseen data.

---

# 🛠️ Tech Stack

### Deep Learning

* PyTorch
* TorchVision
* EfficientNet-B3
* ResNet
* CNN

### Data Processing

* NumPy
* Pandas
* OpenCV
* Pillow

### Visualization

* Matplotlib
* Seaborn

### Web Application

* Streamlit
* HTML/CSS
* Responsive UI

### Deployment

* Docker
* Git
* GitHub
* Streamlit Community Cloud

---

# 📁 Project Structure

```text
Plant-Disease-Classification-using-CNN/
│
├── data/
├── models/
├── notebooks/
├── src/
│   ├── app.py
│   ├── inference.py
│   ├── preprocessing.py
│   └── utils.py
│
├── assets/
├── requirements.txt
├── Dockerfile
└── README.md
```

---

# 🚀 Run Locally

Clone the repository

```bash
git clone https://github.com/moad-cod/-Plant-Disease-Classification-using-CNN.git
```

Go to the project

```bash
cd -Plant-Disease-Classification-using-CNN
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

### Linux / macOS

```bash
source .venv/bin/activate
```

### Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run src/app.py
```

---

# 🐳 Docker

Build

```bash
docker build -t plant-disease-classifier .
```

Run

```bash
docker run -p 8501:8501 plant-disease-classifier
```

---

# 🌾 Use Cases

* Smart Agriculture
* Precision Farming
* Crop Monitoring
* Agricultural Research
* Educational Projects
* AI-powered Farming Assistants

---

# 🔮 Future Improvements

* Mobile application
* Explainable AI (Grad-CAM)
* Multi-language support
* Treatment recommendations
* Disease severity estimation
* REST API
* Cloud deployment on AWS/Azure
* Farmer dashboard
* Batch image prediction
* Historical prediction tracking

---

# 🤝 Contributing

Contributions are welcome!

If you'd like to improve the project:

1. Fork the repository
2. Create a new feature branch
3. Commit your changes
4. Open a Pull Request

---

# ⭐ Support

If you found this project helpful, consider giving it a **⭐ Star** on GitHub.

---

# 📬 Contact

**Mouad Elbaz**

**AI Engineer | Machine Learning | Deep Learning | MLOps**

GitHub:

https://github.com/moad-cod

LinkedIn:

https://www.linkedin.com/in/mouad-elbaz

X:

https://x.com/MouadEl_AI
