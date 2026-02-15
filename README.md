# Turkish Sign Language Alphabet Recognition  
# Türk İşaret Dili Alfabe Tanıma Sistemi

---

## 🇬🇧 English

### 📌 Project Description

This project presents a real-time Turkish Sign Language (TSL) alphabet recognition system using MediaPipe hand landmark detection and classical machine learning techniques.

The system captures hand gestures through a webcam, extracts 3D hand landmark coordinates, and classifies alphabet gestures using a K-Nearest Neighbors (KNN) model.

The project was developed as a Graduation Project in Computer Engineering and is designed with scalability and extensibility in mind.

---

### 🎯 Objectives

- Build a real-time alphabet recognition system
- Use landmark-based feature extraction instead of raw image classification
- Design a scalable data collection pipeline
- Develop a machine learning-based classification system

---

### 🧠 Technologies Used

- Python
- OpenCV
- MediaPipe
- Scikit-learn
- NumPy
- Pandas

---

### ⚙️ System Architecture

1. **Data Collection**
   - Webcam-based gesture capture
   - Structured dataset organization

2. **Landmark Extraction**
   - 21 hand landmarks per hand
   - Normalized coordinate extraction
   - CSV-based dataset generation

3. **Model Training**
   - K-Nearest Neighbors (KNN) classifier
   - Feature vector based learning

4. **Real-Time Prediction**
   - Live hand tracking
   - Instant alphabet classification

---

### 🚀 Installation

```bash
git clone https://github.com/yourusername/turkish-sign-language-recognition.git
cd turkish-sign-language-recognition
pip install -r requirements.txt
