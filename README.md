# Turkish Sign Language Alphabet Recognition  
# Türk İşaret Dili Alfabe Tanıma Sistemi

---

# 🇬🇧 English

# Turkish Sign Language Alphabet Recognition System

## 📌 Project Description

This project presents a real-time Turkish Sign Language (TSL) alphabet recognition system developed using MediaPipe hand landmark detection and machine learning algorithms.

The system captures hand gestures through a webcam, extracts 3D hand landmark coordinates for each detected hand, and constructs structured feature vectors. These features are then classified using the K-Nearest Neighbors (KNN) algorithm to recognize alphabet gestures.

The project was developed as a Computer Engineering Graduation Project and is designed with a modular and scalable architecture.

---

## 🎯 Project Objectives

- Develop a real-time alphabet recognition system  
- Use landmark-based feature extraction instead of raw image classification  
- Design a structured and extensible data collection pipeline  
- Perform gesture classification using machine learning  

---

## 🧠 Why Landmark-Based Approach?

Instead of training directly on raw image data, this project uses structured 3D hand landmark coordinates.

Advantages of this approach:

- Lower computational cost  
- Reduced dataset requirement  
- Faster training process  
- Less sensitivity to lighting and background variations  
- More interpretable and structured feature vectors  

This design makes the system efficient and suitable for real-time applications, while also allowing future integration with deep learning models.

---

## ⚙️ System Architecture

```
Webcam Input
      ↓
MediaPipe Hand Detection
      ↓
21 Landmark Extraction (per hand)
      ↓
Feature Vector Construction
      ↓
CSV Dataset Generation
      ↓
KNN Model Training
      ↓
Real-Time Prediction
```

---

## 🧩 System Components

- Data Collection Module  
- Landmark Extraction Module  
- CSV Dataset Generation  
- Machine Learning Model (KNN)  
- Real-Time Prediction System  

---

## 🛠 Technologies Used

- Python  
- OpenCV  
- MediaPipe  
- Scikit-learn  
- NumPy  
- Pandas  

---

## 🚀 Installation

```bash
git clone https://github.com/yourusername/turkish-sign-language-recognition.git
cd turkish-sign-language-recognition
pip install -r requirements.txt
```

---

## ▶️ Usage

Data collection:

```bash
python src/data_collection.py
```

Landmark extraction:

```bash
python src/extract_landmarks.py
```

Model training:

```bash
python src/train_knn.py
```

Real-time prediction:

```bash
python src/predict.py
```

---

## 📈 Future Improvements

- Integration of deep learning models (CNN)  
- Sequence-based recognition (LSTM)  
- Multi-user generalization  
- Two-hand gesture optimization  
- Web or mobile application deployment  
- Performance improvement through dataset augmentation  

---

## 🎓 Academic Context

This project was developed as a Graduation Project in Computer Engineering. The system architecture is intentionally designed to be extensible and suitable for future research and advanced model integration.

------------------------------------------------------------------------------------------------------------------------------------------------

🇹🇷 Türkçe

# Türk İşaret Dili Alfabe Tanıma Sistemi

## 📌 Proje Tanımı

Bu proje, MediaPipe el landmark tespiti ve makine öğrenmesi algoritmaları kullanılarak geliştirilmiş gerçek zamanlı bir Türk İşaret Dili (TİD) alfabe tanıma sistemidir.

Sistem, webcam aracılığıyla el hareketlerini algılar, her el için 3 boyutlu landmark koordinatlarını çıkarır ve oluşturulan özellik vektörlerini kullanarak K-En Yakın Komşu (KNN) algoritması ile harf sınıflandırması gerçekleştirir.

Proje, Bilgisayar Mühendisliği Bitirme Projesi kapsamında geliştirilmiş olup modüler ve ölçeklenebilir bir mimari ile tasarlanmıştır.

---

## 🎯 Projenin Amacı

- Gerçek zamanlı harf tanıma sistemi geliştirmek  
- Ham görüntü yerine landmark tabanlı özellik çıkarımı kullanmak  
- Yapılandırılmış ve genişletilebilir bir veri toplama altyapısı oluşturmak  
- Makine öğrenmesi ile el hareketlerini sınıflandırmak  

---

## 🧠 Neden Landmark Tabanlı Yaklaşım?

Bu projede doğrudan görüntü sınıflandırması yerine el landmark koordinatları kullanılmaktadır.

Bu yaklaşımın avantajları:

- Daha düşük hesaplama maliyeti  
- Daha az veri ihtiyacı  
- Daha hızlı eğitim süreci  
- Aydınlatma ve arka plan değişimlerinden daha az etkilenme  
- Daha yorumlanabilir ve düzenli özellik vektörleri  

Bu tasarım sayesinde sistem gerçek zamanlı çalışmaya uygundur ve ileride derin öğrenme modelleri ile entegre edilebilir.

---

## ⚙️ Sistem Mimarisi

```
Webcam Girdisi
      ↓
MediaPipe El Tespiti
      ↓
Her el için 21 Landmark Çıkarımı
      ↓
Özellik Vektörü Oluşturma
      ↓
CSV Veri Seti Üretimi
      ↓
KNN Model Eğitimi
      ↓
Gerçek Zamanlı Tahmin
```

---

## 🧩 Sistem Bileşenleri

- Veri Toplama Modülü  
- Landmark Çıkarma Modülü  
- Veri Seti Oluşturma (CSV)  
- Makine Öğrenmesi Modeli (KNN)  
- Gerçek Zamanlı Tahmin Sistemi  

---

## 🛠 Kullanılan Teknolojiler

- Python  
- OpenCV  
- MediaPipe  
- Scikit-learn  
- NumPy  
- Pandas  

---

## 🚀 Kurulum

```bash
git clone https://github.com/kullaniciadi/turkish-sign-language-recognition.git
cd turkish-sign-language-recognition
pip install -r requirements.txt
```

---

## ▶️ Kullanım

Veri toplama:

```bash
python src/data_collection.py
```

Landmark çıkarımı:

```bash
python src/extract_landmarks.py
```

Model eğitimi:

```bash
python src/train_knn.py
```

Gerçek zamanlı tahmin:

```bash
python src/predict.py
```

---

## 📈 Gelecek Çalışmalar

- Derin öğrenme tabanlı modellerin entegrasyonu (CNN)  
- Zaman serisi tabanlı tanıma (LSTM)  
- Çoklu kullanıcı genellemesi  
- İki el koordinasyon optimizasyonu  
- Web veya mobil uygulama geliştirme  
- Veri artırma teknikleri ile performans iyileştirme  

---

## 🎓 Akademik Bağlam

Bu proje, Bilgisayar Mühendisliği Bitirme Projesi kapsamında geliştirilmiştir. Sistem mimarisi, ileride genişletilebilecek ve daha gelişmiş makine öğrenmesi / derin öğrenme modelleri ile entegre edilebilecek şekilde tasarlanmıştır.
