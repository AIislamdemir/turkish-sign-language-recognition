# HandsWith-AI-Connect
# Turkish Sign Language Recognition System | Türk İşaret Dili Tanıma Sistemi

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python"/>
  <img src="https://img.shields.io/badge/Flask-Web%20App-black?style=for-the-badge&logo=flask"/>
  <img src="https://img.shields.io/badge/MediaPipe-Hand%20Detection-green?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/scikit--learn-RandomForest-orange?style=for-the-badge"/>
  <img src="https://img.shields.io/badge/TensorFlow-LSTM-red?style=for-the-badge&logo=tensorflow"/>
  <img src="https://img.shields.io/badge/Accuracy-99.9%25-brightgreen?style=for-the-badge"/>
</p>

---

# 🇬🇧 English

## 📌 Project Description

**HandsWith-AI-Connect** is a real-time Turkish Sign Language (TSL) recognition system developed as a Computer Engineering Graduation Project at Batman University.

The system captures hand gestures through a webcam, extracts normalized 3D landmark coordinates from both hands using MediaPipe, and classifies them using machine learning and deep learning models. Recognized letters and words are displayed instantly on a professional Flask-based web interface.

The project covers the full pipeline from data collection to real-time inference — including letter recognition, word recognition, and a production-grade web UI.

---

## 🎯 Project Objectives

- Develop a real-time Turkish Sign Language recognition system covering both letters and words
- Use wrist-normalized, dual-hand landmark vectors instead of raw image classification
- Build a stable, freeze-free real-time inference pipeline using a 3-thread architecture
- Integrate an LSTM-based word recognition model alongside the letter classifier
- Deliver a clean, professional Flask web interface suitable for real-world use

---

## 🧠 Why Landmark-Based Approach?

Instead of training on raw images, this project uses structured 3D hand landmark coordinates extracted by MediaPipe. Each hand produces 21 landmarks × 3 axes = 63 features. With dual-hand support, the full feature vector is **126 features**.

**Normalization Pipeline:**
```
Raw Coordinates
      ↓
Wrist-Relative Shift  (coords -= coords[0])
      ↓
Scale Normalization   (coords /= max(abs(coords)))
      ↓
Normalized vector in [-1.0, +1.0]
```

This normalization makes the system invariant to hand size, camera distance, and hand position — enabling consistent predictions across different users.

---

## ⚙️ System Architecture

```
Webcam Input
      ↓
┌─────────────────────────────────┐
│  Thread-1: Capture Thread       │  ← Reads frames from camera only
│  OpenCV + cv2.flip              │
└────────────────┬────────────────┘
                 ↓ (Queue)
┌─────────────────────────────────┐
│  Thread-2: Inference Thread     │
│  MediaPipe → 126-feature vector │
│  ├── RandomForest → Letter      │  ← 99.9% accuracy
│  └── LSTM (30-frame window)     │  ← 100.0% accuracy
│       → Word                    │
└────────────────┬────────────────┘
                 ↓ (shared JPEG)
┌─────────────────────────────────┐
│  Thread-3: Flask Web Server     │  ← Streams video + serves API
│  MJPEG stream + REST endpoints  │
└─────────────────────────────────┘
```

---

## 📊 Models & Results

### Letter Recognition — RandomForest

| Metric | Value |
|--------|-------|
| Classes | 23 Turkish alphabet letters |
| Feature Vector | 126 (dual-hand, normalized) |
| Training Samples | 5,919 images |
| Test Accuracy | **99.9%** |
| Inference Time | < 1ms per frame |
| GIL Impact | None |

### Word Recognition — LSTM

| Metric | Value |
|--------|-------|
| Classes | 29 daily-use words |
| Feature Vector | 126 features × 30 frames |
| Training Samples | 1,740 sequences |
| Test Accuracy | **100.0%** |
| Architecture | 3-layer LSTM (256→128→64) |

### LSTM Architecture
```
Input: (30, 126)
      ↓
LSTM(256, return_sequences=True)  + L2 + BatchNorm
      ↓
LSTM(128, return_sequences=True)  + L2 + BatchNorm
      ↓
LSTM(64,  return_sequences=False) + L2 + BatchNorm
      ↓
Dense(128, ReLU) + Dropout(0.3)
      ↓
Dense(64,  ReLU) + Dropout(0.2)
      ↓
Dense(29, Softmax)
```

**Optimizations:** Gradient Clipping (clipnorm=1.0) · Adam Optimizer · EarlyStopping · ReduceLROnPlateau

---

## 🗂️ Dataset

### Letter Dataset
- 23 Turkish Sign Language letters
- ~200–250 images per class
- Total: **5,919 labeled images**
- Format: 126-feature normalized CSV

### Word Dataset
- 29 daily-use words: *anne, baba, ben, biz, ev, evet, gel, git, güle güle, hayır, iyi, iş, kim, kötü, lütfen, merhaba, nasıl, ne, ne zaman, nerde, o, okul, onlar, para, sen, siz, tamam, telefon, teşekkürler*
- 60 sequences per word × 30 frames per sequence
- Total: **1,740 labeled sequences**
- Format: `.npy` files (shape: 30 × 126)

---

## 🖥️ Web Interface Features

- **Live camera stream** via MJPEG in the browser
- **Letter recognition card** with confidence bar and hold-progress ring
- **Word recognition card** — LSTM-powered, displayed in amber
- **Auto letter combining** — letter added automatically after 0.9s hold
- **Auto word spacing** — word saved after 2s without hand detection
- **Manual controls** — keyboard shortcuts (E: add letter, W: add word, Space: save, Backspace: delete)
- **Dark theme** professional UI
- **Toggle** between automatic and manual mode

---

## 🛠 Technologies Used

| Tool | Purpose |
|------|---------|
| Python 3.10 | Core language |
| OpenCV | Camera capture & image processing |
| MediaPipe | Hand landmark detection (21 points) |
| scikit-learn | RandomForest letter classifier |
| TensorFlow / Keras | LSTM word recognition model |
| Flask | Real-time web interface |
| NumPy / Pandas | Data processing |

---

## 📁 Project Structure

```
Graduation_Project_thesis/
├── dataset/                    # Letter images (23 classes)
├── words_datasets/words/       # Word sequences (.npy, 29 classes)
├── handswith_ai_connect/
│   ├── extract_landmarks_fixed.py   # Landmark extraction + normalization
│   └── train_model.py               # RF + Keras training
├── models/
│   ├── LSTM.PY                      # LSTM model architecture
│   └── train_knn.py                 # Baseline KNN (reference)
├── src/
│   ├── collect_word_sequences.py    # Word data collection
│   └── realtime_letter_test.py      # OpenCV standalone test
├── UI/
│   └── app.py                       # Flask web application
├── train_lstm.py                    # LSTM training script
├── landmarks.csv                    # Extracted landmark dataset
├── letter_model_rf.pkl              # RandomForest model
├── word_model_lstm.h5               # LSTM word model
├── label_classes.npy                # Letter class labels
├── word_label_classes.npy           # Word class labels
└── requirements.txt
```

---

## 🚀 Installation

```bash
git clone https://github.com/AIislamdemir/turkish-sign-language-recognition.git
cd turkish-sign-language-recognition
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1. Extract landmarks from dataset
```bash
python handswith_ai_connect/extract_landmarks_fixed.py
```

### 2. Train letter recognition models (RF + Keras)
```bash
python handswith_ai_connect/train_model.py
```

### 3. Collect word sequences
```bash
python src/collect_word_sequences.py
```

### 4. Train LSTM word model
```bash
python train_lstm.py
```

### 5. Launch the web interface
```bash
python UI/app.py
```
Then open `http://127.0.0.1:5000` in your browser.

---

## 🔧 Key Technical Decisions

### Python GIL Problem → Solution
TensorFlow's `model.predict()` holds Python's GIL for hundreds of milliseconds per call, blocking the camera stream and causing UI freezes. This was solved by switching to scikit-learn's RandomForest, whose `predict_proba()` completes in under 1ms and releases the GIL almost immediately — enabling truly parallel threading.

### Dual-Hand Support
MediaPipe was configured with `max_num_hands=2`. Left hand features occupy indices 0–62, right hand features occupy 63–125. Missing hands are zero-padded, keeping the vector length constant at 126.

### Wrist-Relative Normalization
All coordinates are shifted relative to the wrist (landmark 0) and then scaled by the maximum absolute value, producing values in [-1.0, +1.0]. This ensures predictions are invariant to hand position, size, and camera distance.

---

## 📈 Future Work

- Extended word and sentence dataset for broader vocabulary
- Text-to-Speech (TTS) integration for audio output
- Mobile application (Android / iOS)
- Context-aware language model integration for improved accuracy
- Multi-user stress testing and generalization improvements
- Data augmentation for increased model robustness

---

## 🎓 Academic Context

**Institution:** Batman University, Faculty of Engineering, Department of Computer Engineering

**Type:** Graduation Project (Bitirme Projesi)

**Advisor:** Assoc. Prof. Dr. Yılmaz Kaya

**Developer:** İslam Demir

**Year:** 2025

---

---

# 🇹🇷 Türkçe

## 📌 Proje Tanımı

**HandsWith-AI-Connect**, Batman Üniversitesi Bilgisayar Mühendisliği Bölümü bitirme projesi kapsamında geliştirilmiş gerçek zamanlı bir Türk İşaret Dili (TİD) tanıma sistemidir.

Sistem; webcam aracılığıyla el hareketlerini algılar, MediaPipe kullanarak her iki elden normalize edilmiş 3B landmark koordinatları çıkarır ve bunları makine öğrenmesi ile derin öğrenme modelleri aracılığıyla sınıflandırır. Tanınan harfler ve kelimeler, profesyonel Flask tabanlı web arayüzünde anlık olarak gösterilir.

Proje; veri toplama, landmark çıkarımı, harf tanıma, kelime tanıma ve gerçek zamanlı web arayüzünü kapsayan uçtan uca tam bir pipeline sunar.

---

## 🎯 Projenin Amacı

- Türk İşaret Dili'ni hem harf hem kelime düzeyinde gerçek zamanlı tanıyan bir sistem geliştirmek
- Ham görüntü yerine bilek-normalize edilmiş çift el landmark vektörleri kullanmak
- 3 thread'li mimari ile donmasız, kararlı gerçek zamanlı tahmin altyapısı oluşturmak
- Harf sınıflandırıcısının yanına LSTM tabanlı kelime tanıma modeli entegre etmek
- Gerçek hayatta kullanılabilir, profesyonel bir Flask web arayüzü sunmak

---

## 🧠 Neden Landmark Tabanlı Yaklaşım?

Bu projede ham görüntü yerine MediaPipe'ın çıkardığı 3B el landmark koordinatları kullanılmaktadır. Her el 21 landmark × 3 eksen = 63 özellik üretir. Çift el desteğiyle toplam özellik vektörü **126** boyutuna ulaşır.

**Normalizasyon Süreci:**
```
Ham Koordinatlar
      ↓
Wrist Referanslı Kaydırma  (coords -= coords[0])
      ↓
Scale Normalizasyon         (coords /= max(abs(coords)))
      ↓
[-1.0, +1.0] aralığında normalize vektör
```

Bu normalizasyon; tahminleri el boyutundan, kamera mesafesinden ve el konumundan bağımsız hale getirerek farklı kullanıcılarda tutarlı sonuçlar elde edilmesini sağlar.

---

## ⚙️ Sistem Mimarisi

```
Webcam Girdisi
      ↓
┌─────────────────────────────────┐
│  Thread-1: Capture Thread       │  ← Yalnızca kamera okur
│  OpenCV + cv2.flip              │
└────────────────┬────────────────┘
                 ↓ (Queue)
┌─────────────────────────────────┐
│  Thread-2: Inference Thread     │
│  MediaPipe → 126 özellik vektörü│
│  ├── RandomForest → Harf        │  ← %99.9 doğruluk
│  └── LSTM (30 frame pencere)    │  ← %100.0 doğruluk
│       → Kelime                  │
└────────────────┬────────────────┘
                 ↓ (paylaşılan JPEG)
┌─────────────────────────────────┐
│  Thread-3: Flask Web Sunucusu   │  ← Video stream + API
│  MJPEG stream + REST endpoint   │
└─────────────────────────────────┘
```

---

## 📊 Modeller ve Sonuçlar

### Harf Tanıma — RandomForest

| Ölçüt | Değer |
|-------|-------|
| Sınıf Sayısı | 23 Türk alfabesi harfi |
| Özellik Vektörü | 126 (çift el, normalize) |
| Eğitim Örneği | 5.919 görüntü |
| Test Doğruluğu | **%99.9** |
| Tahmin Süresi | < 1ms / frame |

### Kelime Tanıma — LSTM

| Ölçüt | Değer |
|-------|-------|
| Sınıf Sayısı | 29 günlük kelime |
| Özellik Vektörü | 126 özellik × 30 frame |
| Eğitim Örneği | 1.740 sekans |
| Test Doğruluğu | **%100.0** |
| Mimari | 3 katmanlı LSTM (256→128→64) |

---

## 🗂️ Veri Seti

### Harf Veri Seti
- 23 TİD harfi, sınıf başına ~200-250 görüntü
- Toplam: **5.919 etiketli görüntü**

### Kelime Veri Seti
- 29 günlük kelime: *anne, baba, ben, biz, ev, evet, gel, git, güle güle, hayır, iyi, iş, kim, kötü, lütfen, merhaba, nasıl, ne, ne zaman, nerde, o, okul, onlar, para, sen, siz, tamam, telefon, teşekkürler*
- Kelime başına 60 sekans × sekans başına 30 frame
- Toplam: **1.740 etiketli sekans**

---

## 🖥️ Web Arayüzü Özellikleri

- Tarayıcıda **canlı kamera akışı** (MJPEG)
- Güven çubuğu ve halka geri sayım göstergeli **harf tanıma kartı**
- LSTM destekli **kelime tanıma kartı**
- **Otomatik harf birleştirme** — 0.9 saniye sabit tutulunca harf eklenir
- **Otomatik kelime kaydetme** — 2 saniye el görünmeyince kelime kaydedilir
- **Klavye kısayolları** (E: harf ekle, W: kelime ekle, Space: kaydet, Backspace: sil)
- **Karanlık tema** profesyonel arayüz
- **Otomatik / Manuel mod** geçişi

---

## 🛠 Kullanılan Teknolojiler

| Araç | Kullanım Amacı |
|------|----------------|
| Python 3.10 | Ana programlama dili |
| OpenCV | Kamera görüntüsü işleme |
| MediaPipe | El landmark tespiti (21 nokta) |
| scikit-learn | RandomForest harf sınıflandırıcısı |
| TensorFlow / Keras | LSTM kelime tanıma modeli |
| Flask | Gerçek zamanlı web arayüzü |
| NumPy / Pandas | Veri işleme |

---

## 🚀 Kurulum

```bash
git clone https://github.com/AIislamdemir/turkish-sign-language-recognition.git
cd turkish-sign-language-recognition
pip install -r requirements.txt
```

---

## ▶️ Kullanım

### 1. Landmark çıkarımı
```bash
python handswith_ai_connect/extract_landmarks_fixed.py
```

### 2. Harf tanıma modellerini eğit (RF + Keras)
```bash
python handswith_ai_connect/train_model.py
```

### 3. Kelime verisi topla
```bash
python src/collect_word_sequences.py
```

### 4. LSTM modelini eğit
```bash
python train_lstm.py
```

### 5. Web arayüzünü başlat
```bash
python UI/app.py
```
Tarayıcıda `http://127.0.0.1:5000` adresini açın.

---

## 🔧 Temel Teknik Kararlar

### Python GIL Sorunu → Çözüm
TensorFlow'un `model.predict()` fonksiyonu her çağrıda Python'un GIL kilidini yüzlerce milisaniye boyunca tutmakta ve kamera akışını bloke etmektedir. Bu sorun, scikit-learn'ün RandomForest sınıflandırıcısına geçilerek çözülmüştür. RandomForest'ın `predict_proba()` çağrısı 1ms'nin altında tamamlanır ve GIL'i neredeyse hiç tutmaz.

### Çift El Desteği
MediaPipe `max_num_hands=2` olarak yapılandırılmıştır. Sol el özellikleri 0-62, sağ el özellikleri 63-125 indekslerine yerleştirilir. Eksik el sıfır vektörüyle doldurulur (zero-padding), vektör uzunluğu her zaman 126 kalır.

### Wrist Referanslı Normalizasyon
Tüm koordinatlar bilek noktasına (landmark 0) göre kaydırılır, ardından maksimum mutlak değere bölünür. Bu işlem tahminleri el konumundan, boyutundan ve kamera mesafesinden bağımsız kılar.

---

## 📈 Gelecek Çalışmalar

- Daha geniş kelime ve cümle veri seti oluşturulması
- Text-to-Speech (TTS) entegrasyonu ile sesli çıktı
- Mobil uygulama geliştirme (Android / iOS)
- Bağlam farkındalıklı dil modeli entegrasyonu
- Çok kullanıcılı stres testleri ve genelleme iyileştirmeleri
- Veri artırma teknikleriyle model dayanıklılığının artırılması

---

## 🎓 Akademik Bilgiler

**Kurum:** Batman Üniversitesi, Mühendislik Fakültesi, Bilgisayar Mühendisliği Bölümü

**Tür:** Lisans Bitirme Projesi

**Danışman:** Doç. Dr. Yılmaz Kaya

**Geliştirici:** İslam Demir

**Yıl:** 2025