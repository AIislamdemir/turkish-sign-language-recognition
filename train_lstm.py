"""
train_lstm.py — TİD Kelime Tanıma LSTM Eğitimi
================================================
Kullanım:
    python train_lstm.py
Çıktı:
    word_model_lstm.h5
    word_label_classes.npy
"""

import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import classification_report
from tensorflow.keras.utils  import to_categorical
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# LSTM modelini import et
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "models"))
from LSTM import build_model

# ─────────────────────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────────────────────
DATA_PATH          = "words_datasets/words"
MODEL_SAVE_PATH    = "word_model_lstm.h5"
LABELS_SAVE_PATH   = "word_label_classes.npy"
FRAMES_PER_SEQ     = 30
FEATURE_DIM        = 126   # çift el: 63×2
EPOCHS             = 200
BATCH_SIZE         = 32

# ─────────────────────────────────────────────────────────────
# 1. VERİ YÜKLE
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  TİD LSTM — Kelime Tanıma Eğitimi")
print("=" * 55)
print("\n[1/5] Veri yükleniyor...")

X, y = [], []

words = sorted(os.listdir(DATA_PATH))
words = [w for w in words if os.path.isdir(os.path.join(DATA_PATH, w))]

print(f"  Bulunan kelimeler ({len(words)}): {words}\n")

for word in words:
    word_path = os.path.join(DATA_PATH, word)
    files     = [f for f in os.listdir(word_path) if f.endswith(".npy")]

    if len(files) == 0:
        print(f"  ⚠️  [{word}] klasörü boş, atlanıyor.")
        continue

    for fname in files:
        seq = np.load(os.path.join(word_path, fname))

        # Şekil kontrolü ve düzeltme
        if seq.ndim == 1:
            # (126,) → tek frame, skip
            continue

        if seq.shape != (FRAMES_PER_SEQ, FEATURE_DIM):
            # Eksik frame varsa sıfırla doldur, fazlaysa kes
            if seq.shape[1] != FEATURE_DIM:
                continue  # özellik boyutu uyumsuz
            if seq.shape[0] < FRAMES_PER_SEQ:
                pad = np.zeros((FRAMES_PER_SEQ - seq.shape[0], FEATURE_DIM), dtype=np.float32)
                seq = np.vstack([seq, pad])
            else:
                seq = seq[:FRAMES_PER_SEQ]

        X.append(seq)
        y.append(word)

    print(f"  [{word:>15}]  {len(files)} örnek yüklendi")

X = np.array(X, dtype=np.float32)
y = np.array(y)

print(f"\n  Toplam örnek  : {len(X)}")
print(f"  Veri şekli    : {X.shape}  (örnek, frame, özellik)")
print(f"  Sınıf sayısı  : {len(np.unique(y))}")

if len(X) == 0:
    print("\n[HATA] Hiç veri yüklenemedi! Önce collect_word_sequences.py çalıştırın.")
    exit(1)

# ─────────────────────────────────────────────────────────────
# 2. LABEL ENCODING
# ─────────────────────────────────────────────────────────────
print("\n[2/5] Etiketler işleniyor...")
le        = LabelEncoder()
y_encoded = le.fit_transform(y)
n_classes = len(le.classes_)
y_cat     = to_categorical(y_encoded, n_classes)

np.save(LABELS_SAVE_PATH, le.classes_)
print(f"  Sınıflar : {list(le.classes_)}")
print(f"  {LABELS_SAVE_PATH} kaydedildi.")

# ─────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
print("\n[3/5] Veri bölünüyor (%85 eğitim / %15 test)...")
X_train, X_test, y_train, y_test, y_train_enc, y_test_enc = train_test_split(
    X, y_cat, y_encoded,
    test_size=0.15,
    random_state=42,
    stratify=y_encoded
)
print(f"  Eğitim : {len(X_train)}")
print(f"  Test   : {len(X_test)}")

# ─────────────────────────────────────────────────────────────
# 4. MODEL EĞİT
# ─────────────────────────────────────────────────────────────
print("\n[4/5] LSTM modeli eğitiliyor...")
model = build_model(
    sequence_length=FRAMES_PER_SEQ,
    feature_dim=FEATURE_DIM,
    num_classes=n_classes
)
model.summary()

callbacks = [
    EarlyStopping(
        monitor='val_accuracy', patience=25,
        restore_best_weights=True, verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss', factor=0.5,
        patience=10, verbose=1
    )
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────────────────────────
# 5. KAYDET & SONUÇ
# ─────────────────────────────────────────────────────────────
print("\n[5/5] Model kaydediliyor...")
model.save(MODEL_SAVE_PATH)

loss, acc = model.evaluate(X_test, y_test, verbose=0)
y_pred    = np.argmax(model.predict(X_test, verbose=0), axis=1)

print(f"\n  ✓ {MODEL_SAVE_PATH} kaydedildi!")
print(f"  Test Accuracy : {acc:.4f}  ({acc*100:.1f}%)")
print(f"  Test Loss     : {loss:.4f}")
print("\n  Sınıf bazlı rapor:")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

# Grafik
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('LSTM — Accuracy')
plt.xlabel('Epoch')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('LSTM — Loss')
plt.xlabel('Epoch')
plt.legend()
plt.tight_layout()
plt.savefig("outputs/lstm_training_history.png", dpi=150)
plt.close()
print("  outputs/lstm_training_history.png kaydedildi.")

print("\n" + "=" * 55)
print("  EĞİTİM TAMAMLANDI")
print("=" * 55)
print(f"  Model     : {MODEL_SAVE_PATH}")
print(f"  Etiketler : {LABELS_SAVE_PATH}")
print(f"  Doğruluk  : {acc*100:.1f}%")
print("\n  ✅ Sıradaki adım: app.py'ye kelime tanıma entegre edilecek!")