
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# ─────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────
CSV_PATH    = "landmarks.csv"
MODEL_PATH  = "letter_model.h5"
EPOCHS      = 150
BATCH_SIZE  = 32

# ─────────────────────────────────────────
# 1. VERİ YÜKLE
# ─────────────────────────────────────────
print("[1/5] Veri yükleniyor...")
df = pd.read_csv(CSV_PATH)

feature_cols = [c for c in df.columns if c != "label"]
X = df[feature_cols].values.astype(np.float32)
y_raw = df["label"].values

print(f"  Toplam örnek  : {len(df)}")
print(f"  Sınıf sayısı  : {len(np.unique(y_raw))}")
print(f"  Sınıflar      : {sorted(np.unique(y_raw))}")

# ─────────────────────────────────────────
# 2. LABEL ENCODING
# ─────────────────────────────────────────
print("\n[2/5] Etiketler işleniyor...")
le = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
num_classes = len(le.classes_)
y_cat = to_categorical(y_encoded, num_classes)

print(f"  Sınıf sırası : {list(le.classes_)}")

# LabelEncoder'ı kaydet (realtime'da aynı sıra lazım)
np.save("label_classes.npy", le.classes_)
print("  label_classes.npy kaydedildi.")

# ─────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────
print("\n[3/5] Veri bölünüyor...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y_cat,
    test_size=0.15,
    random_state=42,
    stratify=y_encoded
)
print(f"  Eğitim : {len(X_train)}")
print(f"  Test   : {len(X_test)}")

# ─────────────────────────────────────────
# 4. MODEL
# ─────────────────────────────────────────
print("\n[4/5] Model oluşturuluyor ve eğitiliyor...")
model = Sequential([
    Dense(256, activation='relu', input_shape=(63,)),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(64, activation='relu'),
    Dropout(0.2),

    Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, verbose=1)
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

# ─────────────────────────────────────────
# 5. KAYDET & SONUÇ
# ─────────────────────────────────────────
print("\n[5/5] Model kaydediliyor...")
model.save(MODEL_PATH)
print(f"  ✓ {MODEL_PATH} kaydedildi!")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\n  Test Accuracy : {test_acc:.4f} ({test_acc*100:.1f}%)")
print(f"  Test Loss     : {test_loss:.4f}")

# Grafik kaydet
os.makedirs("outputs", exist_ok=True)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss')
plt.legend()
plt.tight_layout()
plt.savefig("outputs/training_history.png", dpi=150)
plt.close()
print("  outputs/training_history.png kaydedildi.")

print("\n✅ Eğitim tamamlandı! Şimdi realtime_letter_test.py çalıştırabilirsiniz.")