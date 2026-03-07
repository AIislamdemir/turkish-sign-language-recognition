"""
=============================================================
train_model.py  (v2 — Çift El / 126 Özellik)
=============================================================
DEĞİŞİKLİKLER (v1 → v2):
  • input_shape=(63,) → (126,)  — çift el vektörü
  • RandomForest modeli de eğitilip letter_model_rf.pkl kaydedilir
    (app_production_v5.py bu dosyayı kullanır)
  • Keras modeli letter_model.h5 olarak kaydedilmeye devam eder
  • label_classes.npy her iki model için de geçerli

Kullanım:
    python train_model.py
=============================================================
"""

import os
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split
from sklearn.preprocessing    import LabelEncoder
from sklearn.metrics          import accuracy_score, classification_report

import tensorflow as tf
from tensorflow.keras.models     import Sequential
from tensorflow.keras.layers     import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks  import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils      import to_categorical

# ─────────────────────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────────────────────
CSV_PATH      = "landmarks.csv"
RF_MODEL_PATH = "letter_model_rf.pkl"   # app_production_v5.py bunu kullanır
H5_MODEL_PATH = "letter_model.h5"       # Keras modeli (isteğe bağlı)
EPOCHS        = 150
BATCH_SIZE    = 32

# ─────────────────────────────────────────────────────────────
# 1. VERİ YÜKLE
# ─────────────────────────────────────────────────────────────
print("=" * 55)
print("  TİD Eğitim — Çift El Modu (126 özellik)")
print("=" * 55)

print("\n[1/6] Veri yükleniyor...")
df = pd.read_csv(CSV_PATH)

feature_cols = [c for c in df.columns if c != "label"]
X     = df[feature_cols].values.astype(np.float32)
y_raw = df["label"].values

n_features = X.shape[1]
print(f"  Toplam örnek  : {len(df)}")
print(f"  Özellik sayısı: {n_features}")
print(f"  Sınıf sayısı  : {len(np.unique(y_raw))}")
print(f"  Sınıflar      : {sorted(np.unique(y_raw))}")

if n_features == 63:
    print("\n  ⚠️  UYARI: CSV hâlâ 63 özellik içeriyor.")
    print("  Önce extract_landmarks_fixed.py çalıştırarak landmarks.csv'yi yenileyin.")
    print("  Ardından bu scripti tekrar çalıştırın.")
    exit(1)

if n_features != 126:
    print(f"\n  ⚠️  Beklenmeyen özellik sayısı: {n_features} (126 olmalı)")
    exit(1)

print(f"  ✓ Çift el formatı doğrulandı (126 özellik)")

# ─────────────────────────────────────────────────────────────
# 2. LABEL ENCODING
# ─────────────────────────────────────────────────────────────
print("\n[2/6] Etiketler işleniyor...")
le        = LabelEncoder()
y_encoded = le.fit_transform(y_raw)
n_classes = len(le.classes_)
y_cat     = to_categorical(y_encoded, n_classes)

print(f"  Sınıf sırası: {list(le.classes_)}")
np.save("label_classes.npy", le.classes_)
print("  label_classes.npy kaydedildi.")

# ─────────────────────────────────────────────────────────────
# 3. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────
print("\n[3/6] Veri bölünüyor (%85 eğitim / %15 test)...")
X_train, X_test, y_train_cat, y_test_cat, y_train_enc, y_test_enc = train_test_split(
    X, y_cat, y_encoded,
    test_size=0.15,
    random_state=42,
    stratify=y_encoded
)
print(f"  Eğitim : {len(X_train)}")
print(f"  Test   : {len(X_test)}")

# ─────────────────────────────────────────────────────────────
# 4. RANDOM FOREST  (app_production_v5.py için)
# ─────────────────────────────────────────────────────────────
print("\n[4/6] RandomForest eğitiliyor...")
print("  (Bu birkaç dakika sürebilir...)")

rf = RandomForestClassifier(
    n_estimators=200,        # 200 ağaç — iyi denge hız/doğruluk
    max_depth=None,          # tam derinlik
    min_samples_split=2,
    min_samples_leaf=1,
    n_jobs=-1,               # tüm CPU çekirdeklerini kullan
    random_state=42,
    verbose=0
)
rf.fit(X_train, y_train_enc)

rf_pred = rf.predict(X_test)
rf_acc  = accuracy_score(y_test_enc, rf_pred)
print(f"\n  ✓ RandomForest Test Accuracy : {rf_acc:.4f}  ({rf_acc*100:.1f}%)")
print("\n  Sınıf bazlı rapor:")
print(classification_report(y_test_enc, rf_pred, target_names=le.classes_))

# RF modelini kaydet — app_production_v5.py bu formatı bekliyor
with open(RF_MODEL_PATH, "wb") as f:
    pickle.dump({"model": rf, "classes": le.classes_}, f)
print(f"  ✓ {RF_MODEL_PATH} kaydedildi!")

# ─────────────────────────────────────────────────────────────
# 5. KERAS  (isteğe bağlı — letter_model.h5)
# ─────────────────────────────────────────────────────────────
print("\n[5/6] Keras modeli eğitiliyor...")

model = Sequential([
    Dense(512, activation='relu', input_shape=(126,)),   # ★ 126
    BatchNormalization(),
    Dropout(0.3),

    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),

    Dense(128, activation='relu'),
    BatchNormalization(),
    Dropout(0.2),

    Dense(64,  activation='relu'),
    Dropout(0.2),

    Dense(n_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

callbacks = [
    EarlyStopping(monitor='val_accuracy', patience=20,
                  restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                      patience=10, verbose=1)
]

history = model.fit(
    X_train, y_train_cat,
    validation_data=(X_test, y_test_cat),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=1
)

model.save(H5_MODEL_PATH)
keras_loss, keras_acc = model.evaluate(X_test, y_test_cat, verbose=0)
print(f"\n  ✓ {H5_MODEL_PATH} kaydedildi!")
print(f"  Keras Test Accuracy : {keras_acc:.4f}  ({keras_acc*100:.1f}%)")

# ─────────────────────────────────────────────────────────────
# 6. GRAFİKLER
# ─────────────────────────────────────────────────────────────
print("\n[6/6] Grafikler kaydediliyor...")
os.makedirs("outputs", exist_ok=True)

plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'],     label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Keras — Accuracy')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'],     label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Keras — Loss')
plt.xlabel('Epoch')
plt.legend()

plt.tight_layout()
plt.savefig("outputs/training_history.png", dpi=150)
plt.close()
print("  outputs/training_history.png kaydedildi.")

# ─────────────────────────────────────────────────────────────
# ÖZET
# ─────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  EĞİTİM TAMAMLANDI")
print("=" * 55)
print(f"  RandomForest doğruluğu : {rf_acc*100:.1f}%  → {RF_MODEL_PATH}")
print(f"  Keras doğruluğu        : {keras_acc*100:.1f}%  → {H5_MODEL_PATH}")
print(f"  Özellik boyutu         : 126 (çift el)")
print(f"  Sınıf sayısı           : {n_classes}")
print("\n  ✅ app_production_v5.py artık çalıştırılabilir!")