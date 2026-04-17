"""
=============================================================
extract_landmarks_fixed.py  (v2 — Çift El Desteği)
=============================================================
Tüm dataset klasöründen landmarks.csv üretir.

DEĞİŞİKLİKLER (v1 → v2):
  • max_num_hands=2  → her iki el de algılanır
  • Sol el  → sütunlar 0–62   (lx_0..lz_20)
  • Sağ el  → sütunlar 63–125 (rx_0..rz_20)
  • Eksik el → 0.0 ile doldurulur (zero-padding)
  • Toplam özellik: 126 (21×3×2)

Bu şekilde üretilen landmarks.csv ile eğitilen model
app_production_v5.py ile tam uyumludur.

Kullanım:
    python extract_landmarks_fixed.py
Proje kök dizininde çalıştırın (handswith_ai_connect/ içinde).
=============================================================
"""

import os
import csv
import cv2
import numpy as np
import mediapipe as mp

BASE_DIR  = "dataset"
CSV_PATH  = "landmarks.csv"

mp_hands  = mp.solutions.hands

# 21 nokta × 3 eksen = 63 float, her el için
_SINGLE   = 63
_ZEROS    = [0.0] * _SINGLE   # eksik el yerine kullanılacak dolgu


def _hand_to_list(hand_landmarks):
    """Bir elin landmark'larını normalize edilmiş 63 floatlık listeye dönüştür.

    Normalizasyon:
      1. Wrist (bilek, nokta 0) referans alınarak tüm koordinatlar kaydırılır.
      2. Maksimum mutlak değere bölünerek ölçek normalize edilir.
    Bu sayede model farklı el boyutlarına ve kamera mesafelerine dayanıklı olur.
    """
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
                      dtype=np.float32)
    # 1. Wrist referanslı kaydırma
    coords -= coords[0]
    # 2. Scale normalizasyon
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val
    return coords.flatten().tolist()


def process_image(image_path, class_label, hands):
    """
    Görüntüden çift el feature vektörü çıkar.

    Dönüş: [sol_63_float + sag_63_float + label]  (127 elemanlı liste)
           Hiç el bulunamazsa None döner.
    """
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result  = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None   # hiç el yok → bu görüntüyü atla

    left_data  = None
    right_data = None

    for hlm, handedness in zip(result.multi_hand_landmarks,
                                result.multi_handedness):
        label = handedness.classification[0].label  # "Left" veya "Right"
        if label == "Left":
            left_data  = _hand_to_list(hlm)
        else:
            right_data = _hand_to_list(hlm)

    # En az bir el bulundu — diğeri yoksa sıfır doldur
    left_data  = left_data  if left_data  is not None else list(_ZEROS)
    right_data = right_data if right_data is not None else list(_ZEROS)

    row = left_data + right_data   # 126 float
    row.append(class_label)         # 127. eleman = sınıf etiketi
    return row


def build_headers():
    """126 özellik + label için sütun başlıklarını üret."""
    headers = []
    for prefix in ("l", "r"):          # l = sol, r = sağ
        for i in range(21):
            headers.extend([f"{prefix}x_{i}", f"{prefix}y_{i}", f"{prefix}z_{i}"])
    headers.append("label")
    return headers


def main():
    if not os.path.exists(BASE_DIR):
        print(f"[HATA] '{BASE_DIR}' klasörü bulunamadı!")
        print("Bu scripti projenin kök dizininde çalıştırdığınızdan emin olun.")
        return

    headers    = build_headers()
    total_ok   = 0
    total_skip = 0

    with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,            # ★ ÇİFT EL
            min_detection_confidence=0.5
        ) as hands:

            classes = sorted(os.listdir(BASE_DIR))
            print(f"[BİLGİ] {len(classes)} sınıf bulundu: {classes}")
            print(f"[BİLGİ] Özellik vektörü: 126 (sol 63 + sağ 63)\n")

            for class_name in classes:
                class_path = os.path.join(BASE_DIR, class_name)
                if not os.path.isdir(class_path):
                    continue

                class_ok   = 0
                class_skip = 0

                for file_name in os.listdir(class_path):
                    if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                        continue

                    img_path = os.path.join(class_path, file_name)
                    row      = process_image(img_path, class_name, hands)

                    if row is not None:
                        writer.writerow(row)
                        class_ok += 1
                    else:
                        class_skip += 1

                total_ok   += class_ok
                total_skip += class_skip
                print(f"  [{class_name:>3}]  ✓ {class_ok} kayıt   ✗ {class_skip} atlandı")

    print(f"\n{'─'*50}")
    print(f"[TAMAMLANDI] {CSV_PATH} oluşturuldu.")
    print(f"  Toplam kayıt   : {total_ok}")
    print(f"  Toplam atlanan : {total_skip}")
    print(f"  Özellik sayısı : 126 (çift el)")
    print(f"\nSıradaki adım → python train_model.py")


if __name__ == "__main__":
    main()