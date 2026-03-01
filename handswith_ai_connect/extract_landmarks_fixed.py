"""
=============================================================
extract_landmarks_fixed.py
Tüm dataset klasöründen landmarks.csv üretir.
=============================================================
Kullanım:
    python extract_landmarks_fixed.py

Proje kök dizininde çalıştırın (handswith_ai_connect/ içinde).
"""

import os
import csv
import cv2
import mediapipe as mp

BASE_DIR = "dataset"
CSV_PATH = "landmarks.csv"
mp_hands = mp.solutions.hands


def process_image(image_path, class_label, hands):
    img = cv2.imread(image_path)
    if img is None:
        return None

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if not result.multi_hand_landmarks:
        return None

    hand_landmarks = result.multi_hand_landmarks[0]

    row = []
    for lm in hand_landmarks.landmark:
        row.extend([lm.x, lm.y, lm.z])  # 21 nokta x 3 = 63 özellik

    row.append(class_label)
    return row


def main():
    if not os.path.exists(BASE_DIR):
        print(f"[HATA] '{BASE_DIR}' klasörü bulunamadı!")
        print("Bu scripti projenin kök dizininde çalıştırdığınızdan emin olun.")
        return

    headers = []
    for i in range(21):
        headers.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    headers.append("label")

    total_ok = 0
    total_skip = 0

    with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:

            classes = sorted(os.listdir(BASE_DIR))
            print(f"[BİLGİ] {len(classes)} sınıf bulundu: {classes}\n")

            for class_name in classes:
                class_path = os.path.join(BASE_DIR, class_name)
                if not os.path.isdir(class_path):
                    continue

                class_ok = 0
                class_skip = 0

                for file_name in os.listdir(class_path):
                    if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                        continue

                    img_path = os.path.join(class_path, file_name)
                    row = process_image(img_path, class_name, hands)

                    if row is not None:
                        writer.writerow(row)
                        class_ok += 1
                    else:
                        class_skip += 1

                total_ok += class_ok
                total_skip += class_skip
                print(f"  [{class_name}] ✓ {class_ok} kayıt  |  ✗ {class_skip} atlandı (el bulunamadı)")

    print(f"\n[TAMAMLANDI] {CSV_PATH}")
    print(f"  Toplam kayıt : {total_ok}")
    print(f"  Toplam atlanan: {total_skip}")


if __name__ == "__main__":
    main()