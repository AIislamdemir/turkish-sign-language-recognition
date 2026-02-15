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
        row.extend([lm.x, lm.y, lm.z])

    row.append(class_label)
    return row

def main():
    headers = []
    for i in range(21):
        headers.extend([f"x_{i}", f"y_{i}", f"z_{i}"])
    headers.append("label")

    with open(CSV_PATH, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers)

        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=1,
            min_detection_confidence=0.5
        ) as hands:

            for class_name in os.listdir(BASE_DIR):
                class_path = os.path.join(BASE_DIR, class_name)
                if not os.path.isdir(class_path):
                    continue

                for file_name in os.listdir(class_path):
                    if not file_name.lower().endswith((".jpg", ".png", ".jpeg")):
                        continue

                    img_path = os.path.join(class_path, file_name)
                    row = process_image(img_path, class_name, hands)

                    if row is not None:
                        writer.writerow(row)
                        print(f"OK: {img_path}")
                    else:
                        print(f"SKIP (el bulunamadı): {img_path}")

    print(f"İşlem tamamlandı. -> {CSV_PATH}")

if __name__ == "__main__":
    main()
