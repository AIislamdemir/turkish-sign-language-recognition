
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model
from collections import deque, Counter

# ─────────────────────────────────────────
# MODEL & SINIFLAR
# ─────────────────────────────────────────
print("Model yükleniyor...")
model = load_model("letter_model.h5")

# train_model.py'nin kaydettiği sınıf sırasını yükle
# (eğitimde LabelEncoder hangi sırayla eğittiyse o sıra geçerli)
try:
    classes = list(np.load("label_classes.npy", allow_pickle=True))
    print(f"Sınıflar yüklendi: {classes}")
except FileNotFoundError:
    # Yedek: manuel liste (eğitimde kaç sınıf varsa sırayla)
    classes = ['A','B','C','Ç','D','E','F','G','Ğ','H','I','İ','J','K',
               'L','M','N','O','Ö','P','R','S','Ş','T','U','Ü','V','Y','Z']
    print("[UYARI] label_classes.npy bulunamadı, varsayılan liste kullanılıyor.")

# ─────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.80  # Minimum güven skoru
BUFFER_SIZE = 15              # Kaç tahmin ortalaması alınacak

prediction_buffer = deque(maxlen=BUFFER_SIZE)

# ─────────────────────────────────────────
# MEDİAPİPE
# ─────────────────────────────────────────
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ─────────────────────────────────────────
# KAMERA
# ─────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

print("Kamera açıldı. 'Q' ile çıkın.")

final_text = ""
confidence_display = 0.0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = hands.process(rgb)
    rgb.flags.writeable = True

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        # El iskeletini çiz
        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style()
        )

        # 63 özelliği çıkar (21 nokta × 3)
        row = []
        for lm in hand_landmarks.landmark:
            row.extend([lm.x, lm.y, lm.z])

        if len(row) == 63:
            input_data = np.array(row, dtype=np.float32).reshape(1, 63)
            prediction = model.predict(input_data, verbose=0)

            class_id = int(np.argmax(prediction))
            confidence = float(np.max(prediction))
            confidence_display = confidence

            if confidence > CONFIDENCE_THRESHOLD:
                prediction_buffer.append(classes[class_id])

    # Majority voting ile kararlı tahmin
    if len(prediction_buffer) > 5:
        most_common = Counter(prediction_buffer).most_common(1)[0][0]
        final_text = most_common

    # ─── EKRAN YAZILARI ───

    h, w = frame.shape[:2]

    # Arka plan kutusu (sol üst)
    cv2.rectangle(frame, (0, 0), (350, 110), (0, 0, 0), -1)

    # Tahmin edilen harf
    cv2.putText(frame, final_text,
                (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX,
                3.5, (0, 255, 0), 6)

    # Güven skoru
    conf_color = (0, 255, 0) if confidence_display > CONFIDENCE_THRESHOLD else (0, 165, 255)
    cv2.putText(frame,
                f"Guven: {confidence_display:.0%}",
                (w - 230, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8, conf_color, 2)

    # Çıkış ipucu
    cv2.putText(frame, "Q: Cikis",
                (w - 160, h - 15),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (180, 180, 180), 1)

    cv2.imshow("TID - Harf Tanima", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Program sonlandırıldı.")