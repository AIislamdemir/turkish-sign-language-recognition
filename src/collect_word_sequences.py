"""
collect_word_sequences.py  (v2)
================================
Günlük konuşma için 30 kelimelik TİD sekans verisi toplar.

Kullanım:
    python src/collect_word_sequences.py
    (Proje kök dizininden çalıştırın)

Kontroller:
    SPACE  → Kayıt başlat
    N      → Bu kelimeyi atla, sonrakine geç
    Q      → Programdan çık
"""

import os
import cv2
import numpy as np
import mediapipe as mp

# ─────────────────────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────────────────────
DATA_PATH          = "words_datasets/words"
FRAMES_PER_SEQUENCE = 30    # Her kayıt kaç frame
SAMPLES_PER_WORD   = 60     # Her kelime için kaç örnek

# Kelime listesi — words_datasets/words/ klasör isimleriyle birebir eşleşir
WORDS = [
    "anne", "baba", "ben", "biz", "ev",
    "evet", "gel", "gitt", "güle güle", "hayır",
    "iyiyim", "iş", "ozur dilerim", "kötü", "lütfen",
    "merhaba", "nasılsın", "ne", "ne zaman", "nerde",
    "o", "okul", "onlar", "para", "senin",
    "siz", "tamam", "telefon", "teşekürler", "memnun oldum",
    "görüşürüz","rica ederim","benim","adım",
]

# ─────────────────────────────────────────────────────────────
# MEDİAPİPE — Normalizasyonlu landmark çıkarıcı
# ─────────────────────────────────────────────────────────────
mp_hands  = mp.solutions.hands
hands_sol = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)

_ZEROS = np.zeros(63, dtype=np.float32)


def _normalize(hand_lm):
    """Wrist referanslı + scale normalizasyonu (extract_landmarks_fixed ile aynı)."""
    coords = np.array([[lm.x, lm.y, lm.z] for lm in hand_lm.landmark],
                      dtype=np.float32)
    coords -= coords[0]
    max_val = np.max(np.abs(coords))
    if max_val > 0:
        coords /= max_val
    return coords.flatten()


def extract(frame):
    """Frame'den normalize edilmiş çift el vektörü çıkar (126 float)."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb.flags.writeable = False
    result = hands_sol.process(rgb)
    rgb.flags.writeable = True

    if not result.multi_hand_landmarks:
        return None

    left  = _ZEROS.copy()
    right = _ZEROS.copy()

    for hlm, handedness in zip(result.multi_hand_landmarks,
                                result.multi_handedness):
        label = handedness.classification[0].label
        vec   = _normalize(hlm)
        if label == "Left":
            left  = vec
        else:
            right = vec

        # El iskeletini çiz
        h, w = frame.shape[:2]
        colour = (60, 200, 170) if label == "Left" else (40, 170, 255)
        for conn in mp_hands.HAND_CONNECTIONS:
            x1 = int(hlm.landmark[conn[0]].x * w)
            y1 = int(hlm.landmark[conn[0]].y * h)
            x2 = int(hlm.landmark[conn[1]].x * w)
            y2 = int(hlm.landmark[conn[1]].y * h)
            cv2.line(frame, (x1, y1), (x2, y2), colour, 2)
        for lm in hlm.landmark:
            cv2.circle(frame, (int(lm.x*w), int(lm.y*h)), 4, (255,255,255), -1)

    return np.concatenate([left, right])   # 126 float


# ─────────────────────────────────────────────────────────────
# ANA FONKSİYON
# ─────────────────────────────────────────────────────────────
def main():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    print("=" * 50)
    print("  TİD Kelime Veri Toplama")
    print("=" * 50)
    print(f"  Kelime sayısı : {len(WORDS)}")
    print(f"  Örnek/kelime  : {SAMPLES_PER_WORD}")
    print(f"  Frame/örnek   : {FRAMES_PER_SEQUENCE}")
    print()
    print("  SPACE → Kayıt başlat")
    print("  N     → Sonraki kelime")
    print("  Q     → Çık")
    print("=" * 50)

    for word in WORDS:
        word_path = os.path.join(DATA_PATH, word)
        os.makedirs(word_path, exist_ok=True)

        existing   = len([f for f in os.listdir(word_path) if f.endswith(".npy")])
        sample_num = existing

        if sample_num >= SAMPLES_PER_WORD:
            print(f"\n  [{word}] zaten tamamlanmış ({existing} örnek), atlanıyor.")
            continue

        print(f"\n{'─'*50}")
        print(f"  KELİME: {word.upper().replace('_',' ')}  ({sample_num}/{SAMPLES_PER_WORD})")
        print(f"{'─'*50}")

        skip_word = False

        while sample_num < SAMPLES_PER_WORD:
            sequence  = []
            recording = False

            while True:
                ret, frame = cap.read()
                if not ret:
                    continue

                frame     = cv2.flip(frame, 1)
                landmarks = extract(frame)
                key       = cv2.waitKey(1) & 0xFF

                if key == ord(' '):
                    recording = True
                elif key == ord('n'):
                    print(f"  [{word}] atlandı.")
                    skip_word = True
                    break
                elif key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    hands_sol.close()
                    print("\nProgram sonlandırıldı.")
                    return

                if recording and landmarks is not None:
                    sequence.append(landmarks)

                # ── Ekran bilgisi ──
                status = "KAYIT" if recording else "HAZIR — SPACE ile baslat"
                colour = (0, 60, 255) if recording else (200, 200, 200)

                # Arka plan şeridi
                cv2.rectangle(frame, (0, 0), (640, 140), (0, 0, 0), -1)

                cv2.putText(frame,
                            f"Kelime: {word.upper().replace('_',' ')}",
                            (12, 38), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                            (0, 255, 0), 2)
                cv2.putText(frame,
                            f"Ornek: {sample_num + 1}/{SAMPLES_PER_WORD}",
                            (12, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (255, 255, 0), 2)
                cv2.putText(frame,
                            f"Frame: {len(sequence)}/{FRAMES_PER_SEQUENCE}   {status}",
                            (12, 118), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                            colour, 2)

                # Frame progress bar
                if recording:
                    bar_w = int((len(sequence) / FRAMES_PER_SEQUENCE) * 600)
                    cv2.rectangle(frame, (12, 128), (12 + bar_w, 136), (0, 200, 150), -1)

                cv2.imshow("TID — Kelime Veri Toplama", frame)

                # Yeterli frame toplandıysa kaydet
                if len(sequence) == FRAMES_PER_SEQUENCE:
                    arr       = np.array(sequence, dtype=np.float32)
                    save_path = os.path.join(word_path, f"{sample_num}.npy")
                    np.save(save_path, arr)
                    print(f"  ✓ Kaydedildi: {save_path}  (şekil: {arr.shape})")
                    sample_num += 1
                    break

            if skip_word:
                break

        if not skip_word:
            print(f"\n  ✅ [{word}] tamamlandı — {sample_num} örnek")

    cap.release()
    cv2.destroyAllWindows()
    hands_sol.close()
    print("\n" + "=" * 50)
    print("  Tüm veri toplama tamamlandı!")
    print("=" * 50)


if __name__ == "__main__":
    main()