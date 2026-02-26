import os
import cv2
import numpy as np
from extract_landmarks import HandLandmarkExtractor

# ====== AYARLAR ======
DATA_PATH = "words_datasets/words"
FRAMES_PER_SEQUENCE = 30
SAMPLES_PER_WORD = 60
# =====================


def main():

    extractor = HandLandmarkExtractor()
    cap = cv2.VideoCapture(0)

    words = os.listdir(DATA_PATH)

    print("SPACE  -> Kayıt başlat")
    print("N      -> Sonraki kelimeye geç")
    print("Q      -> Programdan çık\n")

    for word in words:

        word_path = os.path.join(DATA_PATH, word)
        os.makedirs(word_path, exist_ok=True)

        print(f"\n===== {word.upper()} kelimesi =====")

        existing_samples = len(os.listdir(word_path))

        sample_num = existing_samples

        while sample_num < SAMPLES_PER_WORD:

            sequence = []
            recording = False

            while True:

                ret, frame = cap.read()
                if not ret:
                    continue

                # 🔥 Ayna görüntü
                frame = cv2.flip(frame, 1)

                landmarks = extractor.extract(frame)

                key = cv2.waitKey(1) & 0xFF

                # SPACE ile kayıt başlat
                if key == ord(' '):
                    recording = True

                # N ile sonraki kelime
                if key == ord('n'):
                    print(f"{word} atlandı.")
                    break

                # Q ile çık
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return

                if recording and landmarks is not None:
                    sequence.append(landmarks)

                cv2.putText(frame,
                            f"Kelime: {word}",
                            (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            (0, 255, 0),
                            2)

                cv2.putText(frame,
                            f"Sample: {sample_num + 1}/{SAMPLES_PER_WORD}",
                            (10, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 0),
                            2)

                cv2.putText(frame,
                            f"Frame: {len(sequence)}/{FRAMES_PER_SEQUENCE}",
                            (10, 120),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 255),
                            2)

                cv2.imshow("Collecting Word Sequences", frame)

                if len(sequence) == FRAMES_PER_SEQUENCE:
                    sequence = np.array(sequence)
                    save_path = os.path.join(word_path, f"{sample_num}.npy")
                    np.save(save_path, sequence)
                    print("Kaydedildi:", save_path)

                    sample_num += 1
                    break

            # Eğer N ile çıktıysa kelime değiştir
            if key == ord('n'):
                break

        print(f"{word} tamamlandı veya atlandı.")

    cap.release()
    cv2.destroyAllWindows()
    print("Tüm veri toplama tamamlandı.")


if __name__ == "__main__":
    main()