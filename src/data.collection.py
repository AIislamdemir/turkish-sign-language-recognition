import cv2
import os

BASE_DIR = "dataset"

CLASSES = [
"A","B","C","Ç","D","E","F","G","Ğ","H",
"I","İ","J","K","L","M","N","O","Ö","P",
"R","S","Ş","T","U","Ü","V","Y","Z"
]

TARGET_PER_CLASS = 300

def make_dirs():
    os.makedirs(BASE_DIR, exist_ok=True)
    for c in CLASSES:
        os.makedirs(os.path.join(BASE_DIR, c), exist_ok=True)

def initial_counts():
    counts = {}
    for c in CLASSES:
        class_path = os.path.join(BASE_DIR, c)
        counts[c] = len([f for f in os.listdir(class_path) if f.endswith(".jpg")])
    return counts

def main():
    make_dirs()
    counts = initial_counts()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not opened.")
        return

    current_index = 0

    print("SPACE -> Save image")
    print("N -> Next letter")
    print("Q -> Quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        current_letter = CLASSES[current_index]
        current_count = counts[current_letter]

        # Sayaç ve bilgiler
        cv2.putText(frame,
                    f"Letter: {current_letter}",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

        cv2.putText(frame,
                    f"Count: {current_count}/{TARGET_PER_CLASS}",
                    (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8, (0,255,0), 2)

        cv2.imshow("Alphabet Data Collector", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        elif key == ord(' '):
            if current_count < TARGET_PER_CLASS:
                save_path = os.path.join(
                    BASE_DIR,
                    current_letter,
                    f"{current_letter}_{current_count+1}.jpg"
                )
                cv2.imwrite(save_path, frame)
                counts[current_letter] += 1

        elif key == ord('n'):
            current_index = (current_index + 1) % len(CLASSES)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
