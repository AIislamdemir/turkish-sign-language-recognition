import cv2

def main():
    cap = cv2.VideoCapture(0)  # default camera

    if not cap.isOpened():
        print("Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Mirror effect (flip horizontally)
        frame = cv2.flip(frame, 1)

        cv2.imshow("Camera Test (Mirror)", frame)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
