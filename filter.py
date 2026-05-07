import cv2
import numpy as np

def good_brightness(frame, min_threshold=50, max_threshold=200):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    brightness = np.mean(gray)
    return brightness > min_threshold and brightness < max_threshold, brightness

def main():
    cap = cv2.VideoCapture("dylan-right-iris-video.mkv")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        ok, brightness = good_brightness(frame)

        if not ok:
            print(f"Frame rejected: {brightness:.2f}")
            continue

        cv2.imshow("Frame", frame)

        if cv2.waitKey(30) == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
