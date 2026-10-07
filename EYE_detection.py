from pathlib import Path

import cv2


def main():
    data_dir = Path(__file__).resolve().parent / "data"
    face_cascade = cv2.CascadeClassifier(
        str(data_dir / "haarcascade_frontalface_default (1).xml")
    )
    eye_cascade = cv2.CascadeClassifier(
        str(data_dir / "haarcascade_eye_tree_eyeglasses.xml")
    )

    if face_cascade.empty():
        raise FileNotFoundError(f"Could not load face cascade from {data_dir}")
    if eye_cascade.empty():
        raise FileNotFoundError(f"Could not load eye cascade from {data_dir}")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open the webcam. Check that it is connected and available.")

    # Request a modest resolution to reduce the amount of work per frame.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                raise RuntimeError("Could not read a frame from the webcam.")

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(
                gray, scaleFactor=1.3, minNeighbors=5
            )

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                roi_gray = gray[y:y + h, x:x + w]
                roi_color = frame[y:y + h, x:x + w]

                eyes = eye_cascade.detectMultiScale(roi_gray)
                for (ex, ey, ew, eh) in eyes:
                    cv2.rectangle(
                        roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2
                    )
                    label = "Left Eye" if ex < w / 2 else "Right Eye"
                    cv2.putText(
                        roi_color,
                        label,
                        (ex, ey - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        1,
                    )

            # Keep the window responsive even when no face or eyes are detected.
            cv2.imshow("Face Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
