import cv2
from detect_face import FaceDetector

print("file is running")

def main():
    detector = FaceDetector()
    print("inside def main")

    # Try different backends
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    if not cap.isOpened():
        print("DSHOW failed, trying MSMF")
        cap = cv2.VideoCapture(0, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("MSMF failed, trying default backend")
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not detected!")
        return
    
    print("Camera opened successfully!")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to read frame!")
            break
        

        bboxes = detector.detect_faces(frame)

        # Draw bounding boxes
        for (x1, y1, x2, y2) in bboxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

        frame=cv2.flip(frame,1)
        cv2.imshow("Face Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
