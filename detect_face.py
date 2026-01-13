import cv2



import mediapipe as mp

print("det_face imported")

class FaceDetector:
   
    def __init__(self, min_detection_confidence=0.6):
        """Real-time Mediapipe face detector (bounding boxes only)."""
       

        self.mp_face = mp.solutions.face_detection

        
        self.mp_draw = mp.solutions.drawing_utils
        self.detector = self.mp_face.FaceDetection(
            model_selection=0,
            min_detection_confidence=min_detection_confidence
        )

    def detect_faces(self, frame):
        """Detect faces in a BGR frame and return bounding boxes."""
        h, w, _ = frame.shape
        results = self.detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        
        bboxes = []

        if results.detections:
            for det in results.detections:
                box = det.location_data.relative_bounding_box
                x1 = int(box.xmin * w)
                y1 = int(box.ymin * h)
                x2 = int((box.xmin + box.width) * w)
                y2 = int((box.ymin + box.height) * h)
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(w, x2)
                y2 = min(h, y2)
                bboxes.append((x1, y1, x2, y2))

        return bboxes


