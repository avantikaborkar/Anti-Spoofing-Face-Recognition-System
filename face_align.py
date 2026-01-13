# src/face_align.py

import cv2
import numpy as np

print("face_align imported")

class FaceAligner:
    """
    Face Alignment using eye landmarks:
    - Rotates face so eyes are level
    - Crops face
    - Resizes to model input size
    """

    def __init__(self, target_size=(112, 112)):
        """
        target_size:
            (112,112) -> ArcFace
            (160,160) -> FaceNet
        """
        self.target_size = target_size

    def align(self, frame, bbox, left_eye, right_eye):
        """
        Parameters:
            frame    : Original BGR frame
            bbox     : (x1, y1, x2, y2)
            left_eye : (x, y)
            right_eye: (x, y)

        Returns:
            aligned_face : normalized RGB face
        """

        x1, y1, x2, y2 = bbox

       
        dx = right_eye[0] - left_eye[0]
        dy = right_eye[1] - left_eye[1]

        angle = np.degrees(np.arctan2(dy, dx))

       
        center = ((x1 + x2) // 2, (y1 + y2) // 2)

        rot_mat = cv2.getRotationMatrix2D(center, angle, scale=1.0)
        rotated_frame = cv2.warpAffine(
            frame,
            rot_mat,
            (frame.shape[1], frame.shape[0]),
            flags=cv2.INTER_CUBIC
        )

       
        face = rotated_frame[y1:y2, x1:x2]

        if face is None or face.size == 0:
            return None

       
        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

      
        face_resized = cv2.resize(face_rgb, self.target_size)

       
        face_normalized = (face_resized - 127.5) / 128.0

        return face_normalized

