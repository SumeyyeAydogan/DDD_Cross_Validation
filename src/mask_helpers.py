from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import mediapipe as mp


LEFT_EYE_IDX = [33, 7, 163, 144, 145, 153, 154, 155, 133]
RIGHT_EYE_IDX = [263, 249, 390, 373, 374, 380, 381, 382, 362]
MOUTH_IDX = [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308]
ROI_IDX = LEFT_EYE_IDX + RIGHT_EYE_IDX + MOUTH_IDX

_face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
)


def create_landmark_mask(
    image_np_uint8: np.ndarray,
    img_size: Tuple[int, int],
    background_value: float = 0.0,
    landmark_box_half_size: int = 12,
) -> Optional[np.ndarray]:
    """
    Create dynamic landmark-based mask for a single image.
    
    Args:
        image_np_uint8: (H, W, 3) RGB uint8 numpy array (already at img_size)
        img_size: Target image size (height, width)
        background_value: background fill value outside ROI boxes
        landmark_box_half_size: half side length for each landmark square ROI
    
    Returns:
        (H, W) float32 mask where background = background_mask_value, ROI = 1.0
        Returns None if no face detected
    """
    h, w = img_size
    bg = float(background_value)
    box_half_size = int(landmark_box_half_size)
    
    # MediaPipe expects RGB uint8 array
    # Note: image_np should already be at img_size from TensorFlow dataset
    results = _face_mesh.process(image_np_uint8)
    
    if not results.multi_face_landmarks:
        # No face detected - return None to use fallback
        return None
    
    face = results.multi_face_landmarks[0]
    mask = np.full((h, w), bg, dtype=np.float32)

    for i in ROI_IDX:
        lm = face.landmark[i]
        cx = int(lm.x * w)
        cy = int(lm.y * h)
        x0 = max(0, cx - box_half_size)
        y0 = max(0, cy - box_half_size)
        x1 = min(w, cx + box_half_size + 1)
        y1 = min(h, cy + box_half_size + 1)
        mask[y0:y1, x0:x1] = 1.0

    return mask


def create_static_mask(
    img_size: Tuple[int, int],
    background_value: float = 0.0,
) -> np.ndarray:
    """
    Create simple static eye+mouth ROI mask.
    """
    h, w = img_size
    bg = float(background_value)

    eye_top = int(0.2 * h)
    eye_bottom = int(0.53 * h)
    eye_left = int(0.1 * w)
    eye_right = int(0.9 * w)

    mouth_top = int(0.57 * h)
    mouth_bottom = int(0.9 * h)
    mouth_left = int(0.2 * w)
    mouth_right = int(0.8 * w)

    mask = np.ones((h, w), dtype=np.float32) * bg
    mask[eye_top:eye_bottom, eye_left:eye_right] = 1.0
    mask[mouth_top:mouth_bottom, mouth_left:mouth_right] = 1.0
    return mask