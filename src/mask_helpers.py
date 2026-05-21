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

def image_to_uint8_rgb(image: np.ndarray) -> np.ndarray:
    """
    Convert an HxWx3 RGB array to uint8 (0-255) for MediaPipe FaceMesh.

    Accepts float images in [0, 1] or [0, 255] and uint8 inputs unchanged.
    """
    arr = np.asarray(image)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"Expected HxWx3 RGB image, got shape {arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    arr = arr.astype(np.float32)
    if arr.max() <= 1.0:
        arr = arr * 255.0
    return np.clip(arr, 0, 255).astype(np.uint8)


def image_to_float01_rgb(image: np.ndarray) -> np.ndarray:
    """Convert an HxWx3 RGB array to float32 in [0, 1] for Keras / GradCAM."""
    return image_to_uint8_rgb(image).astype(np.float32) / 255.0


def img_size_from_rgb(image_rgb_uint8: np.ndarray) -> Tuple[int, int]:
    """Return (height, width) — same convention as dataloader ``img_size=(H, W)``."""
    h, w = image_rgb_uint8.shape[:2]
    return (int(h), int(w))
    
def create_landmark_mask(
    image_np_uint8: np.ndarray,
    img_size: Tuple[int, int],
    background_mask_value: float = 0.0,
    landmark_box_half_size: int = 12,
) -> Optional[np.ndarray]:
    """
    Create dynamic landmark-based mask for a single image.
    
    Args:
        image_np_uint8: (H, W, 3) RGB uint8 numpy array (already at img_size)
        img_size: Target image size (height, width)
        background_mask_value: background fill value outside ROI boxes
        landmark_box_half_size: half side length for each landmark square ROI
    
    Returns:
        (H, W) float32 mask where background = background_mask_value, ROI = 1.0
        Returns None if no face detected
    """
    h, w = img_size
    bg = float(background_mask_value)
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
    background_mask_value: float = 0.0,
) -> np.ndarray:
    """
    Create simple static eye+mouth ROI mask.
    """
    h, w = img_size
    bg = float(background_mask_value)

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