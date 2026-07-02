"""
Landmark-based face alignment with graceful degradation.

Preferred backends (auto-detected, in order):
  1. dlib          – 68-point predictor (needs shape_predictor_68 .dat).
  2. insightface   – 5-point keypoints from RetinaFace.
  3. opencv-haar   – Haar cascade face + eye detection (ships with OpenCV).
  4. center-crop   – the original resize + center crop, always available.

Alignment performs a similarity transform so the eyes are horizontal and at a
canonical position/scale, then crops to ``size x size``. If no face/eyes are
found, we fall back to a plain center crop so preprocessing never fails.
"""
import os
import numpy as np
import cv2


def _center_crop(img: np.ndarray, size: int) -> np.ndarray:
    """Resize (short side -> size) then center crop to size x size."""
    h, w = img.shape[:2]
    scale = size / min(h, w)
    img = cv2.resize(img, (max(int(w * scale), size), max(int(h * scale), size)))
    h, w = img.shape[:2]
    top = (h - size) // 2
    left = (w - size) // 2
    return img[top:top + size, left:left + size]


def _similarity_align(img, left_eye, right_eye, size,
                      eye_y=0.38, eye_dist_ratio=0.36):
    """
    Align so both eyes sit on a horizontal line at height ``eye_y * size`` with
    a target inter-ocular distance of ``eye_dist_ratio * size``.
    """
    left_eye = np.array(left_eye, dtype=np.float32)
    right_eye = np.array(right_eye, dtype=np.float32)
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))
    dist = np.hypot(dx, dy)
    if dist < 1e-3:
        return _center_crop(img, size)

    target_dist = eye_dist_ratio * size
    scale = target_dist / dist
    eyes_center = ((left_eye + right_eye) / 2.0).tolist()

    M = cv2.getRotationMatrix2D(tuple(eyes_center), angle, scale)
    # Shift so the eye center lands at the canonical location.
    tx = size * 0.5 - eyes_center[0]
    ty = size * eye_y - eyes_center[1]
    M[0, 2] += tx
    M[1, 2] += ty
    return cv2.warpAffine(img, M, (size, size), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


class FaceAligner:
    """Detects a face + eye landmarks and returns an aligned crop."""

    def __init__(self, backend: str = "auto",
                 dlib_predictor: str = None):
        self.backend = backend
        self.dlib_predictor_path = dlib_predictor or os.environ.get(
            "DLIB_PREDICTOR", "shape_predictor_68_face_landmarks.dat")
        self._impl = None
        self._resolved = None
        self._init_backend()

    def _init_backend(self):
        order = ([self.backend] if self.backend != "auto"
                 else ["dlib", "insightface", "opencv"])
        for name in order:
            try:
                getattr(self, f"_init_{name}")()
                self._resolved = name
                return
            except Exception:
                continue
        self._resolved = "center"

    # --- backend initialisers ---
    def _init_dlib(self):
        import dlib
        if not os.path.exists(self.dlib_predictor_path):
            raise FileNotFoundError(self.dlib_predictor_path)
        self._detector = dlib.get_frontal_face_detector()
        self._predictor = dlib.shape_predictor(self.dlib_predictor_path)
        self._impl = self._eyes_dlib

    def _init_insightface(self):
        from insightface.app import FaceAnalysis
        app = FaceAnalysis(allowed_modules=["detection"])
        app.prepare(ctx_id=-1, det_size=(320, 320))
        self._app = app
        self._impl = self._eyes_insightface

    def _init_opencv(self):
        base = cv2.data.haarcascades
        self._face_cc = cv2.CascadeClassifier(base + "haarcascade_frontalface_default.xml")
        self._eye_cc = cv2.CascadeClassifier(base + "haarcascade_eye.xml")
        if self._face_cc.empty() or self._eye_cc.empty():
            raise RuntimeError("Haar cascades unavailable")
        self._impl = self._eyes_opencv

    # --- eye detectors (return left_eye, right_eye or None) ---
    def _eyes_dlib(self, img):
        import dlib  # noqa
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dets = self._detector(gray, 1)
        if not dets:
            return None
        shape = self._predictor(gray, dets[0])
        pts = np.array([[shape.part(i).x, shape.part(i).y] for i in range(68)])
        left = pts[36:42].mean(axis=0)
        right = pts[42:48].mean(axis=0)
        return left, right

    def _eyes_insightface(self, img):
        bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        faces = self._app.get(bgr)
        if not faces:
            return None
        kps = faces[0].kps  # 5 points: left eye, right eye, nose, mouth corners
        return kps[0], kps[1]

    def _eyes_opencv(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        faces = self._face_cc.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
        if len(faces) == 0:
            return None
        x, y, w, h = sorted(faces, key=lambda f: f[2] * f[3])[-1]
        roi = gray[y:y + h, x:x + w]
        eyes = self._eye_cc.detectMultiScale(roi, 1.1, 5)
        if len(eyes) < 2:
            return None
        eyes = sorted(eyes, key=lambda e: e[2] * e[3])[-2:]
        centers = [(x + ex + ew / 2, y + ey + eh / 2) for ex, ey, ew, eh in eyes]
        centers = sorted(centers, key=lambda c: c[0])  # left first
        return np.array(centers[0]), np.array(centers[1])

    def align(self, img: np.ndarray, size: int = 256) -> np.ndarray:
        """Return an aligned ``size x size`` RGB crop, or center crop on failure."""
        if self._impl is not None:
            try:
                eyes = self._impl(img)
                if eyes is not None:
                    return _similarity_align(img, eyes[0], eyes[1], size)
            except Exception:
                pass
        return _center_crop(img, size)


# Module-level convenience singleton (lazily created).
_DEFAULT_ALIGNER = None


def align_face(img: np.ndarray, size: int = 256, backend: str = "auto") -> np.ndarray:
    """Functional entry point used by preprocessing."""
    global _DEFAULT_ALIGNER
    if _DEFAULT_ALIGNER is None or _DEFAULT_ALIGNER.backend != backend:
        _DEFAULT_ALIGNER = FaceAligner(backend=backend)
    return _DEFAULT_ALIGNER.align(img, size)
