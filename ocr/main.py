from __future__ import annotations

from dataclasses import dataclass
import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import pytesseract


@dataclass(frozen=True)
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    conf: Optional[float] = None


_PLATE_RE = re.compile(r"^[A-Z]{3,4}\d{3,4}$")
_NON_ALNUM_RE = re.compile(r"[^A-Z0-9]")


def normalize_plate(text: str) -> str:
    text = (text or "").upper()
    text = _NON_ALNUM_RE.sub("", text)
    if not text:
        return text

    # Fix a common OCR duplication: if OCR outputs one extra trailing character
    # and removing it yields a valid plate, prefer the corrected version.
    if len(text) > 7 and len(text) >= 2 and text[-1] == text[-2]:
        trimmed = text[:-1]
        if _PLATE_RE.match(trimmed) and not _PLATE_RE.match(text):
            return trimmed

    return text


def _clamp_bbox(bbox: BBox, w: int, h: int) -> BBox:
    x1 = max(0, min(w - 1, int(bbox.x1)))
    y1 = max(0, min(h - 1, int(bbox.y1)))
    x2 = max(0, min(w - 1, int(bbox.x2)))
    y2 = max(0, min(h - 1, int(bbox.y2)))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return BBox(x1=x1, y1=y1, x2=x2, y2=y2, conf=bbox.conf)


def crop_plate(
    image_bgr: np.ndarray,
    bbox: BBox,
    *,
    margin: float = 0.20,
    min_pad_px: int = 10,
) -> np.ndarray:
    """
    Crop plate region from the original image using a YOLO-style xyxy bbox.
    `margin` expands the bbox to improve OCR accuracy.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image passed to OCR")

    h, w = image_bgr.shape[:2]
    bbox = _clamp_bbox(bbox, w=w, h=h)

    bw = bbox.x2 - bbox.x1
    bh = bbox.y2 - bbox.y1
    mx = max(min_pad_px, int(round(bw * margin)))
    my = max(min_pad_px, int(round(bh * margin)))

    x1 = max(0, bbox.x1 - mx)
    y1 = max(0, bbox.y1 - my)
    x2 = min(w, bbox.x2 + mx)
    y2 = min(h, bbox.y2 + my)

    crop = image_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        raise ValueError("Cropped plate is empty")
    return crop


def _preprocess_for_tesseract(crop_bgr: np.ndarray, *, invert: bool) -> np.ndarray:
    """
    Preprocess crop to a high-contrast binary image.
    Using a couple of preprocessing variants (invert on/off) improves robustness.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    # Scale up for Tesseract; improves small-font recognition.
    scale = 2.5
    gray = cv2.resize(
        gray,
        dsize=None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )

    # Reduce noise while preserving edges.
    gray = cv2.bilateralFilter(gray, d=7, sigmaColor=50, sigmaSpace=50)

    # Otsu binarization + optional inversion.
    _, th_otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    th = cv2.bitwise_not(th_otsu) if invert else th_otsu

    # Morphology to connect character strokes.
    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th


def _preprocess_adaptive(crop_bgr: np.ndarray, *, invert: bool) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    scale = 2.5
    gray = cv2.resize(
        gray,
        dsize=None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_CUBIC,
    )
    gray = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive thresholding can work better under uneven lighting.
    th = cv2.adaptiveThreshold(
        gray,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        2,
    )
    if invert:
        th = cv2.bitwise_not(th)

    kernel = np.ones((3, 3), np.uint8)
    th = cv2.morphologyEx(th, cv2.MORPH_CLOSE, kernel, iterations=1)
    return th


def _tesseract_ocr(bin_img: np.ndarray) -> str:
    # Try multiple PSM modes; different plate layouts can benefit from different settings.
    whitelist = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    results: list[str] = []
    for psm in (6, 7, 8, 10, 11, 13):
        config = f"--oem 3 --psm {psm} -c tessedit_char_whitelist={whitelist}"
        try:
            text = pytesseract.image_to_string(bin_img, config=config)
        except Exception:
            text = ""
        results.append(normalize_plate(text))

    # Return the best candidate among those PSM tries.
    best = ""
    best_score = -10**9
    for r in results:
        s = _score_candidate(r)
        if s > best_score:
            best_score = s
            best = r
    return best


def _preprocess_gray_clahe(crop_bgr: np.ndarray, *, invert: bool) -> np.ndarray:
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # Scale up for small characters
    scale = 4.0
    gray = cv2.resize(gray, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    gray = cv2.bilateralFilter(gray, 9, 75, 75)
    if invert:
        gray = cv2.bitwise_not(gray)
    return gray


def _score_candidate(text: str) -> int:
    if not text:
        return -1
    if _PLATE_RE.match(text):
        # Favor exact-format matches, then longer strings.
        return 1000 + len(text)
    # Partial candidates are still useful but get a lower score.
    # Heuristic: more alnum chars = higher score.
    alnum = len(text)
    return min(alnum, 50)


def ocr_plate_from_bbox(
    image_path: str | Path,
    bbox: BBox,
    *,
    margin: float = 0.20,
    debug_dir: Optional[str | Path] = None,
    debug_prefix: str = "",
) -> str:
    """
    Crop -> preprocess -> OCR for the given plate bbox.
    Returns a normalized plate string (A-Z/0-9 only).
    """
    image_path = Path(image_path)
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    crop = crop_plate(image_bgr, bbox, margin=margin)

    debug_path = Path(debug_dir) if debug_dir else None
    if debug_path:
        debug_path.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(debug_path / f"{debug_prefix}crop.jpg"), crop)

    candidates: list[str] = []
    for variant_name, preprocess_fn in (
        ("gray_clahe", lambda inv: _preprocess_gray_clahe(crop, invert=inv)),
        ("otsu", lambda inv: _preprocess_for_tesseract(crop, invert=inv)),
        ("adaptive", lambda inv: _preprocess_adaptive(crop, invert=inv)),
    ):
        for invert in (False, True):
            proc = preprocess_fn(invert)
            if debug_path:
                cv2.imwrite(
                    str(debug_path / f"{debug_prefix}preprocessed_{variant_name}_invert_{int(invert)}.jpg"),
                    proc,
                )
            candidates.append(_tesseract_ocr(proc))

    # Select best candidate via regex+heuristic scoring.
    best = ""
    best_score = -10**9
    for c in candidates:
        s = _score_candidate(c)
        if s > best_score:
            best_score = s
            best = c
    return best


def ocr_plate_image(image_bgr: np.ndarray) -> str:
    """
    Fallback OCR for callers that already cropped the plate image.
    """
    if image_bgr is None or image_bgr.size == 0:
        return ""

    # Similar preprocessing as bbox flow, but operate directly on the crop.
    candidates = []
    for invert in (False, True):
        proc = _preprocess_for_tesseract(image_bgr, invert=invert)
        candidates.append(_tesseract_ocr(proc))

    best = ""
    best_score = -10**9
    for c in candidates:
        s = _score_candidate(c)
        if s > best_score:
            best_score = s
            best = c
    return best
