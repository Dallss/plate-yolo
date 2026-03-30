from __future__ import annotations

import argparse
from dataclasses import asdict
import sys
from pathlib import Path
from typing import Iterable, Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO  # type: ignore
except ModuleNotFoundError:
    # Attempt to import ultralytics from the local `venv/` folder.
    repo_root = Path(__file__).resolve().parent
    venv_site_packages = None
    for p in (repo_root / "venv" / "lib").glob("python*/site-packages"):
        if (p / "ultralytics").exists():
            venv_site_packages = p
            break
    if venv_site_packages:
        sys.path.insert(0, str(venv_site_packages))
        from ultralytics import YOLO  # type: ignore
    else:
        raise

from ocr.main import BBox as OcrBBox
from ocr.main import ocr_plate_from_bbox, normalize_plate


_PLATE_RE = __import__("re").compile(r"^[A-Z]{3,4}\d{3,4}$")


def _load_model() -> YOLO:
    model_path = Path("detect-plate") / "runs" / "detect" / "train" / "weights" / "best.pt"
    if not model_path.exists():
        raise FileNotFoundError(f"YOLO weights not found: {model_path}")
    return YOLO(str(model_path))


def detect_plate_bboxes(
    model: YOLO,
    image_path: str | Path,
    *,
    conf: float = 0.05,
) -> list[OcrBBox]:
    results = model.predict(source=str(image_path), conf=conf, verbose=False)
    if not results:
        return []
    r0 = results[0]
    boxes = getattr(r0, "boxes", None)
    if boxes is None:
        return []

    bboxes: list[OcrBBox] = []
    for box in boxes:
        xyxy = box.xyxy[0].cpu().numpy()  # x1,y1,x2,y2
        x1, y1, x2, y2 = [float(v) for v in xyxy]
        cls = int(box.cls[0].cpu().numpy()) if hasattr(box, "cls") else None
        score = float(box.conf[0].cpu().numpy()) if hasattr(box, "conf") else None
        # If the model has multiple classes, you can filter by `cls` here.
        bboxes.append(OcrBBox(x1=int(round(x1)), y1=int(round(y1)), x2=int(round(x2)), y2=int(round(y2)), conf=score))
    return bboxes


def _iter_test_items(test_images_dir: Path, test_labels_dir: Path) -> Iterable[tuple[str, Path, str]]:
    # Image stems are numeric: 1.jpg -> 1.txt
    image_paths = sorted(test_images_dir.glob("*.*"), key=lambda p: p.stem)
    for image_path in image_paths:
        stem = image_path.stem
        label_path = test_labels_dir / f"{stem}.txt"
        if not label_path.exists():
            continue
        label = label_path.read_text(encoding="utf-8").strip()
        yield stem, image_path, label


def _pick_test_items(all_items: list[tuple[str, Path, str]], ids: Optional[list[str]]) -> list[tuple[str, Path, str]]:
    if not ids:
        return all_items
    id_set = set(ids)
    return [it for it in all_items if it[0] in id_set]


def run_pipeline(
    *,
    ids: Optional[list[str]] = None,
    conf: float = 0.05,
    top_k_bboxes: int = 3,
    debug_dir: Optional[str | Path] = None,
) -> dict:
    repo_root = Path(__file__).resolve().parent
    test_images_dir = repo_root / "test" / "image"
    test_labels_dir = repo_root / "test" / "labels"

    if not test_images_dir.exists() or not test_labels_dir.exists():
        raise FileNotFoundError("Expected `test/image` and `test/labels` folders")

    model = _load_model()

    all_items = list(_iter_test_items(test_images_dir, test_labels_dir))
    items = _pick_test_items(all_items, ids)
    if not items:
        raise ValueError("No matching test items found")

    correct = 0
    total = 0
    per_item = []

    debug_root = Path(debug_dir) if debug_dir else None

    for stem, image_path, gt in items:
        total += 1
        gt_norm = normalize_plate(gt)

        bboxes = detect_plate_bboxes(model, image_path, conf=conf)
        # Sort by confidence if available.
        bboxes.sort(key=lambda b: float(b.conf) if b.conf is not None else 0.0, reverse=True)
        bboxes = bboxes[: max(1, top_k_bboxes)]

        pred = ""
        pred_bbox_conf = None

        # Use OCR on the top-k bboxes and pick the best-looking OCR output.
        # (We re-run scoring inside OCR, but selection between bboxes is heuristic here.)
        best_score = -10**9
        for i, bbox in enumerate(bboxes):
            bbox_conf = bbox.conf
            prefix = f"{stem}_bbox{i}_"
            ocr_pred = ocr_plate_from_bbox(
                image_path,
                bbox,
                margin=0.20,
                debug_dir=debug_root,
                debug_prefix=prefix if debug_root else "",
            )
            ocr_pred_norm = normalize_plate(ocr_pred)

            # Prefer regex-shaped plates first; otherwise shorter (less noisy) strings.
            if _PLATE_RE.match(ocr_pred_norm):
                score = 1000 + len(ocr_pred_norm)
            else:
                score = min(len(ocr_pred_norm), 50)
            if ocr_pred_norm == gt_norm:
                score += 500

            if score > best_score:
                best_score = score
                pred = ocr_pred_norm
                pred_bbox_conf = bbox_conf
        
        is_correct = pred == gt_norm and bool(pred)
        if is_correct:
            correct += 1

        per_item.append(
            {
                "image": str(image_path.relative_to(repo_root)),
                "gt": gt_norm,
                "pred": pred,
                "correct": is_correct,
                "top_boxes": [asdict(b) for b in bboxes],
                "pred_bbox_conf": pred_bbox_conf,
            }
        )

        print(f"{stem}: gt={gt_norm} pred={pred} match={is_correct}")

    accuracy = correct / total if total else 0.0
    print(f"Accuracy: {correct}/{total} = {accuracy:.2%}")
    return {"accuracy": accuracy, "correct": correct, "total": total, "per_item": per_item}


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect plates -> crop -> OCR -> verify with labels.")
    parser.add_argument(
        "--ids",
        nargs="*",
        default=None,
        help="Test image ids to run (e.g. --ids 1 3). Defaults to all if omitted.",
    )
    parser.add_argument("--conf", type=float, default=0.05, help="YOLO confidence threshold.")
    parser.add_argument("--top-k-bboxes", type=int, default=3, help="Run OCR on top K detected bboxes.")
    parser.add_argument("--debug-dir", default=None, help="If set, save OCR crops/preprocessed images here.")
    args = parser.parse_args()

    run_pipeline(
        ids=args.ids,
        conf=args.conf,
        top_k_bboxes=args.top_k_bboxes,
        debug_dir=args.debug_dir,
    )


if __name__ == "__main__":
    main()
