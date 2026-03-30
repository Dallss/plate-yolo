from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2


def _import_yolo():
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO
    except ModuleNotFoundError:
        # Mirror `pipline.py`: attempt to import from the local `venv/`.
        repo_root = Path(__file__).resolve().parent.parent
        venv_site_packages = None
        for p in (repo_root / "venv" / "lib").glob("python*/site-packages"):
            if (p / "ultralytics").exists():
                venv_site_packages = p
                break
        if not venv_site_packages:
            raise
        sys.path.insert(0, str(venv_site_packages))
        from ultralytics import YOLO  # type: ignore

        return YOLO


YOLO = _import_yolo()


def _pick_default_source(repo_root: Path) -> Path:
    # Prefer local dataset folders with images.
    candidates = [
        repo_root / "test" / "image",
        repo_root / "detect-plate" / "valid" / "images",
        repo_root / "detect-plate" / "train" / "images",
    ]
    for d in candidates:
        if not d.exists() or not d.is_dir():
            continue
        imgs = sorted([p for p in d.glob("*.*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}])
        if imgs:
            return d

    any_img = next(repo_root.glob("**/*.jpg"), None) or next(repo_root.glob("**/*.png"), None)
    if not any_img:
        raise FileNotFoundError("Could not find any image in the repo to run prediction on.")
    return any_img


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    default_model = repo_root / "detect-plate" / "runs" / "detect" / "train" / "weights" / "best.pt"

    parser = argparse.ArgumentParser(description="YOLO license-plate detection (CLI).")
    parser.add_argument("--model", type=Path, default=default_model, help="Path to YOLO weights (.pt).")
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Image file or directory to run inference on. If omitted, uses a default dataset folder.",
    )
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size.")
    parser.add_argument("--device", default=None, help="Inference device: 'cpu', 'mps', or '0' for GPU.")
    parser.add_argument("--max-images", type=int, default=50, help="Max number of images to process.")
    parser.add_argument("--show", action="store_true", help="Show result windows (if supported).")

    parser.add_argument("--save-dir", type=Path, default=None, help="If set, saves annotated outputs here.")
    parser.add_argument("--save-txt", action="store_true", help="If set, saves predicted boxes to .txt files.")
    parser.add_argument("--plot-conf", action="store_true", help="Overlay confidence values on saved images.")
    parser.add_argument("--class-id", type=int, default=None, help="Filter detections by class id.")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model weights not found: {args.model}")

    source = args.source or _pick_default_source(repo_root)
    if not source.exists():
        raise FileNotFoundError(f"Source not found: {source}")

    save_dir = args.save_dir
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    results = model.predict(
        source=str(source),
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        verbose=False,
    )

    for idx, r in enumerate(results[: args.max_images]):
        img_path = getattr(r, "path", None) or getattr(r, "im_file", None)
        img_stem = Path(str(img_path)).stem if img_path else f"image_{idx}"

        boxes = getattr(r, "boxes", None)
        dets: list[tuple[int | None, float | None, list[float]]] = []
        if boxes is not None and getattr(boxes, "xyxy", None) is not None:
            for i in range(len(boxes)):
                cls = int(boxes.cls[i].item()) if getattr(boxes, "cls", None) is not None else None
                conf = float(boxes.conf[i].item()) if getattr(boxes, "conf", None) is not None else None
                if args.class_id is not None and cls != args.class_id:
                    continue
                xyxy = boxes.xyxy[i].cpu().numpy()
                dets.append((cls, conf, [float(v) for v in xyxy]))

        print(f"{img_stem}: {len(dets)} detections (conf>={args.conf})")
        for j, (cls, conf, xyxy) in enumerate(dets[:10]):
            x1, y1, x2, y2 = xyxy
            conf_s = f"{conf:.3f}" if conf is not None else "n/a"
            cls_s = str(cls) if cls is not None else "n/a"
            print(f"  {j}: class={cls_s} conf={conf_s} xyxy=[{x1:.1f},{y1:.1f},{x2:.1f},{y2:.1f}]")

        if args.show:
            r.show()

        if save_dir:
            annotated = r.plot(conf=args.plot_conf, labels=True, boxes=True, show=False)
            out_img = save_dir / f"{img_stem}_pred.jpg"
            cv2.imwrite(str(out_img), annotated)

            if args.save_txt:
                out_txt = save_dir / f"{img_stem}.txt"
                r.save_txt(out_txt, save_conf=True)


if __name__ == "__main__":
    main()