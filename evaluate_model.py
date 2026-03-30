from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


def _import_yolo():
    try:
        from ultralytics import YOLO  # type: ignore

        return YOLO
    except ModuleNotFoundError:
        # Attempt to import ultralytics from the local `venv/` folder.
        repo_root = Path(__file__).resolve().parent
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


def _count_images(img_dir: Path) -> int:
    if not img_dir.exists() or not img_dir.is_dir():
        return 0
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
    return sum(1 for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in exts)


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    default_model = repo_root / "detect-plate" / "runs" / "detect" / "train" / "weights" / "best.pt"
    default_data = repo_root / "detect-plate" / "data.yaml"

    parser = argparse.ArgumentParser(description="Evaluate YOLO model effectiveness using Ultralytics val().")
    parser.add_argument("--model", type=Path, default=default_model, help="Path to YOLO weights (.pt).")
    parser.add_argument("--data", type=Path, default=default_data, help="Path to YOLO dataset YAML.")
    parser.add_argument(
        "--split",
        choices=["val", "test", "train"],
        default="val",
        help="Dataset split to evaluate.",
    )
    parser.add_argument("--imgsz", type=int, default=640, help="Validation image size.")
    parser.add_argument("--plots", action="store_true", help="Save validation plots (confusion matrix, curves, etc).")
    parser.add_argument("--verbose", action="store_true", help="Print more Ultralytics logs.")
    args = parser.parse_args()

    if not args.model.exists():
        raise FileNotFoundError(f"Model weights not found: {args.model}")
    if not args.data.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {args.data}")

    data_cfg = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    split_key = args.split  # YOLO expects 'val', 'test', 'train' in `split=...`
    if split_key not in data_cfg:
        raise KeyError(f"YAML file {args.data} does not define '{split_key}'. Found keys: {sorted(data_cfg.keys())}")

    # data_cfg values are relative to the YAML directory.
    # Example: val: valid/images
    img_rel = str(data_cfg[split_key])
    img_dir = args.data.parent / img_rel
    n_images = _count_images(img_dir)
    if n_images == 0:
        print(f"Split '{split_key}' has no images at: {img_dir}")
        print("Nothing to evaluate. (Check your dataset paths / ensure images exist.)")
        return

    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data),
        split=split_key,
        imgsz=args.imgsz,
        plots=args.plots,
        verbose=args.verbose,
    )

    # `metrics` is a DetMetrics instance with `results_dict`.
    results = getattr(metrics, "results_dict", {})

    def g(key: str) -> str:
        if key not in results:
            return "n/a"
        return f"{results[key]:.6f}"

    print(f"Effective evaluation on split='{split_key}' (images={n_images})")
    print(f"Precision (B): {g('metrics/precision(B)')}")
    print(f"Recall (B):    {g('metrics/recall(B)')}")
    print(f"mAP50 (B):     {g('metrics/mAP50(B)')}")
    print(f"mAP50-95 (B):  {g('metrics/mAP50-95(B)')}")


if __name__ == "__main__":
    main()

