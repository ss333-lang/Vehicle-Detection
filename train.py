"""Training script for YOLOv11m on the Indian Vehicle Classification dataset.

Fine-tunes the pretrained YOLO11m weights on 9 Indian vehicle categories.
Truck axle variants are merged into a single Truck class by the preprocessing
step before training begins.
Writes training artefacts (weights, plots, metrics) to ``runs/``.

Usage:
    source .venv/bin/activate
    python train.py
"""

import pathlib

import torch
import yaml
from ultralytics import YOLO

from preprocess import NEW_CLASSES, run_preprocessing

# ── Paths ──────────────────────────────────────────────────────────────────────

_PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent

DATASET_DIR: pathlib.Path = (
    _PROJECT_ROOT / "Indian Vehicle Classification.v3i.yolov11"
)
WEIGHTS_PATH: pathlib.Path = _PROJECT_ROOT / "yolo11m.pt"  # official pretrained base
FIXED_YAML: pathlib.Path = _PROJECT_ROOT / "data_train.yaml"
RUNS_DIR: pathlib.Path = _PROJECT_ROOT / "runs"

# ── Hyperparameters ────────────────────────────────────────────────────────────

EPOCHS: int = 200        # Enough epochs for YOLOv11m to converge on custom data.
IMG_SIZE: int = 640
BATCH_SIZE: int = 16
PATIENCE: int = 30       # Early-stop if val mAP stalls for this many epochs.
WORKERS: int = 4
LR0: float = 0.01        # Initial learning rate.
LRF: float = 0.01        # Final lr as a fraction of LR0.
MOMENTUM: float = 0.937
WEIGHT_DECAY: float = 5e-4
WARMUP_EPOCHS: int = 3
RUN_NAME: str = "vehicle_detection_v2"


# ── Helpers ────────────────────────────────────────────────────────────────────

def fix_data_yaml() -> pathlib.Path:
    """Create a corrected data.yaml with absolute image paths.

    The Roboflow-exported yaml uses ``../train/images`` which resolves
    incorrectly when the working directory is the project root.  This
    function writes a clean copy with absolute paths that work regardless
    of where the script is invoked.

    Returns:
        pathlib.Path: Path to the corrected yaml file.
    """
    src_yaml = DATASET_DIR / "data.yaml"
    with open(src_yaml) as fh:
        cfg: dict = yaml.safe_load(fh)

    cfg["train"] = str(DATASET_DIR / "train" / "images")
    cfg["val"] = str(DATASET_DIR / "valid" / "images")
    cfg["test"] = str(DATASET_DIR / "test" / "images")

    # Use the merged class list (Truck variants collapsed into one).
    cfg["names"] = NEW_CLASSES
    cfg["nc"] = len(NEW_CLASSES)

    with open(FIXED_YAML, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)

    print(f"[data.yaml]  written → {FIXED_YAML}")
    return FIXED_YAML


def select_device() -> str:
    """Return the best available compute device for this machine.

    Returns:
        str: ``"mps"`` on Apple Silicon, ``"cuda"`` on NVIDIA GPU,
            or ``"cpu"`` as a fallback.
    """
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full training pipeline and print the best-weights path."""
    # Step 1: merge Truck variant labels (skipped if backup already exists).
    run_preprocessing()
    print()

    device = select_device()
    print(f"[device]     {device}")
    print(f"[weights]    {WEIGHTS_PATH}")
    print(f"[dataset]    {DATASET_DIR.name}")
    print(f"[classes]    {len(NEW_CLASSES)} classes after Truck merge")
    print(f"[epochs]     {EPOCHS}  |  batch {BATCH_SIZE}  |  imgsz {IMG_SIZE}")
    print("-" * 60)

    data_yaml = fix_data_yaml()
    model = YOLO(str(WEIGHTS_PATH))

    results = model.train(
        data=str(data_yaml),
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        workers=WORKERS,
        device=device,
        project=str(RUNS_DIR),
        name=RUN_NAME,
        exist_ok=True,
        resume=False,
        optimizer="auto",
        lr0=LR0,
        lrf=LRF,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        val=True,
        plots=True,
        verbose=True,
        half=True,
        # Augmentation tuned for dense Indian traffic scenes.
        mosaic=1.0,        # Tile 4 images — critical for overlapping vehicles.
        mixup=0.1,         # Blend two images — helps with occlusion.
        copy_paste=0.1,    # Copy-paste vehicles across scenes.
        degrees=5.0,       # Slight rotation for camera tilt variation.
        translate=0.1,     # Random translation.
        scale=0.5,         # Scale jitter for near/far vehicles.
        hsv_h=0.015,       # Hue jitter.
        hsv_s=0.7,         # Saturation jitter — handles shadow/sunlight.
        hsv_v=0.4,         # Value jitter — handles overcast/bright days.
        fliplr=0.5,        # Horizontal flip.
    )

    best_weights = pathlib.Path(results.save_dir) / "weights" / "best.pt"
    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Best weights : {best_weights}")
    print(f"Results dir  : {results.save_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
