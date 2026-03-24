"""Label preprocessing: merge all Truck variant classes into a single Truck class.

Original dataset has 13 classes with 5 separate Truck axle variants (IDs 8-12).
This script remaps all those IDs to a single ``Truck`` label (ID 8), producing
9 classes total.  Label files are modified in-place; a backup is created first.

Usage:
    source .venv/bin/activate
    python preprocess.py
"""

import pathlib
import shutil

# ── Paths ──────────────────────────────────────────────────────────────────────

_PROJECT_ROOT: pathlib.Path = pathlib.Path(__file__).resolve().parent

DATASET_DIR: pathlib.Path = (
    _PROJECT_ROOT / "Indian Vehicle Classification.v3i.yolov11"
)
BACKUP_DIR: pathlib.Path = (
    _PROJECT_ROOT / "Indian Vehicle Classification.v3i.yolov11_labels_backup"
)

SPLITS: list[str] = ["train", "valid", "test"]

# ── Class definitions ──────────────────────────────────────────────────────────

# All five Truck axle variants collapse to the same new ID.
# Every other class keeps its original ID unchanged.
CLASS_REMAP: dict[int, int] = {
    0:  0,   # Auto Rickshaw  → Auto Rickshaw
    1:  1,   # Bus -2 axle    → Bus -2 axle
    2:  2,   # Bus -3 axle    → Bus -3 axle
    3:  3,   # Car            → Car
    4:  4,   # E-Rickshaw     → E-Rickshaw
    5:  5,   # Mini Loading   → Mini Loading
    6:  6,   # Motor Cycle    → Motor Cycle
    7:  7,   # Scooty         → Scooty
    8:  8,   # Truck - 2 axle → Truck
    9:  8,   # Truck - 3 axle → Truck
    10: 8,   # Truck - 4 axle → Truck
    11: 8,   # Truck - 5 axle → Truck
    12: 8,   # Truck - 6 axle → Truck
}

NEW_CLASSES: list[str] = [
    "Auto Rickshaw",
    "Bus -2 axle",
    "Bus -3 axle",
    "Car",
    "E-Rickshaw",
    "Mini Loading",
    "Motor Cycle",
    "Scooty",
    "Truck",
]


# ── Helpers ────────────────────────────────────────────────────────────────────

def backup_labels() -> None:
    """Copy original label folders to a backup directory before modifying.

    Skips backup creation if the backup directory already exists so that
    re-running the script never overwrites a previous backup.
    """
    if BACKUP_DIR.exists():
        print(f"[backup]  already exists — skipping  ({BACKUP_DIR.name})")
        return

    BACKUP_DIR.mkdir(parents=True)
    for split in SPLITS:
        src = DATASET_DIR / split / "labels"
        dst = BACKUP_DIR / split / "labels"
        if src.exists():
            shutil.copytree(src, dst)
            print(f"[backup]  {split}/labels  →  backup")

    print(f"[backup]  complete → {BACKUP_DIR}")


def remap_label_file(path: pathlib.Path) -> int:
    """Rewrite a single YOLO label file with remapped class IDs.

    Each line has the format ``class_id x y w h``.  Only the class_id
    column is changed; bounding-box coordinates are preserved exactly.

    Args:
        path (pathlib.Path): Path to the ``.txt`` label file.

    Returns:
        int: Number of annotations that were remapped (changed).
    """
    lines = path.read_text().splitlines()
    new_lines: list[str] = []
    remapped = 0

    for line in lines:
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        old_id = int(parts[0])
        new_id = CLASS_REMAP.get(old_id, old_id)
        if new_id != old_id:
            remapped += 1
        new_lines.append(" ".join([str(new_id)] + parts[1:]))

    path.write_text("\n".join(new_lines) + "\n")
    return remapped


def remap_split(split: str) -> tuple[int, int]:
    """Remap all label files in one dataset split.

    Args:
        split (str): One of ``"train"``, ``"valid"``, or ``"test"``.

    Returns:
        tuple[int, int]: ``(files_processed, annotations_remapped)``.
    """
    labels_dir = DATASET_DIR / split / "labels"
    if not labels_dir.exists():
        print(f"[{split}]  labels dir not found — skipping")
        return 0, 0

    label_files = list(labels_dir.glob("*.txt"))
    total_remapped = 0

    for lf in label_files:
        total_remapped += remap_label_file(lf)

    return len(label_files), total_remapped


# ── Main ───────────────────────────────────────────────────────────────────────

def run_preprocessing() -> None:
    """Execute the full preprocessing pipeline and print a summary."""
    print("=" * 60)
    print("Label preprocessing — merging Truck variants → Truck")
    print("=" * 60)

    backup_labels()
    print()

    grand_files = 0
    grand_remapped = 0

    for split in SPLITS:
        files, remapped = remap_split(split)
        grand_files += files
        grand_remapped += remapped
        print(
            f"[{split:<6}]  {files:>5} label files  |  "
            f"{remapped:>6} annotations remapped"
        )

    print()
    print(f"Total files     : {grand_files}")
    print(f"Total remapped  : {grand_remapped} annotations")
    print()
    print("New class list  :")
    for i, name in enumerate(NEW_CLASSES):
        print(f"  {i:>2} : {name}")
    print()
    print("Preprocessing complete.")
    print("=" * 60)


if __name__ == "__main__":
    run_preprocessing()
