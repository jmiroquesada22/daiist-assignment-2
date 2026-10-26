"""
Download the Safety Vests dataset from Roboflow Universe and describe it.

    uv run python dataset.py

Downloads into datasets/ (git-ignored) and prints the composition stats the
report needs: images per split, instances per class per split, how many images
contain the positive class, and object size relative to the image.

train.py imports `download_dataset()` from here, so the dataset is fetched the
same way in both places.
"""

from collections import Counter
from pathlib import Path

import yaml

from api_key import load_api_key

# Exact dataset identity -- must match what REPORT.md records.
WORKSPACE = "roboflow-universe-projects"
PROJECT = "safety-vests"
VERSION = 14  # check on the Versions tab, change if you pick another

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "datasets" / f"{PROJECT}-v{VERSION}"
SPLITS = ("train", "valid", "test")


def download_dataset() -> Path:
    """Download the dataset once (skipped if already there) and return the path
    to a data.yaml with absolute split paths, ready for Ultralytics."""
    yaml_path = DATA_DIR / "data.yaml"
    if not yaml_path.exists():
        from roboflow import Roboflow

        rf = Roboflow(api_key=load_api_key())
        version = rf.workspace(WORKSPACE).project(PROJECT).version(VERSION)
        version.download("yolov11", location=str(DATA_DIR))

    # Roboflow writes relative paths that depend on the working directory.
    # Rewrite them as absolute paths so training works from anywhere.
    config = yaml.safe_load(yaml_path.read_text())
    config["path"] = str(DATA_DIR)
    for split, key in (("train", "train"), ("valid", "val"), ("test", "test")):
        if (DATA_DIR / split / "images").is_dir():
            config[key] = f"{split}/images"
    yaml_path.write_text(yaml.safe_dump(config, sort_keys=False))
    return yaml_path


def class_names(yaml_path: Path) -> list[str]:
    names = yaml.safe_load(yaml_path.read_text())["names"]
    return list(names.values()) if isinstance(names, dict) else list(names)


def describe(yaml_path: Path) -> None:
    names = class_names(yaml_path)
    print(f"Dataset: {WORKSPACE}/{PROJECT}/{VERSION}")
    print(f"Classes: {names}\n")

    header = f"{'split':<6} {'images':>7} {'empty':>6} " + " ".join(
        f"{n:>16}" for n in names
    ) + " " + " ".join(f"{'imgs w/ ' + n:>24}" for n in names)
    print(header)

    areas: dict[str, list[float]] = {n: [] for n in names}
    for split in SPLITS:
        label_dir = DATA_DIR / split / "labels"
        if not label_dir.is_dir():
            print(f"{split:<6} (missing)")
            continue
        image_dir = DATA_DIR / split / "images"
        n_images = sum(1 for _ in image_dir.iterdir())
        instances: Counter = Counter()
        images_with: Counter = Counter()
        empty = 0
        for label_file in label_dir.glob("*.txt"):
            rows = [r.split() for r in label_file.read_text().splitlines() if r.strip()]
            if not rows:
                empty += 1
            seen = set()
            for row in rows:
                cls = names[int(row[0])]
                instances[cls] += 1
                seen.add(cls)
                # YOLO boxes are normalised, so w*h is the share of the image.
                areas[cls].append(float(row[3]) * float(row[4]))
            for cls in seen:
                images_with[cls] += 1
        print(
            f"{split:<6} {n_images:>7} {empty:>6} "
            + " ".join(f"{instances[n]:>16}" for n in names)
            + " "
            + " ".join(f"{images_with[n]:>24}" for n in names)
        )

    print("\nBox area as a share of the image (all splits):")
    for n in names:
        a = sorted(areas[n])
        if not a:
            continue
        median = a[len(a) // 2]
        small = sum(x < 0.01 for x in a) / len(a)
        print(
            f"  {n:<16} median {median:6.2%}   "
            f"under 1% of the image: {small:5.1%}   (n={len(a)})"
        )


if __name__ == "__main__":
    describe(download_dataset())
