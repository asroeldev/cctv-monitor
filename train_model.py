import argparse
import logging
import math
from pathlib import Path

import cv2
import yaml

from detection import load_model

CLASSES = {0: "person", 1: "no_mask", 2: "no_hairnet"}
ROOT = Path(__file__).resolve().parent
logger = logging.getLogger("training")


def validate_dataset(config_path):
    config_path = Path(config_path).resolve()
    with config_path.open() as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict) or config.get("names") != CLASSES:
        raise ValueError("Dataset names must be 0: person, 1: no_mask, 2: no_hairnet")
    if not isinstance(config.get("path"), str):
        raise ValueError("Dataset path must be a local directory")
    root = (config_path.parent / config["path"]).resolve()
    directories = []
    for split in ("train", "val"):
        if config.get(split) != f"images/{split}":
            raise ValueError(f"{split} must be images/{split}")
        image_dir = root / "images" / split
        label_dir = root / "labels" / split
        if not image_dir.is_dir() or not label_dir.is_dir():
            raise ValueError(f"Missing local images/labels directory for {split}")
        directories.append(image_dir.resolve())
        images = sorted(path for path in image_dir.iterdir()
                        if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not images:
            raise ValueError(f"No training images in {image_dir}")
        seen_stems = set()
        found_classes = set()
        for image in images:
            if image.stem in seen_stems:
                raise ValueError(f"Duplicate image stem in {split}: {image.stem}")
            seen_stems.add(image.stem)
            if cv2.imread(str(image)) is None:
                raise ValueError(f"Unreadable image: {image}")
            label = label_dir / f"{image.stem}.txt"
            if not label.is_file():
                raise ValueError(f"Missing label: {label}; use an empty file for background images")
            for line_number, line in enumerate(label.read_text().splitlines(), 1):
                fields = line.split()
                if len(fields) != 5:
                    raise ValueError(f"{label}:{line_number}: expected class x_center y_center width height")
                class_id = int(fields[0])
                values = [float(value) for value in fields[1:]]
                if class_id not in CLASSES or not all(math.isfinite(value) and 0 <= value <= 1 for value in values):
                    raise ValueError(f"{label}:{line_number}: invalid class or normalized coordinates")
                if values[2] <= 0 or values[3] <= 0:
                    raise ValueError(f"{label}:{line_number}: box width and height must be positive")
                found_classes.add(class_id)
        if found_classes != set(CLASSES):
            raise ValueError(f"{split} needs labeled examples of all three classes")
    if directories[0] == directories[1]:
        raise ValueError("Training and validation directories must be separate")
    # Resolve relative dataset roots explicitly before passing them to Ultralytics.
    config["path"] = str(root)
    return config


def positive_int(value):
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("Must be a positive integer")
    return result


def main():
    parser = argparse.ArgumentParser(description="Train local person, no_mask, and no_hairnet detection")
    parser.add_argument("--data", type=Path, default=ROOT / "dataset.example.yaml")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--epochs", type=positive_int, default=50)
    parser.add_argument("--imgsz", type=positive_int, default=640)
    parser.add_argument("--batch", type=positive_int, default=4)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    try:
        config = validate_dataset(args.data)
    except (OSError, ValueError, yaml.YAMLError) as error:
        parser.error(str(error))
    if args.check_only:
        logger.info("Dataset structure and labels validated; accuracy has not been evaluated")
        return
    output = ROOT / "models"
    output.mkdir(exist_ok=True)
    resolved_config = output / "dataset.resolved.yaml"
    with resolved_config.open("w") as file:
        yaml.safe_dump(config, file)
    model = load_model(args.model)
    model.train(data=str(resolved_config), epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=args.device, project=str(output),
                name="ppe", exist_ok=False, workers=0)
    weights = Path(model.trainer.best)
    if not weights.is_file():
        raise RuntimeError("Training did not produce best.pt")
    logger.info("Training completed. Evaluate held-out videos before use. MODEL_PATH=%s", weights)


if __name__ == "__main__":
    main()
