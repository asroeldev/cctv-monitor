import os
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from train_model import CLASSES, validate_dataset


class TrainingDatasetTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(folder.cleanup)
        self.root = Path(folder.name)
        self.config = self.root / "dataset.yaml"
        self.config.write_text(yaml.safe_dump({
            "path": "dataset", "train": "images/train", "val": "images/val", "names": CLASSES,
        }))
        for split in ("train", "val"):
            images = self.root / "dataset" / "images" / split
            labels = self.root / "dataset" / "labels" / split
            images.mkdir(parents=True)
            labels.mkdir(parents=True)
            cv2.imwrite(str(images / "sample.png"), np.zeros((64, 64, 3), dtype=np.uint8))
            (labels / "sample.txt").write_text(
                "0 0.5 0.5 0.8 0.8\n1 0.5 0.3 0.2 0.2\n2 0.5 0.2 0.2 0.2\n"
            )

    def test_valid_dataset_uses_config_relative_absolute_root(self):
        result = validate_dataset(self.config)
        self.assertEqual(result["path"], str((self.root / "dataset").resolve()))

    def test_missing_labels_are_not_silently_treated_as_background(self):
        (self.root / "dataset" / "labels" / "train" / "sample.txt").unlink()
        with self.assertRaisesRegex(ValueError, "Missing label"):
            validate_dataset(self.config)

    def test_invalid_boxes_and_missing_classes_are_rejected(self):
        label = self.root / "dataset" / "labels" / "train" / "sample.txt"
        for text, message in [
            ("0 nan 0.5 0.2 0.2\n", "invalid class or normalized"),
            ("0 0.5 0.5 0 0.2\n", "must be positive"),
            ("0 0.5 0.5 0.2 0.2\n", "all three classes"),
        ]:
            with self.subTest(text=text):
                label.write_text(text)
                with self.assertRaisesRegex(ValueError, message):
                    validate_dataset(self.config)


if __name__ == "__main__":
    unittest.main()
