"""Interpret the explicit violation classes provided by a trained model."""

import base64
import os
from pathlib import Path

import cv2

VIOLATIONS = {
    "no_mask": "Tidak memakai masker",
    "no_hairnet": "Tidak memakai hairnet",
}


def person_crop(frame, coordinates):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = map(int, coordinates)
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(width, x2), min(height, y2)
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Kotak orang tidak valid untuk tangkapan gambar")
    return frame[y1:y2, x1:x2], (x1, y1, x2, y2)


def jpeg_bytes(image, max_side=512):
    scale = min(1, max_side / max(image.shape[:2]))
    if scale < 1:
        image = cv2.resize(image, (max(1, round(image.shape[1] * scale)),
                                   max(1, round(image.shape[0] * scale))))
    ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise OSError("Tidak dapat menyiapkan tangkapan JPEG")
    return encoded.tobytes()


def load_face_detector():
    path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(str(path))
    if detector.empty():
        raise OSError("Model deteksi wajah lokal tidak tersedia")
    return detector


def person_captures(frame, detections, face_detector):
    people = []
    for kind, score, coordinates in detections:
        if kind != "person":
            continue
        crop, _ = person_crop(frame, coordinates)
        head = crop[:max(1, crop.shape[0] // 2)]
        gray = cv2.cvtColor(head, cv2.COLOR_BGR2GRAY)
        scale = min(1, 512 / max(gray.shape[:2]))
        if scale < 1:
            gray = cv2.resize(gray, (max(1, round(gray.shape[1] * scale)),
                                    max(1, round(gray.shape[0] * scale))))
        faces = face_detector.detectMultiScale(cv2.equalizeHist(gray), scaleFactor=1.1,
                                              minNeighbors=5, minSize=(20, 20))
        face_image = None
        if len(faces) == 1:
            x, y, width, height = faces[0]
            x_ratio, y_ratio = head.shape[1] / gray.shape[1], head.shape[0] / gray.shape[0]
            face, _ = person_crop(head, (x * x_ratio, y * y_ratio,
                                        (x + width) * x_ratio, (y + height) * y_ratio))
            face_image = base64.b64encode(jpeg_bytes(face, max_side=160)).decode("ascii")
        people.append({
            "id": len(people) + 1, "score": score,
            "image": base64.b64encode(jpeg_bytes(crop)).decode("ascii"),
            "face": face_image,
            "head": None if face_image else base64.b64encode(jpeg_bytes(head, max_side=160)).decode("ascii"),
            "face_status": "detected" if face_image else "multiple" if len(faces) > 1 else "not_detected",
        })
    return people


def load_model(path):
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(__file__).resolve().parent / "data" / "ultralytics"))
    from ultralytics import YOLO, settings

    integrations = ("sync", "clearml", "comet", "dvc", "mlflow", "neptune", "raytune", "wandb")
    settings.update({name: False for name in integrations if name in settings})
    return YOLO(str(path))


def annotations(detections, names, confidence):
    return [
        (names[class_id], score, coordinates)
        for class_id, score, coordinates in detections
        if names.get(class_id) in {*VIOLATIONS, "person"} and score >= confidence
    ]


def violations(detections, names, confidence):
    results = []
    for detection in detections:
        class_id, score, coordinates = detection
        name = names.get(class_id, "")
        if name in VIOLATIONS and score >= confidence:
            results.append((name, score, coordinates))
    return results
