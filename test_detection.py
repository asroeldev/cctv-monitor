import base64
import os
import tempfile
import unittest
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import cv2

import app
from detection import annotations, person_captures, violations


class DetectionTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(folder.cleanup)
        database = patch.object(app, "DATABASE", Path(folder.name) / "events.sqlite3")
        database.start()
        self.addCleanup(database.stop)
        app.init_db()

    def test_camera_reconnects_in_same_thread_and_releases_previous_capture(self):
        stop = threading.Event()
        camera = app.Camera(1, "rtsp://old", None, threading.Lock())
        old_capture, new_capture = Mock(), Mock()
        old_capture.open.return_value = new_capture.open.return_value = True
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        def old_read():
            camera.change_url("rtsp://new")
            return True, frame
        reads = 0
        def new_read():
            nonlocal reads
            reads += 1
            if reads == 1:
                return True, frame
            stop.set()
            return False, None
        old_capture.read.side_effect = old_read
        new_capture.read.side_effect = new_read
        with patch.object(app.cv2, "VideoCapture", side_effect=[old_capture, new_capture]):
            camera.run(stop)
        self.assertEqual(old_capture.open.call_args.args[0], "rtsp://old")
        self.assertEqual(new_capture.open.call_args.args[0], "rtsp://new")
        old_capture.release.assert_called_once()
        new_capture.release.assert_called_once()
        self.assertEqual(reads, 2)

    def test_prediction_from_previous_address_is_discarded(self):
        model = Mock()
        camera = app.Camera(1, "rtsp://old", model, threading.Lock())
        box = SimpleNamespace(cls=np.array(0), conf=np.array(0.9),
                              xyxy=np.array([[10, 30, 40, 60]]))
        def prediction(*args, **kwargs):
            camera.change_url("rtsp://new")
            return [SimpleNamespace(boxes=[box], names={0: "person"})]
        model.predict.side_effect = prediction
        camera.detect(np.zeros((80, 80, 3), dtype=np.uint8), 1, revision=0)
        self.assertIsNone(camera.people_count)
        self.assertEqual(camera.annotations, [])
        self.assertIsNone(camera.captures)

    def test_visible_count_uses_latest_people_above_threshold_not_cumulative_total(self):
        def box(class_id, score):
            return SimpleNamespace(cls=np.array(class_id), conf=np.array(score),
                                   xyxy=np.array([[10, 30, 40, 60]]))
        prediction = SimpleNamespace(
            boxes=[box(0, 0.9), box(0, 0.8), box(0, 0.1), box(1, 0.9)],
            names={0: "person", 1: "car"},
        )
        model = Mock()
        model.predict.return_value = [prediction]
        camera = app.Camera(1, "", model, threading.Lock())
        camera.status = "Terhubung"
        self.assertIsNone(camera.get_status()["people_count"])
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        camera.detect(frame, 1)
        camera.detect(frame, 2)
        self.assertEqual(camera.get_status()["people_count"], 2)
        self.assertEqual(len(camera.get_status()["captures"]["people"]), 2)
        prediction.boxes = []
        camera.detect(frame, 3)
        self.assertEqual(camera.get_status()["people_count"], 0)
        self.assertEqual(camera.get_status()["captures"]["people"], [])

    def test_person_crops_are_clipped_and_face_crop_is_from_the_same_original_image(self):
        frame = np.zeros((80, 100, 3), dtype=np.uint8)
        frame[5:25, 10:30] = (250, 0, 0)
        detector = Mock()
        detector.detectMultiScale.side_effect = [np.array([[10, 5, 20, 20]]), ()]
        result = person_captures(frame, [
            ("person", 0.9, [-10, -5, 70, 80]),
            ("person", 0.8, [80, 0, 110, 80]),
            ("car", 0.9, [0, 0, 100, 80]),
        ], detector)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]["face_status"], "detected")
        self.assertEqual(result[1]["face_status"], "not_detected")
        self.assertIsNone(result[1]["face"])
        head = cv2.imdecode(np.frombuffer(base64.b64decode(result[1]["head"]), np.uint8), 1)
        self.assertEqual(head.shape[:2], (40, 20))
        self.assertIsNone(result[0]["head"])
        person = cv2.imdecode(np.frombuffer(base64.b64decode(result[0]["image"]), np.uint8), 1)
        face = cv2.imdecode(np.frombuffer(base64.b64decode(result[0]["face"]), np.uint8), 1)
        self.assertEqual(person.shape[:2], (80, 70))
        self.assertEqual(face.shape[:2], (20, 20))
        self.assertGreater(face[:, :, 0].mean(), 240)
        self.assertLess(face[:, :, 1:].mean(), 5)

    def test_preview_sizes_are_bounded_and_face_coordinates_map_to_original_resolution(self):
        detector = Mock()
        detector.detectMultiScale.return_value = np.array([[64, 0, 128, 100]])
        result = person_captures(np.zeros((400, 800, 3), np.uint8),
                                 [("person", 0.9, [0, 0, 800, 400])], detector)[0]
        person = cv2.imdecode(np.frombuffer(base64.b64decode(result["image"]), np.uint8), 1)
        face = cv2.imdecode(np.frombuffer(base64.b64decode(result["face"]), np.uint8), 1)
        self.assertEqual(detector.detectMultiScale.call_args.args[0].shape, (128, 512))
        self.assertEqual(person.shape[:2], (256, 512))
        self.assertEqual(face.shape[:2], (125, 160))

    def test_multiple_faces_are_not_arbitrarily_assigned_to_a_person(self):
        detector = Mock()
        detector.detectMultiScale.return_value = np.array([[5, 5, 20, 20], [30, 5, 20, 20]])
        result = person_captures(np.zeros((80, 80, 3), np.uint8),
                                 [("person", 0.9, [0, 0, 80, 80])], detector)[0]
        self.assertIsNone(result["face"])
        self.assertEqual(result["face_status"], "multiple")

    def test_captures_sample_all_people_and_clear_when_they_leave_or_camera_changes(self):
        camera = app.Camera(1, "old", None, threading.Lock())
        camera.status = "Terhubung"
        camera.face_detector = Mock()
        camera.face_detector.detectMultiScale.return_value = ()
        frame = np.zeros((80, 80, 3), np.uint8)
        people = [("person", 0.9, [0, 0, 80, 80])]
        camera.capture_people(frame, people, 1, 1, 0)
        first = camera.captures
        camera.capture_people(frame, people, 1, 2, 0)
        self.assertIs(camera.captures, first)
        camera.capture_people(frame, people * 2, 2, 2, 0)
        self.assertEqual(len(camera.captures["people"]), 2)
        camera.capture_people(frame, [], 0, 2.2, 0)
        self.assertEqual(camera.captures["people"], [])
        camera.capture_people(frame, people, 1, 2.3, 0)
        self.assertEqual(len(camera.captures["people"]), 1)
        camera.change_url("new")
        self.assertIsNone(camera.captures)

    def test_capture_errors_are_visible_without_stopping_video_and_stale_captures_are_discarded(self):
        camera = app.Camera(1, "old", None, threading.Lock())
        camera.status = "Terhubung"
        frame = np.zeros((80, 80, 3), np.uint8)
        people = [("person", 0.9, [0, 0, 80, 80])]
        with patch.object(app, "load_face_detector", side_effect=OSError("missing")), \
                self.assertLogs("cctv", level="ERROR"):
            camera.capture_people(frame, people, 1, 1, 0)
        self.assertEqual(camera.status, "Terhubung")
        self.assertIsNotNone(camera.get_status()["capture_error"])
        camera.face_detector = Mock()
        def prepare(*args):
            camera.change_url("new")
            return [{"id": 1}]
        with patch.object(app, "person_captures", side_effect=prepare):
            camera.capture_people(frame, people, 1, 4, 0)
        self.assertIsNone(camera.captures)

    def test_count_is_unavailable_when_disconnected_or_person_class_is_absent(self):
        model = Mock()
        model.predict.return_value = [SimpleNamespace(boxes=[], names={0: "person"})]
        camera = app.Camera(1, "", model, threading.Lock())
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        camera.detect(frame, 1)
        self.assertIsNone(camera.get_status()["people_count"])
        camera.status = "Terhubung"
        model.predict.return_value = [SimpleNamespace(boxes=[], names={0: "no_mask"})]
        camera.detect(frame, 2)
        self.assertIsNone(camera.get_status()["people_count"])

    def test_annotations_include_people_without_reporting_them_as_violations(self):
        names = {0: "person", 1: "no_mask", 2: "no_hairnet", 3: "car"}
        detections = [
            (0, 0.9, [10, 30, 40, 60]), (1, 0.8, [12, 32, 20, 40]),
            (2, 0.4, [15, 30, 25, 40]), (3, 0.9, [1, 2, 3, 4]),
        ]
        self.assertEqual(annotations(detections, names, 0.5), [
            ("person", 0.9, [10, 30, 40, 60]), ("no_mask", 0.8, [12, 32, 20, 40]),
        ])
        self.assertEqual(violations(detections, names, 0.5), [
            ("no_mask", 0.8, [12, 32, 20, 40]),
        ])

    def test_boxes_persist_between_predictions_and_clear_when_person_leaves(self):
        box = SimpleNamespace(cls=np.array(0), conf=np.array(0.9),
                              xyxy=np.array([[10, 30, 40, 60]]))
        prediction = SimpleNamespace(boxes=[box], names={0: "person"})
        model = Mock()
        model.predict.return_value = [prediction]
        camera = app.Camera(1, "", model, threading.Lock())
        frame = np.zeros((80, 80, 3), dtype=np.uint8)
        camera.detect(frame, 1)
        self.assertFalse(frame.any())
        for _ in range(2):
            displayed = frame.copy()
            camera.draw_annotations(displayed)
            self.assertTrue(displayed.any())
        prediction.boxes = []
        camera.detect(frame, 2)
        camera.draw_annotations(frame)
        self.assertFalse(frame.any())
        self.assertEqual(camera.last_event, {})

    def test_person_only_mode_never_records_ppe_predictions(self):
        box = SimpleNamespace(cls=np.array(1), conf=np.array(0.9),
                              xyxy=np.array([[10, 30, 40, 60]]))
        model = Mock()
        model.predict.return_value = [SimpleNamespace(boxes=[box], names={0: "person", 1: "no_mask"})]
        camera = app.Camera(1, "", model, threading.Lock(), person_only=True)
        camera.detect(np.zeros((80, 80, 3), dtype=np.uint8), 1)
        self.assertEqual(camera.annotations, [])
        self.assertEqual(camera.last_event, {})

    def test_only_known_violations_above_threshold(self):
        names = {0: "no_mask", 1: "no_hairnet", 2: "person"}
        detections = [
            (0, 0.8, [1, 2, 3, 4]),
            (1, 0.5, [2, 3, 4, 5]),
            (1, 0.49, [3, 4, 5, 6]),
            (2, 0.99, [4, 5, 6, 7]),
            (99, 0.99, [5, 6, 7, 8]),
        ]
        self.assertEqual(violations(detections, names, 0.5), [
            ("no_mask", 0.8, [1, 2, 3, 4]),
            ("no_hairnet", 0.5, [2, 3, 4, 5]),
        ])


if __name__ == "__main__":
    unittest.main()
