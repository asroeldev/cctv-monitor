import base64
import json
import os
import tempfile
import threading
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import cv2

import app
from gemini_vision import GeminiClient, GeminiWorker, VisionError


def response(results, finish="STOP"):
    return json.dumps({"candidates": [{
        "finishReason": finish,
        "content": {"parts": [{"text": json.dumps(results)}]},
    }]}).encode()


class GeminiTests(unittest.TestCase):
    def test_gemini_event_persists_with_source_and_original_person_score(self):
        client = Mock(model="gemini-3.5-flash-lite")
        client.assess.return_value = [
            {"id": 1, "mask": "missing", "hairnet": "unclear"},
            {"id": 2, "mask": "missing", "hairnet": "unclear"},
            {"id": 3, "mask": "worn", "hairnet": "worn"},
            {"id": 4, "mask": "unclear", "hairnet": "unclear"},
        ]
        worker = GeminiWorker(client, threading.Event())
        camera = app.Camera(1, "", None, threading.Lock())
        with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as folder:
            data = Path(folder)
            (data / "snapshots").mkdir()
            with patch.object(app, "DATA", data), patch.object(app, "DATABASE", data / "events.sqlite3"), \
                    patch.object(app, "COOLDOWN", 30):
                app.init_db()
                worker.submit(camera, np.zeros((100, 100, 3), dtype=np.uint8),
                              [("person", score, [10, 10, 90, 90]) for score in (0.87, 0.8, 0.79, 0.78)],
                              revision=0)
                job = worker.jobs.get_nowait()
                worker.process(job)
                events = app.events(camera=1)
                self.assertEqual(len(events), 1)
                self.assertEqual(events[0]["kind"], "no_mask")
                self.assertEqual(events[0]["source"], "gemini")
                self.assertEqual(events[0]["confidence"], 0.87)
                self.assertEqual(len(list((data / "snapshots").glob("*.jpg"))), 1)
                history = app.capture_history(camera=1, kind="no_mask")
                self.assertEqual(history["total"], 2)
                self.assertEqual([person["score"] for person in history["people"]], [0.8, 0.87])
                self.assertTrue(all(person["source"] == "gemini" for person in history["people"]))
                dataset = app.dataset_samples(state="pending", offset=0)
                self.assertEqual(dataset["pending"], 4)
                self.assertEqual(dataset["reviewed"], 0)
                self.assertEqual([row["suggested_mask"] for row in dataset["samples"]],
                                 ["missing", "missing", "worn", "unclear"])
                self.assertEqual(app.dataset_image(dataset["samples"][0]["id"]).body, job[4][0][1])
                self.assertEqual(dataset["samples"][0]["person_score"], 0.87)
                self.assertEqual(dataset["samples"][0]["model"], "gemini-3.5-flash-lite")

    def test_request_uses_server_key_and_validates_equipment_results(self):
        client = GeminiClient("test-key")
        results = [{"id": 1, "mask": "worn", "hairnet": "unclear"}]
        with patch("gemini_vision.urllib.request.urlopen") as transport:
            transport.return_value.__enter__.return_value.read.return_value = response(results)
            self.assertEqual(client.assess([(1, b"synthetic-image")]), results)
            request = transport.call_args.args[0]
            self.assertNotIn("test-key", request.full_url)
            self.assertEqual(request.get_header("X-goog-api-key"), "test-key")
            payload = json.loads(request.data)
            self.assertEqual(payload["generationConfig"]["responseMimeType"], "application/json")

    def test_request_includes_all_images_with_sufficient_output_budget(self):
        client = GeminiClient("test-key")
        images = [(i, b"synthetic-image") for i in range(1, 11)]
        results = [{"id": i, "mask": "unclear", "hairnet": "unclear"} for i in range(1, 11)]
        with patch("gemini_vision.urllib.request.urlopen") as transport:
            transport.return_value.__enter__.return_value.read.return_value = response(results)
            self.assertEqual(client.assess(images), results)
            payload = json.loads(transport.call_args.args[0].data)
            parts = payload["contents"][0]["parts"]
            self.assertEqual(sum("inlineData" in part for part in parts), 10)
            self.assertGreaterEqual(payload["generationConfig"]["maxOutputTokens"], 10 * 128)

    def test_invalid_incomplete_and_refused_responses_fail_explicitly(self):
        client = GeminiClient("test-key")
        invalid = [
            response([{"id": 2, "mask": "worn", "hairnet": "worn"}]),
            response([{"id": 1, "mask": "yes", "hairnet": "worn"}]),
            response([]),
            response([{"id": 1, "mask": "worn", "hairnet": "worn"}], finish="SAFETY"),
            b'{"candidates": []}',
        ]
        for body in invalid:
            with self.subTest(body=body), patch("gemini_vision.urllib.request.urlopen") as transport:
                transport.return_value.__enter__.return_value.read.return_value = body
                with self.assertRaises(VisionError):
                    client.assess([(1, b"synthetic-image")])

    def test_http_failure_does_not_expose_response_or_key(self):
        client = GeminiClient("test-key")
        error = urllib.error.HTTPError("https://example.test", 429, "private provider detail", {}, None)
        with patch("gemini_vision.urllib.request.urlopen", side_effect=error):
            with self.assertRaisesRegex(VisionError, "HTTP 429") as raised:
                client.assess([(1, b"synthetic-image")])
        self.assertNotIn("test-key", str(raised.exception))
        self.assertNotIn("private provider detail", str(raised.exception))

    def test_single_job_includes_all_people_and_provides_only_no_mask_thumbnails(self):
        client = Mock(model="gemini-3.5-flash-lite")
        client.assess.return_value = [
            {"id": 1, "mask": "missing", "hairnet": "unclear"},
            {"id": 2, "mask": "worn", "hairnet": "missing"},
            {"id": 3, "mask": "unclear", "hairnet": "unclear"},
            {"id": 4, "mask": "missing", "hairnet": "worn"},
        ]
        worker = GeminiWorker(client, threading.Event())
        camera = app.Camera(1, "", None, threading.Lock())
        camera.save_event = Mock()
        camera.save_missing_people = Mock(return_value=True)
        camera.save_dataset_results = Mock(return_value=True)
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        people = [("person", score, [10, 10, 90, 90]) for score in (0.9, 0.8, 0.79, 0.78)]
        worker.submit(camera, frame, people, revision=0)
        worker.submit(camera, frame, people, revision=0)
        self.assertEqual(worker.jobs.qsize(), 1)
        job = worker.jobs.get_nowait()
        self.assertEqual(len(job[4]), 4)
        worker.process(job)
        self.assertEqual([call.args[0] for call in camera.save_event.call_args_list],
                         ["no_mask", "no_hairnet", "no_mask"])
        self.assertEqual([call.args[1] for call in camera.save_event.call_args_list], [0.9, 0.8, 0.78])
        self.assertTrue(all(call.kwargs["source"] == "gemini" for call in camera.save_event.call_args_list))
        saved = camera.save_missing_people.call_args.args[1]
        self.assertEqual([person[1] for person in saved], [0.9, 0.78])
        self.assertEqual(worker.last_result["total"], 4)
        self.assertEqual(worker.last_result["checked"], 4)
        people = worker.last_result["people"]
        self.assertEqual([person["id"] for person in people if "image" in person], [1, 4])
        crop = cv2.imdecode(np.frombuffer(base64.b64decode(people[0]["image"]), dtype=np.uint8),
                            cv2.IMREAD_COLOR)
        self.assertEqual(crop.shape[:2], (80, 80))

    def test_only_people_strictly_above_seventy_percent_are_submitted_and_coverage_is_honest(self):
        client = Mock(model="gemini-3.5-flash-lite")
        client.assess.return_value = [
            {"id": 1, "mask": "worn", "hairnet": "unclear"},
            {"id": 2, "mask": "worn", "hairnet": "unclear"},
        ]
        worker = GeminiWorker(client, threading.Event())
        camera = app.Camera(1, "", None, threading.Lock())
        frame = np.zeros((100, 100, 3), np.uint8)
        camera.save_dataset_results = Mock(return_value=True)
        detections = [("person", score, [10, 10, 90, 90]) for score in (0.69, 0.7, 0.7001, 0.9)]
        detections.append(("car", 0.99, [10, 10, 90, 90]))
        worker.submit(camera, frame, detections, revision=0)
        job = worker.jobs.get_nowait()
        self.assertEqual(len(job[4]), 2)
        self.assertEqual([score for score, _ in job[5].values()], [0.9, 0.7001])
        worker.process(job)
        self.assertEqual(worker.last_result["checked"], 2)
        self.assertEqual(worker.last_result["total"], 4)

    def test_no_qualifying_people_does_not_encode_send_or_consume_interval(self):
        client = Mock(model="gemini-3.5-flash-lite")
        worker = GeminiWorker(client, threading.Event())
        camera = app.Camera(1, "", None, threading.Lock())
        frame = np.zeros((100, 100, 3), np.uint8)
        with patch("gemini_vision.jpeg_bytes") as encode:
            worker.submit(camera, frame, [("person", 0.7, [10, 10, 90, 90])], revision=0)
            encode.assert_not_called()
        self.assertTrue(worker.jobs.empty())
        self.assertFalse(worker.busy)
        client.assess.assert_not_called()
        self.assertIn("tidak ada gambar dikirim", worker.status)
        worker.submit(camera, frame, [("person", 0.71, [10, 10, 90, 90])], revision=0)
        self.assertEqual(worker.jobs.qsize(), 1)

    def test_camera_change_discards_pending_job_without_cloud_call(self):
        client = Mock(model="gemini-3.5-flash-lite")
        worker = GeminiWorker(client, threading.Event())
        camera = app.Camera(1, "", None, threading.Lock())
        worker.submit(camera, np.zeros((100, 100, 3), dtype=np.uint8),
                      [("person", 0.9, [10, 10, 90, 90])], revision=0)
        camera.change_url("rtsp://new")
        worker.process(worker.jobs.get_nowait())
        client.assess.assert_not_called()
        self.assertIsNone(worker.last_result)

    def test_camera_change_during_api_call_discards_result(self):
        client = Mock(model="gemini-3.5-flash-lite")
        worker = GeminiWorker(client, threading.Event())
        camera = app.Camera(1, "", None, threading.Lock())
        camera.save_event = Mock()
        def assess(images):
            camera.change_url("rtsp://new")
            return [{"id": 1, "mask": "missing", "hairnet": "missing"}]
        client.assess.side_effect = assess
        worker.submit(camera, np.zeros((100, 100, 3), dtype=np.uint8),
                      [("person", 0.9, [10, 10, 90, 90])], revision=0)
        worker.process(worker.jobs.get_nowait())
        camera.save_event.assert_not_called()
        self.assertIsNone(worker.last_result)

    def test_disable_cancels_queued_work_and_ignores_inflight_results(self):
        for timing in ("before", "during"):
            with self.subTest(timing=timing):
                client = Mock(model="gemini-3.5-flash-lite")
                worker = GeminiWorker(client, threading.Event())
                camera = app.Camera(1, "", None, threading.Lock())
                camera.save_event = Mock()
                worker.submit(camera, np.zeros((100, 100, 3), dtype=np.uint8),
                              [("person", 0.9, [10, 10, 90, 90])], revision=0)
                job = worker.jobs.get_nowait()
                if timing == "before":
                    worker.stop.set()
                else:
                    def assess(images):
                        worker.stop.set()
                        return [{"id": 1, "mask": "missing", "hairnet": "missing"}]
                    client.assess.side_effect = assess
                worker.process(job)
                if timing == "before":
                    client.assess.assert_not_called()
                camera.save_event.assert_not_called()
                self.assertIsNone(worker.last_result)
                self.assertFalse(worker.get_status()["enabled"])


if __name__ == "__main__":
    unittest.main()
