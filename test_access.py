import base64
import asyncio
import csv
import io
import os
import sqlite3
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path
from threading import Thread
from unittest.mock import AsyncMock, Mock, patch

from dotenv import dotenv_values
from fastapi import Request
from fastapi.responses import PlainTextResponse
from pydantic import ValidationError
import numpy as np

import app


def request(scheme="https", authorization=None):
    headers = []
    if authorization is not None:
        headers.append((b"authorization", authorization.encode("ascii")))
    return Request({"type": "http", "scheme": scheme, "headers": headers})


def basic(value):
    return "Basic " + base64.b64encode(value).decode("ascii")


class AccessTests(unittest.IsolatedAsyncioTestCase):
    async def test_authorized_https_request(self):
        next_handler = AsyncMock(return_value=PlainTextResponse("ok"))
        with patch.object(app, "credentials", (b"viewer", b"secret")):
            response = await app.protect_access(request(authorization=basic(b"viewer:secret")), next_handler)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.headers["cache-control"], "no-store")
        next_handler.assert_awaited_once()

    async def test_unauthorized_requests_are_rejected(self):
        for header in (None, "Basic !", basic(b"viewer:wrong"), basic(b"viewer")):
            with self.subTest(header=header), patch.object(app, "credentials", (b"viewer", b"secret")):
                next_handler = AsyncMock()
                response = await app.protect_access(request(authorization=header), next_handler)
                self.assertEqual(response.status_code, 401)
                self.assertIn("Basic", response.headers["www-authenticate"])
                next_handler.assert_not_awaited()

    async def test_http_is_rejected_even_with_credentials(self):
        next_handler = AsyncMock()
        with patch.object(app, "credentials", (b"viewer", b"secret")):
            response = await app.protect_access(
                request(scheme="http", authorization=basic(b"viewer:secret")), next_handler
            )
        self.assertEqual(response.status_code, 403)
        next_handler.assert_not_awaited()

    async def test_startup_requires_valid_credentials(self):
        for username, password in (("", "secret"), ("viewer", ""), ("user:name", "secret")):
            with self.subTest(username=username, password=password):
                with patch.dict(app.os.environ, {"APP_USERNAME": username, "APP_PASSWORD": password}):
                    with self.assertRaises(ValueError):
                        async with app.lifespan(app.app):
                            pass


class EventTests(unittest.TestCase):
    def setUp(self):
        self.db = sqlite3.connect(":memory:")
        self.addCleanup(self.db.close)
        connection = patch.object(app.sqlite3, "connect", return_value=self.db)
        connection.start()
        self.addCleanup(connection.stop)
        app.init_db()
        self.db.executemany(
            "INSERT INTO events (camera, kind, confidence, created_at, snapshot) VALUES (?, ?, ?, ?, ?)",
            [(1, "no_mask", 0.9, "2026-11-24T00:00:00+00:00", "one.jpg")]
            + [(2, "no_hairnet", 0.8, "2026-11-24T00:01:00+00:00", "two.jpg")] * 101,
        )

    def test_camera_filter_applies_before_limit(self):
        events = app.events(camera=1)
        self.assertEqual(events, [{
            "id": 1, "camera": 1, "kind": "no_mask", "confidence": 0.9,
            "created_at": "2026-11-24T00:00:00+00:00",
            "source": "yolo",
        }])
        self.assertEqual(app.events(camera=3), [])

    def test_kind_filter_is_applied_before_limit_and_combines_with_camera(self):
        events = app.events(camera=None, kind="no_mask")
        self.assertEqual([event["id"] for event in events], [1])
        self.assertEqual(app.events(camera=1, kind="no_hairnet"), [])

    def test_without_filter_preserves_latest_hundred_events(self):
        events = app.events(camera=None)
        self.assertEqual(len(events), 100)
        self.assertEqual([event["id"] for event in events], list(range(102, 2, -1)))

    def test_legacy_events_are_preserved_when_adding_source(self):
        self.db.execute("DROP TABLE events")
        self.db.execute("""CREATE TABLE events (
            id INTEGER PRIMARY KEY, camera INTEGER NOT NULL, kind TEXT NOT NULL,
            confidence REAL NOT NULL, created_at TEXT NOT NULL, snapshot TEXT NOT NULL
        )""")
        self.db.execute("INSERT INTO events VALUES (7, 1, 'no_mask', 0.8, 'earlier', 'old.jpg')")
        app.init_db()
        result = app.events(camera=1)
        self.assertEqual(result[0]["id"], 7)
        self.assertEqual(result[0]["source"], "yolo")


class CaptureHistoryTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(folder.cleanup)
        database = patch.object(app, "DATABASE", Path(folder.name) / "events.sqlite3")
        database.start()
        self.addCleanup(database.stop)
        app.init_db()
        self.camera = app.Camera(1, "configured", None, threading.Lock())
        ok, image = app.cv2.imencode(".jpg", np.zeros((40, 40, 3), np.uint8))
        self.assertTrue(ok)
        self.jpeg = image.tobytes()
        self.person = {
            "score": 0.9, "image": base64.b64encode(self.jpeg).decode("ascii"),
            "face": None, "head": base64.b64encode(self.jpeg).decode("ascii"),
            "face_status": "not_detected",
        }

    def test_all_records_survive_empty_scene_camera_change_and_restart_but_only_six_are_returned(self):
        self.camera.save_captures([self.person] * 8, "2026-11-24T12:00:00+00:00", 0)
        self.camera.capture_people(np.zeros((40, 40, 3), np.uint8), [], 0, 1, 0)
        self.camera.change_url("new")
        app.init_db()
        restarted = app.Camera(1, "new", None, threading.Lock())
        self.assertIsNone(restarted.captures)
        result = app.capture_history(camera=1)
        self.assertEqual(result["total"], 8)
        self.assertEqual([person["id"] for person in result["people"]], [8, 7, 6, 5, 4, 3])

    def test_order_uses_capture_time_and_filters_camera_and_kind_before_limit(self):
        self.camera.save_captures([self.person], "2026-11-24T12:00:00+00:00", 0)
        self.camera.save_captures([self.person], "2026-11-24T11:00:00+00:00", 0)
        self.camera.save_captures([self.person] * 8, "2026-11-24T13:00:00+00:00", 0,
                                  kind="no_mask", source="gemini")
        other = app.Camera(2, "configured", None, threading.Lock())
        other.save_captures([self.person] * 8, "2026-11-24T14:00:00+00:00", 0)
        result = app.capture_history(camera=1)
        self.assertEqual(result["total"], 2)
        self.assertEqual([person["id"] for person in result["people"]], [1, 2])
        masks = app.capture_history(camera=1, kind="no_mask")
        self.assertEqual(masks["total"], 8)
        self.assertEqual(len(masks["people"]), 6)
        self.assertTrue(all(person["source"] == "gemini" for person in masks["people"]))

    def test_stored_photos_are_served_without_inline_image_data_and_missing_records_return_404(self):
        self.camera.save_captures([self.person], "2026-11-24T12:00:00+00:00", 0)
        person = app.capture_history(camera=1)["people"][0]
        self.assertEqual(person["image"], "/api/captures/1/image")
        self.assertEqual(person["head"], "/api/captures/1/portrait")
        self.assertIsNone(person["face"])
        response = app.capture_image(1, "image")
        self.assertEqual(response.body, self.jpeg)
        self.assertEqual(response.headers["content-type"], "image/jpeg")
        self.assertEqual(app.capture_image(1, "portrait").body, self.jpeg)
        with self.assertRaises(app.HTTPException) as error:
            app.capture_image(99, "image")
        self.assertEqual(error.exception.status_code, 404)

    def test_invalid_batch_is_atomic_and_stale_or_cancelled_work_is_not_saved(self):
        with self.assertRaises(ValueError):
            self.camera.save_captures([self.person, {**self.person, "image": "!"}], "timestamp", 0)
        self.assertFalse(self.camera.save_captures([self.person], "timestamp", 99))
        stop = threading.Event()
        stop.set()
        self.assertFalse(self.camera.save_captures([self.person], "timestamp", 0, stop=stop))
        self.assertEqual(app.capture_history(camera=1)["total"], 0)


class DatasetTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(folder.cleanup)
        database = patch.object(app, "DATABASE", Path(folder.name) / "events.sqlite3")
        database.start()
        self.addCleanup(database.stop)
        app.init_db()
        self.camera = app.Camera(1, "configured", None, threading.Lock())
        self.stop = threading.Event()
        self.time = app.datetime(2026, 11, 25, tzinfo=app.timezone.utc)
        self.images = [(1, b"synthetic-image-1"), (2, b"synthetic-image-2"), (3, b"synthetic-image-3")]
        self.subjects = {i: (score, (0, 0, 10, 10)) for i, score in enumerate((0.9, 0.8, 0.71), 1)}
        self.results = [
            {"id": 1, "mask": "worn", "hairnet": "missing"},
            {"id": 2, "mask": "missing", "hairnet": "unclear"},
            {"id": 3, "mask": "unclear", "hairnet": "worn"},
        ]

    def save(self):
        return self.camera.save_dataset_results(self.time, 0, "gemini-test",
                                                self.images, self.subjects, self.results, self.stop)

    def test_saves_all_suggestions_and_rejects_stale_or_cancelled_jobs(self):
        self.assertTrue(self.save())
        self.stop.set()
        self.assertFalse(self.save())
        self.stop.clear()
        self.camera.change_url("new")
        self.assertFalse(self.save())
        app.init_db()
        result = app.dataset_samples(state="pending", offset=0)
        self.assertEqual((result["pending"], result["reviewed"]), (3, 0))
        self.assertEqual([row["suggested_mask"] for row in result["samples"]],
                         ["worn", "missing", "unclear"])
        self.assertEqual(app.dataset_image(2).body, b"synthetic-image-2")
        with self.assertRaises(app.HTTPException) as error:
            app.dataset_image(99)
        self.assertEqual(error.exception.status_code, 404)

    def test_review_and_correction_preserve_original_suggestion_and_require_valid_labels(self):
        self.save()
        app.review_dataset_sample(1, app.DatasetReview(mask="missing", hairnet="unclear"))
        app.review_dataset_sample(1, app.DatasetReview(mask="worn", hairnet="worn"))
        result = app.dataset_samples(state="reviewed", offset=0)
        self.assertEqual((result["pending"], result["reviewed"]), (2, 1))
        sample = result["samples"][0]
        self.assertEqual((sample["suggested_mask"], sample["suggested_hairnet"]), ("worn", "missing"))
        self.assertEqual((sample["reviewed_mask"], sample["reviewed_hairnet"]), ("worn", "worn"))
        with self.assertRaises(ValidationError):
            app.DatasetReview(mask="yes", hairnet="worn")
        with self.assertRaises(ValidationError):
            app.DatasetReview(mask="worn")
        with self.assertRaises(app.HTTPException) as error:
            app.review_dataset_sample(99, app.DatasetReview(mask="worn", hairnet="worn"))
        self.assertEqual(error.exception.status_code, 404)

    def test_reviewed_export_is_valid_zip_with_labels_and_only_reviewed_images(self):
        self.save()
        with self.assertRaises(app.HTTPException) as error:
            app.export_dataset()
        self.assertEqual(error.exception.status_code, 409)
        app.review_dataset_sample(2, app.DatasetReview(mask="worn", hairnet="unclear"))
        response = app.export_dataset()
        async def collect():
            return b"".join([chunk async for chunk in response.body_iterator])
        content = asyncio.run(collect())
        with zipfile.ZipFile(io.BytesIO(content)) as archive:
            self.assertEqual(set(archive.namelist()), {"labels.csv", "images/2.jpg"})
            self.assertEqual(archive.read("images/2.jpg"), b"synthetic-image-2")
            rows = list(csv.DictReader(io.StringIO(archive.read("labels.csv").decode("utf-8"))))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["suggested_mask"], "missing")
        self.assertEqual(rows[0]["reviewed_mask"], "worn")
        self.assertEqual(rows[0]["reviewed_hairnet"], "unclear")
        self.assertEqual(rows[0]["model"], "gemini-test")
        self.assertEqual(rows[0]["camera"], "1")
        self.assertEqual(rows[0]["captured_at"], self.time.isoformat())
        self.assertTrue(rows[0]["reviewed_at"])

    def test_dataset_pagination_does_not_drop_samples(self):
        for _ in range(5):
            self.save()
        first = app.dataset_samples(state="pending", offset=0)
        second = app.dataset_samples(state="pending", offset=12)
        self.assertEqual(first["pending"], 15)
        self.assertEqual(len(first["samples"]), 12)
        self.assertEqual(len(second["samples"]), 3)
        self.assertEqual([row["id"] for row in first["samples"] + second["samples"]], list(range(1, 16)))


class ModelModeTests(unittest.IsolatedAsyncioTestCase):
    async def test_person_model_is_limited_to_selected_camera(self):
        model = Mock(task="detect", names={0: "person", 1: "car"})
        settings = {
            "APP_USERNAME": "viewer", "APP_PASSWORD": "secret", "MODEL_PATH": "",
            "PERSON_MODEL_PATH": "person.pt", "GEMINI_ENABLED": "0",
            **{f"CAMERA_{i}_URL": "configured" if i <= 2 else "" for i in range(1, 7)},
        }
        with patch.dict(app.os.environ, settings), patch.object(app, "AI_CAMERA_ID", 1), \
                patch.object(app, "load_model", return_value=model), patch.object(app, "init_db"), \
                patch.object(app.threading, "Thread"):
            async with app.lifespan(app.app):
                self.assertIs(app.cameras[1].model, model)
                self.assertTrue(app.cameras[1].person_only)
                self.assertIsNone(app.cameras[2].model)
                self.assertIn("deteksi masker/hairnet belum aktif", app.model_status)

    async def test_invalid_custom_model_does_not_fall_back_to_person_model(self):
        settings = {
            "APP_USERNAME": "viewer", "APP_PASSWORD": "secret",
            "MODEL_PATH": "invalid.pt", "PERSON_MODEL_PATH": "person.pt", "GEMINI_ENABLED": "0",
        }
        model = Mock(task="detect", names={0: "person"})
        with patch.dict(app.os.environ, settings), patch.object(app, "init_db"), \
                patch.object(app, "load_model", return_value=model) as loader:
            with self.assertRaisesRegex(ValueError, "no_mask and no_hairnet"):
                async with app.lifespan(app.app):
                    pass
            loader.assert_called_once_with("invalid.pt")

    async def test_gemini_requires_key_before_starting_camera_threads(self):
        settings = {
            "APP_USERNAME": "viewer", "APP_PASSWORD": "secret", "MODEL_PATH": "",
            "PERSON_MODEL_PATH": "person.pt", "CAMERA_1_URL": "configured",
            "GEMINI_ENABLED": "1", "GEMINI_API_KEY": "",
        }
        model = Mock(task="detect", names={0: "person"})
        with patch.dict(app.os.environ, settings), patch.object(app, "AI_CAMERA_ID", 1), \
                patch.object(app, "init_db"), patch.object(app, "load_model", return_value=model), \
                patch.object(app.threading, "Thread") as thread:
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
                async with app.lifespan(app.app):
                    pass
            thread.assert_not_called()


class CameraSettingsTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(folder.cleanup)
        self.env_file = Path(folder.name) / ".env"
        self.url = "rtsp://viewer:p%40ss@192.168.1.10:554/cam/realmonitor?channel=2&subtype=0"
        self.env_file.write_text(f"CAMERA_1_URL='{self.url}'\nCAMERA_2_URL='unchanged'\n")
        self.camera = app.Camera(1, self.url, None, threading.Lock())
        for patcher in (
            patch.object(app, "ENV_FILE", self.env_file),
            patch.object(app, "cameras", {1: self.camera}),
            patch.dict(app.os.environ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def test_ip_change_is_persistent_and_preserves_credentials_and_stream(self):
        self.camera.frame = b"old frame"
        self.camera.people_count = 3
        result = app.change_camera_address(1, app.CameraAddress(address="192.168.1.20"))
        expected = self.url.replace("192.168.1.10", "192.168.1.20")
        self.assertEqual(self.camera.url, expected)
        self.assertEqual(dotenv_values(self.env_file)["CAMERA_1_URL"], expected)
        self.assertEqual(dotenv_values(self.env_file)["CAMERA_2_URL"], "unchanged")
        self.assertEqual(app.os.environ["CAMERA_1_URL"], expected)
        self.assertEqual(result["address"], "192.168.1.20")
        self.assertTrue(self.camera.reconnect.is_set())
        self.assertIsNone(self.camera.frame)
        self.assertIsNone(self.camera.people_count)

    def test_settings_do_not_expose_credentials_and_ipv6_is_supported(self):
        settings = app.camera_settings()
        self.assertEqual(settings[0], {"id": 1, "configured": True, "address": "192.168.1.10"})
        self.assertEqual(settings[1], {"id": 2, "configured": False, "address": None})
        result = app.replace_camera_address(self.url, "2001:db8::1")
        self.assertIn("@[2001:db8::1]:554/", result)
        for invalid in ("https://192.168.1.20", "hostname.test", "192.168.1.999"):
            with self.subTest(address=invalid), self.assertRaises(ValidationError):
                app.CameraAddress(address=invalid)

    def test_write_failure_keeps_existing_live_camera(self):
        with patch.object(app, "set_key", side_effect=OSError("write denied")), \
                self.assertLogs("cctv", level="ERROR"):
            with self.assertRaises(app.HTTPException) as error:
                app.change_camera_address(1, app.CameraAddress(address="192.168.1.20"))
        self.assertEqual(error.exception.status_code, 500)
        self.assertEqual(self.camera.url, self.url)
        self.assertFalse(self.camera.reconnect.is_set())
        self.assertEqual(dotenv_values(self.env_file)["CAMERA_1_URL"], self.url)


class VisionSettingsTests(unittest.TestCase):
    def setUp(self):
        folder = tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR"))
        self.addCleanup(folder.cleanup)
        self.env_file = Path(folder.name) / ".env"
        self.env_file.write_text(
            "GEMINI_ENABLED='0'\nGEMINI_API_KEY='synthetic-key'\n"
            "GEMINI_MODEL='gemini-3.5-flash-lite'\nGEMINI_INTERVAL='10'\nOTHER_SETTING='keep'\n"
        )
        self.model = Mock(names={0: "person"})
        self.camera = app.Camera(1, "configured", self.model, threading.Lock(), person_only=True)
        self.camera.status = "Terhubung"
        self.camera.people_count = 3
        self.threads = []

        def thread(*args, **kwargs):
            result = Mock()
            result.is_alive.return_value = False
            self.threads.append(result)
            return result

        for patcher in (
            patch.object(app, "ENV_FILE", self.env_file),
            patch.object(app, "cameras", {1: self.camera}),
            patch.object(app, "gemini_worker", None),
            patch.object(app, "stop_event", threading.Event()),
            patch.object(app, "AI_CAMERA_ID", 1),
            patch.object(app.threading, "Thread", side_effect=thread),
            patch.dict(app.os.environ),
        ):
            patcher.start()
            self.addCleanup(patcher.stop)

    def enable(self):
        return app.change_vision(app.VisionSettings(enabled=True, consent=True))

    def test_enable_disable_preserves_camera_detection_and_persists_state(self):
        result = self.enable()
        worker = app.gemini_worker
        self.assertTrue(result["enabled"])
        self.assertIs(self.camera.vision, worker)
        self.assertIsNot(worker.stop, app.stop_event)
        self.threads[0].start.assert_called_once()
        self.assertTrue(self.enable()["enabled"])
        self.assertEqual(len(self.threads), 1)
        self.assertEqual(dotenv_values(self.env_file)["GEMINI_ENABLED"], "1")
        self.assertNotIn("synthetic-key", str(app.status()))
        worker.last_result = {"people": [{"id": 1}]}
        result = app.change_vision(app.VisionSettings(enabled=False))
        self.assertFalse(result["enabled"])
        self.assertIsNone(result["last_result"])
        self.assertTrue(worker.stop.is_set())
        self.assertFalse(app.stop_event.is_set())
        self.assertIsNone(self.camera.vision)
        self.assertIs(self.camera.model, self.model)
        self.assertEqual(self.camera.get_status()["people_count"], 3)
        self.assertEqual(dotenv_values(self.env_file)["GEMINI_ENABLED"], "0")
        self.assertEqual(dotenv_values(self.env_file)["OTHER_SETTING"], "keep")

    def test_enable_requires_explicit_consent_and_boolean_input(self):
        with self.assertRaises(app.HTTPException) as error:
            app.change_vision(app.VisionSettings(enabled=True))
        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(self.threads, [])
        self.assertEqual(dotenv_values(self.env_file)["GEMINI_ENABLED"], "0")
        with self.assertRaises(ValidationError):
            app.VisionSettings(enabled="true", consent=True)

    def test_configuration_errors_do_not_start_ai_or_expose_config_values(self):
        for setting in ("GEMINI_API_KEY=''", "GEMINI_MODEL='invalid-private-value'",
                        "GEMINI_INTERVAL='invalid-private-value'", "GEMINI_INTERVAL='nan'"):
            with self.subTest(setting=setting):
                with self.env_file.open("a") as file:
                    file.write("\n" + setting + "\n")
                with self.assertLogs("cctv", level="ERROR"), self.assertRaises(app.HTTPException) as error:
                    self.enable()
                self.assertEqual(error.exception.status_code, 400)
                self.assertNotIn("invalid-private-value", error.exception.detail)
                self.assertIsNone(app.gemini_worker)
                self.assertEqual(self.threads, [])
                self.env_file.write_text(
                    "GEMINI_ENABLED='0'\nGEMINI_API_KEY='synthetic-key'\n"
                    "GEMINI_MODEL='gemini-3.5-flash-lite'\nGEMINI_INTERVAL='10'\n"
                )
        self.camera.model = None
        with self.assertLogs("cctv", level="ERROR"), self.assertRaises(app.HTTPException) as error:
            self.enable()
        self.assertEqual(error.exception.status_code, 400)
        self.assertEqual(self.threads, [])

    def test_write_failure_rolls_back_activation_but_still_stops_on_disable(self):
        with patch.object(app, "set_key", side_effect=OSError("write denied")), \
                self.assertLogs("cctv", level="ERROR"), self.assertRaises(app.HTTPException) as error:
            self.enable()
        self.assertEqual(error.exception.status_code, 500)
        self.assertIsNone(app.gemini_worker)
        self.assertIsNone(self.camera.vision)
        self.threads[0].join.assert_called_once_with(timeout=1)
        self.assertEqual(dotenv_values(self.env_file)["GEMINI_ENABLED"], "0")
        self.enable()
        worker = app.gemini_worker
        with patch.object(app, "set_key", side_effect=OSError("write denied")), \
                self.assertLogs("cctv", level="ERROR"), self.assertRaises(app.HTTPException):
            app.change_vision(app.VisionSettings(enabled=False))
        self.assertTrue(worker.stop.is_set())
        self.assertIsNone(self.camera.vision)
        self.assertEqual(app.os.environ["GEMINI_ENABLED"], "0")
        self.assertEqual(dotenv_values(self.env_file)["GEMINI_ENABLED"], "1")

    def test_reenable_waits_for_previous_request_and_creates_fresh_worker(self):
        self.enable()
        previous_worker = app.gemini_worker
        self.threads[0].is_alive.return_value = True
        result = app.change_vision(app.VisionSettings(enabled=False))
        self.assertTrue(result["stopping"])
        with self.assertRaises(app.HTTPException) as error:
            self.enable()
        self.assertEqual(error.exception.status_code, 409)
        self.assertEqual(len(self.threads), 1)
        self.threads[0].is_alive.return_value = False
        self.assertTrue(self.enable()["enabled"])
        self.assertIsNot(app.gemini_worker, previous_worker)
        self.assertFalse(app.gemini_worker.stop.is_set())
        self.assertEqual(len(self.threads), 2)

    def test_real_worker_stops_and_discards_synthetic_inflight_result(self):
        started, release = threading.Event(), threading.Event()
        self.camera.save_event = Mock()
        with patch.object(app.threading, "Thread", Thread):
            self.enable()
        worker = app.gemini_worker

        def cleanup():
            worker.stop.set()
            release.set()
            worker.thread.join(timeout=3)

        self.addCleanup(cleanup)

        def assess(images):
            started.set()
            release.wait(timeout=3)
            return [{"id": 1, "mask": "missing", "hairnet": "missing"}]

        worker.client.assess = Mock(side_effect=assess)
        worker.submit(self.camera, np.zeros((80, 80, 3), dtype=np.uint8),
                      [("person", 0.9, [5, 5, 75, 75])], revision=0)
        self.assertTrue(started.wait(timeout=3))
        result = app.change_vision(app.VisionSettings(enabled=False))
        self.assertTrue(result["stopping"])
        release.set()
        worker.thread.join(timeout=3)
        self.assertFalse(worker.thread.is_alive())
        self.camera.save_event.assert_not_called()
        self.assertIsNone(worker.get_status()["last_result"])
        self.assertFalse(app.stop_event.is_set())


if __name__ == "__main__":
    unittest.main()
