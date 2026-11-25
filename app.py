import asyncio
import base64
import binascii
import hmac
import logging
import os
import sqlite3
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal, Optional
from urllib.parse import urlsplit

import cv2
from dotenv import dotenv_values, load_dotenv, set_key
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, PlainTextResponse, Response, StreamingResponse
from pydantic import BaseModel, IPvAnyAddress, StrictBool

from detection import VIOLATIONS, annotations, load_face_detector, load_model, person_captures, violations
from gemini_vision import GeminiClient, GeminiWorker

ROOT = Path(__file__).resolve().parent
ENV_FILE = ROOT / ".env"
load_dotenv(ENV_FILE)
DATA = ROOT / "data"
DATABASE = DATA / "events.sqlite3"
DATA.mkdir(exist_ok=True)
(DATA / "snapshots").mkdir(exist_ok=True)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("cctv")


def number(name, default, minimum, maximum):
    value = float(os.getenv(name, str(default)))
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} must be between {minimum} and {maximum}")
    return value


CONFIDENCE = number("CONFIDENCE", 0.5, 0, 1)
INTERVAL = number("DETECTION_INTERVAL", 2, 0.1, 3600)
COOLDOWN = number("EVENT_COOLDOWN", 30, 0, 86400)
AI_CAMERA_ID = int(os.getenv("AI_CAMERA_ID", "0"))
if not 0 <= AI_CAMERA_ID <= 6:
    raise ValueError("AI_CAMERA_ID must be 0 (all cameras) or between 1 and 6")


def init_db():
    with sqlite3.connect(DATABASE) as db:
        db.execute("""CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY, camera INTEGER NOT NULL, kind TEXT NOT NULL,
            confidence REAL NOT NULL, created_at TEXT NOT NULL, snapshot TEXT NOT NULL,
            source TEXT NOT NULL DEFAULT 'yolo'
        )""")
        if "source" not in {row[1] for row in db.execute("PRAGMA table_info(events)")}:
            db.execute("ALTER TABLE events ADD COLUMN source TEXT NOT NULL DEFAULT 'yolo'")
        db.execute("""CREATE TABLE IF NOT EXISTS captures (
            id INTEGER PRIMARY KEY, camera INTEGER NOT NULL, kind TEXT NOT NULL,
            source TEXT NOT NULL, confidence REAL NOT NULL, created_at TEXT NOT NULL,
            image BLOB NOT NULL, portrait BLOB NOT NULL, face_status TEXT NOT NULL
        )""")
        db.execute("""CREATE INDEX IF NOT EXISTS captures_camera_kind_time
            ON captures (camera, kind, created_at DESC, id DESC)""")


class Camera:
    def __init__(self, number, url, model, model_lock, person_only=False, vision=None):
        self.number = number
        self.url = url
        self.model = model
        self.model_lock = model_lock
        self.person_only = person_only
        self.vision = vision
        self.event_lock = threading.Lock()
        self.capture_lock = threading.Lock()
        self.lock = threading.Lock()
        self.frame = None
        self.status = "Menunggu koneksi"
        self.last_event = {}
        self.annotations = []
        self.annotation_shape = None
        self.people_count = None
        self.captures = None
        self.capture_error = None
        self.last_capture = float("-inf")
        self.face_detector = None
        self.reconnect = threading.Event()
        self.revision = 0

    def change_url(self, url):
        with self.lock:
            self.url = url
            self.revision += 1
            self.frame = None
            self.annotations = []
            self.annotation_shape = None
            self.people_count = None
            self.captures = None
            self.capture_error = None
            self.last_capture = float("-inf")
            self.status = "Menghubungkan ulang ke IP baru"
            self.reconnect.set()
        if self.vision is not None:
            self.vision.invalidate()

    def run(self, stop):
        while not stop.is_set():
            with self.lock:
                self.reconnect.clear()
                url = self.url
                revision = self.revision
            capture = cv2.VideoCapture()
            try:
                # Timeouts are configured before opening for supported FFmpeg builds.
                opened = capture.open(url, cv2.CAP_FFMPEG, [
                    cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000,
                    cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000,
                ])
                if self.reconnect.is_set():
                    continue
                if not opened:
                    with self.lock:
                        if self.revision == revision:
                            self.status = "Koneksi gagal; mencoba kembali"
                    continue
                with self.lock:
                    if self.revision != revision:
                        continue
                    self.status = "Terhubung"
                    self.annotations = []
                    self.annotation_shape = None
                    self.people_count = None
                    self.captures = None
                    self.capture_error = None
                    self.last_capture = float("-inf")
                last_detection = 0
                while not stop.is_set() and not self.reconnect.is_set():
                    ok, frame = capture.read()
                    if self.reconnect.is_set() or stop.is_set():
                        break
                    if not ok:
                        with self.lock:
                            if self.revision == revision:
                                self.status = "Aliran terputus; mencoba kembali"
                        break
                    now = time.monotonic()
                    if self.model is not None and (
                        now - last_detection >= INTERVAL or self.annotation_shape != frame.shape[:2]
                    ):
                        last_detection = now
                        self.detect(frame, now, revision=revision)
                    self.draw_annotations(frame)
                    ok, encoded = cv2.imencode(".jpg", frame)
                    if ok:
                        with self.lock:
                            if self.revision == revision:
                                self.frame = encoded.tobytes()
            except (cv2.error, RuntimeError, sqlite3.Error, OSError) as exc:
                with self.lock:
                    if self.revision == revision:
                        self.status = "Kesalahan pemrosesan; mencoba kembali"
                logger.error("Camera %s processing error: %s", self.number, type(exc).__name__)
            finally:
                capture.release()
                with self.lock:
                    self.frame = None
                    self.annotations = []
                    self.annotation_shape = None
                    self.people_count = None
                    self.captures = None
                    self.capture_error = None
                    self.last_capture = float("-inf")
                if not stop.is_set() and not self.reconnect.is_set():
                    self.reconnect.wait(5)

    def detect(self, frame, now, revision=None):
        with self.model_lock:
            prediction = self.model.predict(frame, conf=CONFIDENCE, verbose=False)[0]
        boxes = prediction.boxes
        detections = [
            (int(box.cls.item()), float(box.conf.item()), box.xyxy[0].tolist())
            for box in boxes
        ]
        if self.person_only:
            detections = [detection for detection in detections
                          if prediction.names.get(detection[0]) == "person"]
        current_annotations = annotations(detections, prediction.names, CONFIDENCE)
        count = None
        if "person" in prediction.names.values():
            count = sum(kind == "person" for kind, _, _ in current_annotations)
        with self.lock:
            if revision is not None and revision != self.revision:
                return
            self.annotations = current_annotations
            self.annotation_shape = frame.shape[:2]
            self.people_count = count
            vision = self.vision
            current_revision = self.revision
        self.capture_people(frame, current_annotations, count, now, current_revision)
        found = violations(detections, prediction.names, CONFIDENCE)
        for kind, score, _ in found:
            snapshot_frame = frame.copy()
            self.draw_annotations(snapshot_frame)
            self.save_event(kind, score, snapshot_frame, revision=revision)
        if vision is not None:
            vision.submit(self, frame, current_annotations, self.revision if revision is None else revision)

    def capture_people(self, frame, detections, count, now, revision):
        if self.number != 1:
            return
        captured_at = datetime.now(timezone.utc).isoformat()
        with self.lock:
            if self.revision != revision:
                return
            if count is None:
                self.captures = None
                self.capture_error = None
                self.last_capture = float("-inf")
                return
            if count == 0:
                if self.captures is None or self.captures["people"]:
                    self.captures = {"created_at": captured_at, "people": []}
                self.capture_error = None
                self.last_capture = float("-inf")
                return
            if now - self.last_capture < 3 and (
                self.captures is None or len(self.captures["people"]) == count
            ):
                return
            self.last_capture = now
        try:
            people = self.prepare_captures(frame, detections)
            if not self.save_captures(people, captured_at, revision):
                return
        except (cv2.error, OSError, ValueError, sqlite3.Error) as error:
            with self.lock:
                if self.revision != revision:
                    return
                self.captures = None
                self.capture_error = "Tangkapan tidak tersimpan; periksa model wajah, ruang disk, dan log server"
            logger.error("Camera %s capture failed: %s", self.number, type(error).__name__)
            return
        with self.lock:
            if self.revision == revision:
                self.captures = {"created_at": captured_at, "people": people}
                self.capture_error = None

    def prepare_captures(self, frame, detections):
        with self.capture_lock:
            if self.face_detector is None:
                self.face_detector = load_face_detector()
            return person_captures(frame, detections, self.face_detector)

    def save_captures(self, people, captured_at, revision, kind="person", source="yolo", stop=None):
        with self.event_lock, self.lock:
            if self.revision != revision or (stop is not None and stop.is_set()):
                return False
            rows = [
                (self.number, kind, source, person["score"], captured_at,
                 base64.b64decode(person["image"], validate=True),
                 base64.b64decode(person["face"] or person["head"], validate=True),
                 person["face_status"])
                for person in people
            ]
            with sqlite3.connect(DATABASE) as db:
                db.executemany("""INSERT INTO captures
                    (camera, kind, source, confidence, created_at, image, portrait, face_status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)""", rows)
                if stop is not None and stop.is_set():
                    db.rollback()
                    return False
            return True

    def save_missing_people(self, frame, detections, captured_at, revision, stop):
        if stop.is_set():
            return False
        people = self.prepare_captures(frame, detections)
        return self.save_captures(people, captured_at.isoformat(), revision,
                                  kind="no_mask", source="gemini", stop=stop)

    def save_event(self, kind, score, frame, source="yolo", captured_at=None, revision=None):
        with self.event_lock:
            with self.lock:
                if revision is not None and revision != self.revision:
                    return
            now = time.monotonic()
            if now - self.last_event.get(kind, float("-inf")) < COOLDOWN:
                return
            timestamp = captured_at or datetime.now(timezone.utc)
            filename = f"{self.number}-{time.time_ns()}-{kind}.jpg"
            path = DATA / "snapshots" / filename
            if not cv2.imwrite(str(path), frame):
                raise OSError("Could not save snapshot")
            try:
                with sqlite3.connect(DATABASE) as db:
                    db.execute("INSERT INTO events (camera, kind, confidence, created_at, snapshot, source) VALUES (?, ?, ?, ?, ?, ?)",
                               (self.number, kind, score, timestamp.isoformat(), filename, source))
            except sqlite3.Error:
                path.unlink(missing_ok=True)
                raise
            self.last_event[kind] = now

    def draw_annotations(self, frame):
        for kind, score, coords in self.annotations:
            x1, y1, x2, y2 = map(int, coords)
            color = (0, 180, 0) if kind == "person" else (0, 0, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, f"{kind} {score:.2f}", (x1, max(20, y1 - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    def get_status(self):
        with self.lock:
            return {
                "id": self.number,
                "status": self.status,
                "people_count": self.people_count if self.status == "Terhubung" else None,
                "captures": self.captures if self.status == "Terhubung" else None,
                "capture_error": self.capture_error if self.status == "Terhubung" else None,
            }


cameras = {}
model_status = "Model belum dikonfigurasi; deteksi tidak aktif"
stop_event = threading.Event()
credentials = None
config_lock = threading.Lock()
gemini_worker = None


def save_setting(key, value):
    if not ENV_FILE.is_file():
        raise OSError("Configuration file is missing")
    saved, _, _ = set_key(ENV_FILE, key, value)
    if not saved:
        raise OSError("Configuration was not saved")
    os.environ[key] = value


def create_gemini_worker(model, settings):
    if model is None or "person" not in model.names.values() or AI_CAMERA_ID not in (0, 1):
        raise ValueError("Gemini memerlukan model deteksi orang aktif pada Kamera 1")
    client = GeminiClient((settings.get("GEMINI_API_KEY") or "").strip(),
                          (settings.get("GEMINI_MODEL", "gemini-3.5-flash-lite") or "").strip())
    try:
        interval = float(settings.get("GEMINI_INTERVAL", "10") or "")
    except ValueError:
        raise ValueError("GEMINI_INTERVAL harus antara 1 dan 3600 detik") from None
    if not 1 <= interval <= 3600:
        raise ValueError("GEMINI_INTERVAL harus antara 1 dan 3600 detik")
    return GeminiWorker(client, threading.Event(), interval)


def vision_status():
    return gemini_worker.get_status() if gemini_worker else {
        "enabled": False, "stopping": False,
        "status": "Gemini dinonaktifkan", "last_result": None,
    }


@asynccontextmanager
async def lifespan(app):
    global credentials, model_status, gemini_worker
    stop_event.clear()
    cameras.clear()
    model_status = "Model belum dikonfigurasi; deteksi tidak aktif"
    gemini_worker = None
    username = os.getenv("APP_USERNAME", "")
    password = os.getenv("APP_PASSWORD", "")
    if not username or ":" in username or not password:
        raise ValueError("APP_USERNAME and APP_PASSWORD must be set; username cannot contain ':'")
    credentials = (username.encode("utf-8"), password.encode("utf-8"))
    init_db()
    model = None
    model_path = os.getenv("MODEL_PATH", "").strip()
    person_only = not model_path
    if person_only:
        model_path = os.getenv("PERSON_MODEL_PATH", "").strip()
    if model_path:
        model = load_model(model_path)
        if model.task != "detect":
            raise ValueError("Configured model must be an object detection model")
        labels = set(model.names.values())
        if person_only and "person" not in labels:
            raise ValueError("Person model must contain the person class")
        if not person_only and not set(VIOLATIONS).issubset(labels):
            raise ValueError("Model must contain no_mask and no_hairnet classes")
        scope = f"Kamera {AI_CAMERA_ID}" if AI_CAMERA_ID else "semua kamera"
        if person_only:
            model_status = f"Deteksi orang aktif ({scope}); deteksi masker/hairnet belum aktif"
        else:
            model_status = f"Model aktif ({scope})"
            if "person" not in labels:
                model_status += "; kelas person tidak tersedia"
    enabled = os.getenv("GEMINI_ENABLED", "0").strip()
    if enabled not in ("0", "1"):
        raise ValueError("GEMINI_ENABLED must be 0 or 1")
    if enabled == "1":
        if not os.getenv("CAMERA_1_URL", "").strip():
            raise ValueError("Gemini requires a person detector enabled on Camera 1")
        gemini_worker = create_gemini_worker(model, os.environ)
    model_lock = threading.Lock()
    threads = []
    try:
        for i in range(1, 7):
            url = os.getenv(f"CAMERA_{i}_URL", "").strip()
            if url:
                camera_model = model if AI_CAMERA_ID in (0, i) else None
                camera = cameras[i] = Camera(i, url, camera_model, model_lock, person_only=person_only,
                                            vision=gemini_worker if i == 1 else None)
                thread = threading.Thread(target=camera.run, args=(stop_event,), daemon=True)
                thread.start()
                threads.append(thread)
        if gemini_worker is not None:
            gemini_worker.thread.start()
        yield
    finally:
        stop_event.set()
        if gemini_worker is not None:
            gemini_worker.stop.set()
        for camera in cameras.values():
            camera.reconnect.set()
        for thread in threads:
            thread.join(timeout=12)
        if gemini_worker is not None and gemini_worker.thread.is_alive():
            gemini_worker.thread.join(timeout=17)


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.middleware("http")
async def protect_access(request: Request, call_next):
    if request.scope["scheme"] != "https":
        return PlainTextResponse("HTTPS diperlukan", status_code=403)
    header = request.headers.get("authorization", "")
    valid = False
    if len(header) <= 8192:
        scheme, separator, token = header.partition(" ")
        if separator and scheme.lower() == "basic":
            try:
                decoded = base64.b64decode(token, validate=True)
                username, separator, password = decoded.partition(b":")
                if separator and credentials is not None:
                    username_matches = hmac.compare_digest(username, credentials[0])
                    password_matches = hmac.compare_digest(password, credentials[1])
                    valid = username_matches and password_matches
            except (binascii.Error, ValueError):
                pass
    if not valid:
        return PlainTextResponse(
            "Autentikasi diperlukan", status_code=401,
            headers={"WWW-Authenticate": 'Basic realm="CCTV Monitor", charset="UTF-8"'},
        )
    response = await call_next(request)
    response.headers["Cache-Control"] = "no-store"
    return response


@app.get("/")
def index():
    return FileResponse(ROOT / "index.html")


@app.get("/api/status")
def status():
    with config_lock:
        vision = vision_status()
        camera = cameras.get(1)
        current_model_status = model_status
        if vision["enabled"] and camera is not None and camera.person_only:
            current_model_status = "Deteksi orang YOLO aktif; pemeriksaan masker/hairnet melalui Gemini"
        return {"model": current_model_status, "vision": vision, "cameras": [
            cameras[i].get_status() if i in cameras else
            {"id": i, "status": "Belum dikonfigurasi", "people_count": None}
            for i in range(1, 7)
        ]}


class VisionSettings(BaseModel):
    enabled: StrictBool
    consent: StrictBool = False


@app.patch("/api/vision")
def change_vision(settings: VisionSettings):
    global gemini_worker
    if settings.enabled and not settings.consent:
        raise HTTPException(400, "Konfirmasi izin pengiriman potongan gambar CCTV ke Google diperlukan")
    with config_lock:
        worker = gemini_worker
        camera = cameras.get(1)
        new_worker = None
        if settings.enabled and (worker is None or worker.stop.is_set()):
            if worker is not None and worker.thread.is_alive():
                raise HTTPException(409, "Permintaan Gemini sebelumnya belum selesai; coba aktifkan lagi nanti")
            if camera is None:
                raise HTTPException(400, "Konfigurasikan Kamera 1 terlebih dahulu")
            try:
                if not ENV_FILE.is_file():
                    raise OSError("Configuration file is missing")
                configuration = {**os.environ, **dotenv_values(ENV_FILE)}
                new_worker = create_gemini_worker(camera.model, configuration)
            except OSError:
                logger.error("Could not read Gemini configuration")
                raise HTTPException(500, "Konfigurasi AI tidak dapat dibaca pada server")
            except ValueError as error:
                logger.error("Invalid Gemini configuration: %s", type(error).__name__)
                raise HTTPException(400, str(error))
            try:
                new_worker.thread.start()
            except RuntimeError:
                logger.error("Could not start Gemini worker")
                raise HTTPException(500, "Worker AI tidak dapat dijalankan")
        if not settings.enabled:
            if worker is not None:
                worker.stop.set()
                with worker.lock:
                    worker.last_result = None
            if camera is not None:
                with camera.lock:
                    camera.vision = None
            os.environ["GEMINI_ENABLED"] = "0"
        try:
            save_setting("GEMINI_ENABLED", "1" if settings.enabled else "0")
        except OSError:
            if new_worker is not None:
                new_worker.stop.set()
                new_worker.thread.join(timeout=1)
            logger.error("Could not save Gemini activation setting")
            detail = "Status AI tidak disimpan; periksa izin tulis konfigurasi server" if settings.enabled else (
                "AI sudah dinonaktifkan, tetapi status tidak tersimpan; periksa izin tulis sebelum restart"
            )
            raise HTTPException(500, detail)
        if settings.enabled:
            if new_worker is not None:
                gemini_worker = new_worker
                with camera.lock:
                    camera.vision = new_worker
        return vision_status()


@app.get("/api/events")
def events(camera: Optional[int] = Query(default=None, ge=1, le=6),
           kind: Optional[Literal["no_mask", "no_hairnet"]] = None):
    with sqlite3.connect(DATABASE) as db:
        db.row_factory = sqlite3.Row
        query = "SELECT id, camera, kind, confidence, created_at, source FROM events"
        conditions, parameters = [], []
        if camera is not None:
            conditions.append("camera = ?")
            parameters.append(camera)
        if kind is not None:
            conditions.append("kind = ?")
            parameters.append(kind)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        return [dict(row) for row in db.execute(
            query + " ORDER BY id DESC LIMIT 100", parameters
        )]


@app.get("/api/captures")
def capture_history(camera: int = Query(default=1, ge=1, le=6),
                    kind: Literal["person", "no_mask"] = "person"):
    with sqlite3.connect(DATABASE) as db:
        db.row_factory = sqlite3.Row
        db.execute("BEGIN")
        parameters = (camera, kind)
        total = db.execute("SELECT COUNT(*) FROM captures WHERE camera = ? AND kind = ?",
                           parameters).fetchone()[0]
        rows = db.execute("""SELECT id, camera, kind, source, confidence AS score, created_at, face_status
            FROM captures WHERE camera = ? AND kind = ?
            ORDER BY created_at DESC, id DESC LIMIT 6""", parameters).fetchall()
    people = []
    for row in rows:
        person = dict(row)
        portrait = f"/api/captures/{person['id']}/portrait"
        person.update(image=f"/api/captures/{person['id']}/image",
                      face=portrait if person["face_status"] == "detected" else None,
                      head=portrait if person["face_status"] != "detected" else None)
        people.append(person)
    return {"total": total, "people": people}


@app.get("/api/captures/{capture_id}/{part}")
def capture_image(capture_id: int, part: Literal["image", "portrait"]):
    with sqlite3.connect(DATABASE) as db:
        row = db.execute("SELECT image, portrait FROM captures WHERE id = ?", (capture_id,)).fetchone()
    if row is None:
        raise HTTPException(404, "Tangkapan tidak ditemukan")
    return Response(content=row[0 if part == "image" else 1], media_type="image/jpeg")


@app.get("/api/events/{event_id}/snapshot")
def snapshot(event_id: int):
    with sqlite3.connect(DATABASE) as db:
        row = db.execute("SELECT snapshot FROM events WHERE id = ?", (event_id,)).fetchone()
    if not row:
        raise HTTPException(404, "Kejadian tidak ditemukan")
    path = DATA / "snapshots" / row[0]
    if not path.is_file():
        raise HTTPException(404, "Foto tidak ditemukan")
    return FileResponse(path, media_type="image/jpeg")


class CameraAddress(BaseModel):
    address: IPvAnyAddress


def replace_camera_address(url, address):
    parsed = urlsplit(url)
    if parsed.scheme != "rtsp" or not parsed.hostname:
        raise ValueError("Konfigurasi RTSP kamera tidak valid")
    userinfo, separator, _ = parsed.netloc.rpartition("@")
    host = f"[{address}]" if ":" in address else address
    netloc = f"{userinfo}@{host}" if separator else host
    if parsed.port is not None:
        netloc += f":{parsed.port}"
    return parsed._replace(netloc=netloc).geturl()


@app.get("/api/camera-settings")
def camera_settings():
    with config_lock:
        result = []
        for i in range(1, 7):
            camera = cameras.get(i)
            try:
                address = urlsplit(camera.url).hostname if camera else None
            except ValueError:
                logger.error("Camera %s has invalid RTSP configuration", i)
                raise HTTPException(500, "Konfigurasi kamera tidak valid")
            result.append({"id": i, "configured": camera is not None, "address": address})
        return result


@app.patch("/api/cameras/{camera_id}/address")
def change_camera_address(camera_id: int, settings: CameraAddress):
    with config_lock:
        camera = cameras.get(camera_id)
        if camera is None:
            raise HTTPException(404, "Kamera belum dikonfigurasi; isi URL RTSP awal terlebih dahulu")
        address = str(settings.address)
        try:
            url = replace_camera_address(camera.url, address)
        except ValueError:
            logger.error("Camera %s has invalid RTSP configuration", camera_id)
            raise HTTPException(400, "Konfigurasi RTSP kamera tidak valid")
        key = f"CAMERA_{camera_id}_URL"
        try:
            save_setting(key, url)
        except OSError as error:
            logger.error("Could not save camera %s address: %s", camera_id, type(error).__name__)
            raise HTTPException(500, "IP tidak disimpan; periksa izin tulis konfigurasi server")
        camera.change_url(url)
        return {"id": camera_id, "address": address, "status": "Menghubungkan ulang"}


@app.get("/api/cameras/{camera_id}/stream")
def stream(camera_id: int):
    camera = cameras.get(camera_id)
    if camera is None:
        raise HTTPException(404, "Kamera belum dikonfigurasi")

    async def frames():
        while not stop_event.is_set():
            with camera.lock:
                frame = camera.frame
            if frame:
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
            await asyncio.sleep(0.1)

    return StreamingResponse(frames(), media_type="multipart/x-mixed-replace; boundary=frame")
