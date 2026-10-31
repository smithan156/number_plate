import base64
import csv
import io
import json
import logging
import re
import time
import uuid
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Annotated

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from sqlalchemy import func, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session
from starlette.concurrency import run_in_threadpool

from app.config import ALLOWED_ORIGINS, DATA_DIR, MAX_UPLOAD_BYTES, UPLOAD_DIR
from app.database import Base, SessionLocal, engine, get_db
from app.models import Detection
from app.services.anpr import DetectionResult, ModelUnavailableError, box_json, encode_jpeg, pipeline

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("smart-anpr")
Base.metadata.create_all(bind=engine)

# Ensure plate_confidence column exists in SQLite database
with engine.connect() as conn:
    try:
        conn.execute(text("SELECT plate_confidence FROM detections LIMIT 1"))
    except Exception:
        try:
            conn.execute(text("ALTER TABLE detections ADD COLUMN plate_confidence FLOAT"))
            conn.commit()
        except Exception:
            pass

app = FastAPI(title="SMART ANPR API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in ALLOWED_ORIGINS if origin.strip()] or ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _file_url(path: str | None) -> str | None:
    return f"/media/{Path(path).name}" if path else None


def _serialize(row: Detection) -> dict:
    return {
        "id": row.id,
        "vehicle_type": row.vehicle_type,
        "vehicle_confidence": row.vehicle_confidence,
        "plate_text": row.plate_text,
        "plate_confidence": getattr(row, "plate_confidence", None),
        "ocr_confidence": row.ocr_confidence,
        "status": row.status,
        "source": row.source,
        "image_url": _file_url(row.image_path),
        "annotated_url": _file_url(row.annotated_path),
        "plate_image_url": _file_url(row.plate_image_path),
        "vehicle_box": json.loads(row.vehicle_box) if row.vehicle_box else [],
        "plate_box": json.loads(row.plate_box) if row.plate_box else None,
        "created_at": row.created_at.isoformat(),
    }


def _safe_name(filename: str | None) -> str:
    suffix = Path(filename or "").suffix.lower()
    if not re.fullmatch(r"\.[a-z0-9]{1,8}", suffix):
        suffix = ".bin"
    return f"{uuid.uuid4().hex}{suffix}"


async def _read_upload(upload: UploadFile) -> bytes:
    data = await upload.read(MAX_UPLOAD_BYTES + 1)
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, f"Upload exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    return data


async def _analyze_frame(frame: np.ndarray) -> tuple[list[DetectionResult], np.ndarray]:
    try:
        return await run_in_threadpool(pipeline.analyze, frame)
    except ModelUnavailableError as exc:
        raise HTTPException(503, str(exc)) from exc


def _persist_frame(
    frame: np.ndarray,
    source: str,
    db: Session,
    results: list[DetectionResult],
    annotated: np.ndarray,
) -> list[dict]:
    if not results:
        return []
    original_name = _safe_name("capture.jpg")
    annotated_name = _safe_name("annotated.jpg")
    if not cv2.imwrite(str(UPLOAD_DIR / original_name), frame):
        raise HTTPException(500, "Could not save uploaded image.")
    if results and not cv2.imwrite(str(UPLOAD_DIR / annotated_name), annotated):
        raise HTTPException(500, "Could not save annotated image.")

    saved = []
    for result in results:
        plate_name = None
        if result.plate_crop is not None:
            plate_name = _safe_name("plate.jpg")
            if not cv2.imwrite(str(UPLOAD_DIR / plate_name), result.plate_crop):
                raise HTTPException(500, "Could not save cropped plate image.")
        row = Detection(
            vehicle_type=result.vehicle_type,
            vehicle_confidence=result.vehicle_confidence,
            plate_text=result.plate_text,
            plate_confidence=getattr(result, "plate_confidence", None),
            ocr_confidence=result.ocr_confidence,
            status=result.status,
            source=source,
            image_path=original_name,
            annotated_path=annotated_name if results else None,
            plate_image_path=plate_name,
            vehicle_box=box_json(result.vehicle_box) or "[]",
            plate_box=box_json(result.plate_box),
        )
        db.add(row)
        db.flush()
        saved.append(_serialize(row))
    db.commit()
    return saved


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok", "models": pipeline.readiness()}


@app.get("/api/stats")
def stats(db: Annotated[Session, Depends(get_db)]) -> dict:
    total = db.scalar(select(func.count(Detection.id))) or 0
    recognized = db.scalar(select(func.count(Detection.id)).where(Detection.status == "recognized")) or 0
    vehicle_counts = dict(
        db.execute(select(Detection.vehicle_type, func.count()).group_by(Detection.vehicle_type)).all()
    )
    today = datetime.now(timezone.utc).date()
    first_day = today - timedelta(days=6)
    daily_start = datetime.combine(first_day, time.min, timezone.utc)
    daily_rows = db.execute(
        select(func.date(Detection.created_at), func.count())
        .where(Detection.created_at >= daily_start)
        .group_by(func.date(Detection.created_at))
    ).all()
    daily_counts = {str(day): int(count) for day, count in daily_rows}
    hour_expression = (
        func.strftime("%H", Detection.created_at)
        if engine.dialect.name == "sqlite"
        else func.extract("hour", Detection.created_at)
    )
    hourly_rows = db.execute(
        select(hour_expression, func.count()).group_by(hour_expression)
    ).all()
    hourly_counts = {int(hour): int(count) for hour, count in hourly_rows}
    return {
        "total_vehicles": total,
        "plates_recognized": recognized,
        "cars_detected": vehicle_counts.get("Car", 0),
        "bikes_detected": vehicle_counts.get("Bike", 0),
        "trucks_detected": vehicle_counts.get("Truck", 0),
        "recognition_rate": round((recognized / total) * 100, 1) if total else 0,
        "by_vehicle_type": vehicle_counts,
        "daily_counts": [
            {"date": (first_day + timedelta(days=offset)).isoformat(),
             "count": daily_counts.get((first_day + timedelta(days=offset)).isoformat(), 0)}
            for offset in range(7)
        ],
        "hourly_counts": [{"hour": hour, "count": hourly_counts.get(hour, 0)} for hour in range(24)],
    }


def _filtered_query(
    q: str | None, vehicle_type: str | None, date_from: date | None, date_to: date | None
):
    statement = select(Detection)
    if q:
        statement = statement.where(Detection.plate_text.ilike(f"%{q.strip()}%"))
    if vehicle_type:
        statement = statement.where(Detection.vehicle_type == vehicle_type)
    if date_from:
        statement = statement.where(Detection.created_at >= datetime.combine(date_from, time.min, timezone.utc))
    if date_to:
        statement = statement.where(Detection.created_at <= datetime.combine(date_to, time.max, timezone.utc))
    return statement


@app.get("/api/detections")
def list_detections(
    db: Annotated[Session, Depends(get_db)],
    q: str | None = None,
    vehicle_type: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    sort_by: str = Query("created_at", pattern="^(created_at|vehicle_type|plate_text|vehicle_confidence)$"),
    order: str = Query("desc", pattern="^(asc|desc)$"),
) -> dict:
    statement = _filtered_query(q, vehicle_type, date_from, date_to)
    total = db.scalar(select(func.count()).select_from(statement.subquery())) or 0
    sort_column = getattr(Detection, sort_by)
    statement = statement.order_by(sort_column.asc() if order == "asc" else sort_column.desc())
    rows = db.scalars(statement.offset((page - 1) * page_size).limit(page_size)).all()
    return {"items": [_serialize(row) for row in rows], "total": total, "page": page, "page_size": page_size}


@app.get("/api/detections/export.csv")
def export_csv(
    db: Annotated[Session, Depends(get_db)],
    q: str | None = None,
    vehicle_type: str | None = None,
    date_from: date | None = None,
    date_to: date | None = None,
):
    rows = db.scalars(_filtered_query(q, vehicle_type, date_from, date_to).order_by(Detection.created_at.desc())).all()
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(["id", "vehicle_type", "plate_text", "vehicle_confidence", "ocr_confidence", "status", "source", "created_at"])
    for row in rows:
        writer.writerow([row.id, row.vehicle_type, row.plate_text or "", row.vehicle_confidence,
                         row.ocr_confidence if row.ocr_confidence is not None else "", row.status, row.source, row.created_at.isoformat()])
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="anpr-detections.csv"'},
    )


@app.get("/api/detections/{detection_id}")
def get_detection(detection_id: int, db: Annotated[Session, Depends(get_db)]) -> dict:
    row = db.get(Detection, detection_id)
    if row is None:
        raise HTTPException(404, "Detection not found.")
    return _serialize(row)


@app.post("/api/detections/image")
async def analyze_image(
    file: UploadFile = File(...), db: Session = Depends(get_db)
) -> dict:
    raw = await _read_upload(file)
    frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise HTTPException(415, "The uploaded file is not a supported image.")
    results, annotated = await _analyze_frame(frame)
    if not results:
        # Save original frame so frontend can still display and download it
        original_name = _safe_name("capture.jpg")
        cv2.imwrite(str(UPLOAD_DIR / original_name), frame)
        return {
            "detections": [],
            "image_url": _file_url(original_name),
            "annotated_url": _file_url(original_name),
            "message": "Number Plate Not Found",
            "status": "plate_not_detected",
        }

    detections = _persist_frame(frame, "image", db, results, annotated)
    first_item = detections[0] if detections else None
    return {
        "detections": detections,
        "image_url": first_item.get("image_url") if first_item else None,
        "annotated_url": first_item.get("annotated_url") if first_item else None,
        "message": None,
    }


@app.get("/api/test-samples")
def get_test_samples() -> list[dict]:
    samples_dir = DATA_DIR / "test_samples"
    if not samples_dir.is_dir():
        return []
    items = []
    for f in sorted(samples_dir.glob("*.jpg")):
        items.append({
            "name": f.name,
            "url": f"/media/samples/{f.name}",
            "size": f.stat().st_size,
        })
    return items


@app.get("/media/samples/{filename}")
def get_sample_media(filename: str):
    if Path(filename).name != filename:
        raise HTTPException(400, "Invalid media path.")
    samples_dir = DATA_DIR / "test_samples"
    path = samples_dir / filename
    if not path.is_file():
        raise HTTPException(404, "Sample image not found.")
    return FileResponse(path)


@app.post("/api/detections/video")
async def analyze_video(
    file: UploadFile = File(...), db: Session = Depends(get_db)
) -> dict:
    raw = await _read_upload(file)
    video_path = UPLOAD_DIR / _safe_name(file.filename)
    video_path.write_bytes(raw)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        video_path.unlink(missing_ok=True)
        raise HTTPException(415, "The uploaded file is not a supported video.")
    all_detections: list[dict] = []
    frame_index = 0
    frame_stride = 10
    try:
        while frame_index < 30_000:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_stride == 0:
                results, annotated = await _analyze_frame(frame)
                all_detections.extend(_persist_frame(frame, "video", db, results, annotated))
            frame_index += 1
    finally:
        capture.release()
        video_path.unlink(missing_ok=True)
    return {"detections": all_detections, "frames_sampled": (frame_index + frame_stride - 1) // frame_stride,
            "message": "No vehicles detected in sampled frames." if not all_detections else None}


@app.get("/media/{filename}")
def get_media(filename: str):
    if Path(filename).name != filename:
        raise HTTPException(400, "Invalid media path.")
    path = UPLOAD_DIR / filename
    if not path.is_file():
        raise HTTPException(404, "Media not found.")
    return FileResponse(path)


@app.websocket("/api/ws/live")
async def live_detection(websocket: WebSocket):
    await websocket.accept()
    last_persisted = 0.0
    try:
        while True:
            payload = await websocket.receive_json()
            encoded = payload.get("image")
            if not isinstance(encoded, str):
                await websocket.send_json({"error": "Expected a base64-encoded JPEG in the image field."})
                continue
            if len(encoded) > 12 * 1024 * 1024:
                await websocket.send_json({"error": "Live frame exceeds the 9 MB limit."})
                continue
            try:
                raw = base64.b64decode(encoded, validate=True)
                frame = cv2.imdecode(np.frombuffer(raw, dtype=np.uint8), cv2.IMREAD_COLOR)
                if frame is None:
                    raise ValueError("Image data is not a supported frame.")
                results, annotated = await run_in_threadpool(pipeline.analyze, frame)
                persisted_ids: list[int] = []
                persistence_error = None
                if results and time.monotonic() - last_persisted >= 5:
                    try:
                        with SessionLocal() as db:
                            saved = _persist_frame(frame, "webcam", db, results, annotated)
                        persisted_ids = [item["id"] for item in saved]
                        last_persisted = time.monotonic()
                    except (SQLAlchemyError, OSError, HTTPException) as exc:
                        persistence_error = exc.detail if isinstance(exc, HTTPException) else str(exc)
                        logger.exception("Could not save live detection results")
                await websocket.send_json({
                    "detections": [
                        {"vehicle_type": result.vehicle_type, "vehicle_confidence": result.vehicle_confidence,
                         "plate_text": result.plate_text, "ocr_confidence": result.ocr_confidence,
                         "status": result.status, "vehicle_box": result.vehicle_box, "plate_box": result.plate_box}
                        for result in results
                    ],
                    "annotated_image": base64.b64encode(encode_jpeg(annotated)).decode("ascii"),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "persisted_ids": persisted_ids,
                    "persistence_error": persistence_error,
                })
            except ModelUnavailableError as exc:
                await websocket.send_json({"error": str(exc)})
            except (ValueError, cv2.error) as exc:
                await websocket.send_json({"error": str(exc)})
    except WebSocketDisconnect:
        logger.info("Live detection client disconnected")
