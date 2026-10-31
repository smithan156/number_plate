from __future__ import annotations

import json
import logging
import re
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    import easyocr
    from ultralytics import YOLO

from app.config import OCR_MIN_CONFIDENCE, PLATE_MODEL_PATH, VEHICLE_MODEL_PATH

logger = logging.getLogger("smart-anpr.pipeline")

VEHICLE_CLASSES = {
    "bike": "Bike",
    "bicycle": "Bike",
    "motorcycle": "Bike",
    "motorbike": "Bike",
    "scooter": "Bike",
    "car": "Car",
    "suv": "Car",
    "sedan": "Car",
    "bus": "Bus",
    "truck": "Truck",
    "lorry": "Truck",
    "van": "Van",
    "minivan": "Van",
    "auto": "Auto",
    "auto rickshaw": "Auto",
    "autorickshaw": "Auto",
    "rickshaw": "Auto",
    "tractor": "Tractor",
    "train": "Train",
}

INDIAN_PLATE_PATTERN = re.compile(
    r"^(?:[A-Z]{2}\d{1,2}[A-Z]{1,3}\d{1,4}|"
    r"[A-Z]{2}\d{2}[A-Z]{0,3}\d{1,4}|"
    r"\d{2}BH\d{4}[A-Z]{1,2})$"
)

COUNTRY_AND_IGNORE_WORDS = {
    "BRASIL", "BRAZIL", "MEXICO", "IND", "INDIA", "CALIFORNIA", "TEXAS", "NY", "FLORIDA",
    "CHRYSLER", "HONDA", "TOYOTA", "HYUNDAI", "SUZUKI", "MARUTI", "TATA", "MAHINDRA",
    "FORD", "CHEVROLET", "BMW", "MERCEDES", "AUDI", "VOLKSWAGEN", "NISSAN", "MOTORS",
    "DEALER", "PLATE", "OBJECT", "DETECTION", "VEHICLE", "STATE", "GOVT", "GOVERNMENT",
    "POLICE", "HIGHWAY", "TRANSPORT", "AUTHORITY", "DEALERSHIP", "AUTO", "CARS", "CAR",
    "BR", "USA", "MEX", "UK", "EU", "D", "F", "NL", "PL", "RO", "CH", "A", "I", "E",
}


class ModelUnavailableError(RuntimeError):
    pass


@dataclass
class DetectionResult:
    vehicle_type: str
    vehicle_confidence: float
    plate_text: str | None
    ocr_confidence: float | None
    status: str
    vehicle_box: list[int]
    plate_box: list[int] | None
    plate_crop: np.ndarray | None
    plate_confidence: float | None = None


class ANPRPipeline:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._vehicle_model: YOLO | None = None
        self._plate_model: YOLO | None = None
        self._reader: easyocr.Reader | None = None

    def readiness(self) -> dict[str, bool]:
        plate_ready = bool(PLATE_MODEL_PATH and Path(PLATE_MODEL_PATH).is_file())
        vehicle_ready = bool(VEHICLE_MODEL_PATH and Path(VEHICLE_MODEL_PATH).is_file())
        return {
            "vehicle_model_configured": vehicle_ready,
            "plate_model_configured": plate_ready,
            "ocr_loaded": self._reader is not None,
        }

    def _resolve_model(self, configured_path: str, fallback_filename: str) -> str:
        candidates = [
            Path(configured_path),
            Path(__file__).resolve().parents[2] / "models" / fallback_filename,
            Path(__file__).resolve().parents[2] / fallback_filename,
            Path(__file__).resolve().parents[1] / "models" / fallback_filename,
            Path(__file__).resolve().parents[1] / fallback_filename,
            Path(fallback_filename),
        ]
        for p in candidates:
            if p.is_file():
                return str(p.resolve())
        return configured_path

    def _load_models(self) -> None:
        v_path = self._resolve_model(VEHICLE_MODEL_PATH, "yolo11n.pt")
        p_path = self._resolve_model(PLATE_MODEL_PATH, "plate_model.pt")

        if not Path(v_path).is_file() and not Path(VEHICLE_MODEL_PATH).is_file():
            raise ModelUnavailableError(f"Vehicle detector weights not found at '{v_path}'.")
        if not Path(p_path).is_file() and not Path(PLATE_MODEL_PATH).is_file():
            raise ModelUnavailableError(
                f"Number-plate detector weights not found at '{p_path}'. Configure a valid PLATE_MODEL_PATH."
            )

        try:
            import easyocr
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModelUnavailableError(
                "AI packages unavailable. Install requirements before running detection."
            ) from exc

        try:
            if self._vehicle_model is None:
                actual_v = v_path if Path(v_path).is_file() else VEHICLE_MODEL_PATH
                self._vehicle_model = YOLO(actual_v)
            if self._plate_model is None:
                actual_p = p_path if Path(p_path).is_file() else PLATE_MODEL_PATH
                self._plate_model = YOLO(actual_p)
            if self._reader is None:
                self._reader = easyocr.Reader(["en"], gpu=False)
        except Exception as exc:
            raise ModelUnavailableError(f"Unable to load ANPR models: {exc}") from exc

    def analyze(self, frame: np.ndarray) -> tuple[list[DetectionResult], np.ndarray]:
        if frame is None or frame.size == 0:
            raise ValueError("The uploaded frame is empty or is not a supported image.")

        with self._lock:
            self._load_models()
            assert self._vehicle_model is not None
            assert self._plate_model is not None
            assert self._reader is not None

            # 1. Detect vehicles
            v_pred = self._vehicle_model.predict(frame, verbose=False, conf=0.25)[0]
            # 2. Detect license plates
            p_pred = self._plate_model.predict(frame, verbose=False, conf=0.15)[0]

            v_boxes = v_pred.boxes
            p_boxes = p_pred.boxes

            plate_candidates: list[tuple[list[int], float, np.ndarray]] = []
            if p_boxes is not None:
                for box in p_boxes:
                    coords = [int(v) for v in box.xyxy[0].tolist()]
                    x1, y1, x2, y2 = self._clip_box(coords, frame.shape)
                    pw = x2 - x1
                    ph = y2 - y1
                    if pw < 8 or ph < 6:
                        continue
                    # Apply margin padding so characters near bounding box edges aren't clipped
                    pad_x = max(2, int(pw * 0.08))
                    pad_y = max(2, int(ph * 0.12))
                    px1 = max(0, x1 - pad_x)
                    py1 = max(0, y1 - pad_y)
                    px2 = min(frame.shape[1], x2 + pad_x)
                    py2 = min(frame.shape[0], y2 + pad_y)
                    crop = frame[py1:py2, px1:px2].copy()
                    plate_candidates.append(([x1, y1, x2, y2], float(box.conf[0]), crop))

            matched_plates: set[int] = set()
            results: list[DetectionResult] = []
            annotated = frame.copy()

            # Process detected vehicles
            if v_boxes is not None and len(v_boxes) > 0:
                for vehicle in v_boxes:
                    class_id = int(vehicle.cls[0])
                    raw_name = re.sub(
                        r"[_-]+", " ", str(v_pred.names.get(class_id, "vehicle")).lower()
                    ).strip()
                    vehicle_type = VEHICLE_CLASSES.get(raw_name)
                    if vehicle_type is None:
                        continue

                    coords = [int(v) for v in vehicle.xyxy[0].tolist()]
                    vx1, vy1, vx2, vy2 = self._clip_box(coords, frame.shape)
                    vbox = [vx1, vy1, vx2, vy2]

                    # Find matching plate candidates
                    associated: list[tuple[int, tuple[list[int], float, np.ndarray]]] = []
                    for idx, candidate in enumerate(plate_candidates):
                        if self._is_associated(candidate[0], vbox):
                            associated.append((idx, candidate))

                    best_plate_crop: np.ndarray | None = None
                    best_plate_box: list[int] | None = None
                    best_plate_conf: float | None = None

                    if associated:
                        best_idx, best_item = max(associated, key=lambda item: item[1][1])
                        matched_plates.add(best_idx)
                        best_plate_box = best_item[0]
                        best_plate_conf = round(float(best_item[1]), 3)
                        best_plate_crop = best_item[2]
                    else:
                        # Fallback: check vehicle region for plate if detector missed it
                        fallback_box, fallback_crop = self._find_fallback_plate(frame, vbox)
                        if fallback_box is not None and fallback_crop is not None:
                            best_plate_box = fallback_box
                            best_plate_crop = fallback_crop
                            best_plate_conf = 0.50

                    plate_text = None
                    ocr_conf = None
                    if best_plate_crop is not None and best_plate_crop.size > 0:
                        plate_text, ocr_conf = self._read_plate(best_plate_crop)

                    status = (
                        "recognized"
                        if plate_text and (ocr_conf or 0) >= OCR_MIN_CONFIDENCE
                        else "low_confidence"
                        if best_plate_crop is not None
                        else "plate_not_detected"
                    )

                    res = DetectionResult(
                        vehicle_type=vehicle_type,
                        vehicle_confidence=float(vehicle.conf[0]),
                        plate_text=plate_text,
                        ocr_confidence=ocr_conf,
                        status=status,
                        vehicle_box=vbox,
                        plate_box=best_plate_box,
                        plate_crop=best_plate_crop,
                        plate_confidence=best_plate_conf,
                    )
                    results.append(res)
                    self._draw_detection(annotated, res)

            # If plates were detected but no vehicle matched (e.g., close-up image)
            for idx, candidate in enumerate(plate_candidates):
                if idx in matched_plates:
                    continue
                p_box, p_conf, p_crop = candidate
                plate_text, ocr_conf = self._read_plate(p_crop)
                status = (
                    "recognized"
                    if plate_text and (ocr_conf or 0) >= OCR_MIN_CONFIDENCE
                    else "low_confidence"
                )
                res = DetectionResult(
                    vehicle_type="Vehicle",
                    vehicle_confidence=float(p_conf),
                    plate_text=plate_text,
                    ocr_confidence=ocr_conf,
                    status=status,
                    vehicle_box=p_box,
                    plate_box=p_box,
                    plate_crop=p_crop,
                    plate_confidence=round(float(p_conf), 3),
                )
                results.append(res)
            # Prioritize vehicles with recognized plates first
            results.sort(
                key=lambda r: (
                    2 if r.plate_text else (1 if r.plate_box else 0),
                    r.ocr_confidence or 0.0,
                    r.vehicle_confidence,
                ),
                reverse=True,
            )

            return results, annotated

    def _read_plate(self, crop: np.ndarray) -> tuple[str | None, float | None]:
        assert self._reader is not None
        if crop is None or crop.size == 0:
            return None, None

        h, w = crop.shape[:2]
        if h < 8 or w < 12:
            return None, None

        # Scale up so characters have sufficient pixel height for OCR (at least 140px height)
        scale = max(1.0, 140.0 / max(h, 1))
        target_w = max(1, int(w * scale))
        target_h = max(1, int(h * scale))
        scaled_crop = cv2.resize(crop, (target_w, target_h), interpolation=cv2.INTER_CUBIC)

        gray = cv2.cvtColor(scaled_crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray)
        _, otsu = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, otsu_inv = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        bilateral = cv2.bilateralFilter(gray, 9, 75, 75)

        # Multi-pass preprocessing candidates
        images_to_try = [scaled_crop, clahe, otsu, otsu_inv, bilateral]

        # Also try perspective correction if valid
        corrected = self._perspective_correct(crop)
        if corrected is not None and corrected.size > 0 and corrected.shape != crop.shape:
            ch, cw = corrected.shape[:2]
            c_scale = max(1.0, 140.0 / max(ch, 1))
            c_scaled = cv2.resize(corrected, (int(cw * c_scale), int(ch * c_scale)), interpolation=cv2.INTER_CUBIC)
            c_gray = cv2.cvtColor(c_scaled, cv2.COLOR_BGR2GRAY)
            images_to_try.append(c_gray)

        pass_candidates: list[list[dict]] = []

        for img in images_to_try:
            try:
                results = self._reader.readtext(img, detail=1, paragraph=False)
                if not results:
                    continue
                cleaned = []
                for bbox, text, conf in results:
                    c_text = re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()
                    if not c_text or c_text in COUNTRY_AND_IGNORE_WORDS:
                        continue
                    if len(c_text) == 1 and float(conf) < 0.65:
                        continue
                    if float(conf) < 0.12:
                        continue
                    ys = [p[1] for p in bbox]
                    xs = [p[0] for p in bbox]
                    bh = max(ys) - min(ys)
                    bw = max(xs) - min(xs)
                    ym = sum(ys) / len(ys)
                    xm = sum(xs) / len(xs)
                    cleaned.append({
                        "text": c_text,
                        "conf": float(conf),
                        "h": bh,
                        "w": bw,
                        "ym": ym,
                        "xm": xm,
                    })
                if cleaned:
                    pass_candidates.append(cleaned)
            except Exception:
                pass

        best_result: str | None = None
        best_conf: float = 0.0
        best_score: float = -1.0

        for tokens in pass_candidates:
            max_h = max(t["h"] for t in tokens)
            # Retain only characters of the dominant line (at least 45% of maximum character height)
            main_tokens = [t for t in tokens if t["h"] >= max_h * 0.45]
            if not main_tokens:
                continue

            # Group tokens into horizontal lines
            lines: list[list[dict]] = []
            for t in sorted(main_tokens, key=lambda item: item["ym"]):
                placed = False
                for line in lines:
                    if abs(line[0]["ym"] - t["ym"]) < max_h * 0.40:
                        line.append(t)
                        placed = True
                        break
                if not placed:
                    lines.append([t])

            for line in lines:
                line.sort(key=lambda item: item["xm"])
                line_words = [item["text"] for item in line]
                full_str = " ".join(line_words)
                norm = re.sub(r"[^A-Z0-9]", "", full_str)
                if len(norm) < 3:
                    continue
                avg_conf = sum(item["conf"] for item in line) / len(line)
                has_letters = bool(re.search(r"[A-Z]", norm))
                has_digits = bool(re.search(r"[0-9]", norm))
                # Extra weighting for realistic license plate structure (alphanumeric combination)
                score = avg_conf + (0.25 if (has_letters and has_digits) else 0.0) + min(len(norm), 10) * 0.02
                if score > best_score:
                    best_score = score
                    best_conf = avg_conf
                    best_result = full_str

        if best_result is not None and best_conf >= 0.18:
            return best_result, round(best_conf, 2)

        return None, round(best_conf, 2) if best_conf > 0 else None

    def _find_fallback_plate(
        self, frame: np.ndarray, vehicle_box: list[int]
    ) -> tuple[list[int] | None, np.ndarray | None]:
        vx1, vy1, vx2, vy2 = vehicle_box
        vw = vx2 - vx1
        vh = vy2 - vy1
        if vw < 30 or vh < 30:
            return None, None

        # Inspect lower 50% of vehicle (common bumper plate area)
        roi_y1 = vy1 + int(vh * 0.45)
        roi_y2 = vy2
        roi_x1 = vx1 + int(vw * 0.1)
        roi_x2 = vx2 - int(vw * 0.1)

        roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
        if roi.size == 0 or self._reader is None:
            return None, None

        try:
            texts = self._reader.readtext(roi, detail=1, paragraph=False)
            for bbox, text, conf in texts:
                clean = re.sub(r"[^A-Za-z0-9]", "", str(text)).upper()
                if len(clean) >= 4 and float(conf) >= 0.4:
                    xs = [int(p[0]) for p in bbox]
                    ys = [int(p[1]) for p in bbox]
                    px1 = max(0, roi_x1 + min(xs) - 4)
                    py1 = max(0, roi_y1 + min(ys) - 4)
                    px2 = min(frame.shape[1], roi_x1 + max(xs) + 4)
                    py2 = min(frame.shape[0], roi_y1 + max(ys) + 4)
                    crop = frame[py1:py2, px1:px2].copy()
                    return [px1, py1, px2, py2], crop
        except Exception:
            pass

        return None, None

    @staticmethod
    def _draw_detection(annotated: np.ndarray, res: DetectionResult) -> None:
        vx1, vy1, vx2, vy2 = res.vehicle_box
        # Vehicle box: bright electric blue / cyan
        v_color = (245, 160, 30)
        cv2.rectangle(annotated, (vx1, vy1), (vx2, vy2), v_color, 2)

        v_label = f"{res.vehicle_type.upper()} ({int(res.vehicle_confidence * 100)}%)"
        (tw, th), _ = cv2.getTextSize(v_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        top_y = max(vy1 - 6, th + 8)
        cv2.rectangle(annotated, (vx1, top_y - th - 6), (vx1 + tw + 10, top_y + 2), v_color, -1)
        cv2.putText(
            annotated, v_label, (vx1 + 5, top_y - 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 15, 25), 2, cv2.LINE_AA
        )

        if res.plate_box:
            px1, py1, px2, py2 = res.plate_box
            # Plate box: vibrant neon green
            p_color = (50, 235, 120)
            cv2.rectangle(annotated, (px1, py1), (px2, py2), p_color, 3)

            conf_str = f" ({int(res.plate_confidence * 100)}%)" if res.plate_confidence else ""
            p_label = f"PLATE: {res.plate_text or 'LOCATED'}{conf_str}"
            (ptw, pth), _ = cv2.getTextSize(p_label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            p_top_y = max(py1 - 6, pth + 8)
            cv2.rectangle(annotated, (px1, p_top_y - pth - 6), (px1 + ptw + 10, p_top_y + 2), p_color, -1)
            cv2.putText(
                annotated, p_label, (px1 + 5, p_top_y - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 25, 15), 2, cv2.LINE_AA
            )

    @staticmethod
    def _perspective_correct(crop: np.ndarray) -> np.ndarray:
        if crop.shape[0] < 15 or crop.shape[1] < 30:
            return crop
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 130)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        crop_area = crop.shape[0] * crop.shape[1]
        for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:10]:
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
            if len(polygon) != 4 or cv2.contourArea(polygon) < crop_area * 0.35:
                continue
            points = polygon.reshape(4, 2).astype("float32")
            sums = points.sum(axis=1)
            differences = np.diff(points, axis=1).reshape(4)
            ordered = np.array(
                [
                    points[np.argmin(sums)],
                    points[np.argmin(differences)],
                    points[np.argmax(sums)],
                    points[np.argmax(differences)],
                ],
                dtype="float32",
            )
            top_left, top_right, bottom_right, bottom_left = ordered
            width = max(
                np.linalg.norm(bottom_right - bottom_left),
                np.linalg.norm(top_right - top_left),
            )
            height = max(
                np.linalg.norm(top_right - bottom_right),
                np.linalg.norm(top_left - bottom_left),
            )
            if width < 15 or height < 10:
                continue
            destination = np.array(
                [[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]],
                dtype="float32",
            )
            transform = cv2.getPerspectiveTransform(ordered, destination)
            return cv2.warpPerspective(crop, transform, (int(width), int(height)))
        return crop

    @staticmethod
    def _clip_box(box: list[int], shape: tuple[int, ...]) -> tuple[int, int, int, int]:
        height, width = shape[:2]
        x1, y1, x2, y2 = box
        return max(0, x1), max(0, y1), min(width, x2), min(height, y2)

    @staticmethod
    def _is_associated(plate: list[int], vehicle: list[int]) -> bool:
        px1, py1, px2, py2 = plate
        vx1, vy1, vx2, vy2 = vehicle
        center_x, center_y = (px1 + px2) / 2, (py1 + py2) / 2
        # Check center inside
        if vx1 <= center_x <= vx2 and vy1 <= center_y <= vy2:
            return True
        # Check intersection
        ix1 = max(px1, vx1)
        iy1 = max(py1, vy1)
        ix2 = min(px2, vx2)
        iy2 = min(py2, vy2)
        if ix2 > ix1 and iy2 > iy1:
            i_area = (ix2 - ix1) * (iy2 - iy1)
            p_area = max(1, (px2 - px1) * (py2 - py1))
            return (i_area / p_area) >= 0.2
        return False


pipeline = ANPRPipeline()


def encode_jpeg(image: np.ndarray) -> bytes:
    success, encoded = cv2.imencode(".jpg", image)
    if not success:
        raise ValueError("Could not encode the processed image.")
    return encoded.tobytes()


def box_json(box: list[int] | None) -> str | None:
    return json.dumps(box) if box is not None else None
