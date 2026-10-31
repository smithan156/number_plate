import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.getenv("ANPR_DATA_DIR", BACKEND_DIR / "data")).resolve()
UPLOAD_DIR = DATA_DIR / "uploads"
DATABASE_URL = os.getenv("ANPR_DATABASE_URL", f"sqlite:///{(DATA_DIR / 'anpr.db').as_posix()}")

def _resolve_model_path(env_var: str, default_filename: str) -> str:
    from_env = os.getenv(env_var)
    if from_env and Path(from_env).is_file():
        return str(Path(from_env).resolve())
    candidates = [
        BACKEND_DIR / "models" / default_filename,
        BACKEND_DIR / default_filename,
        BACKEND_DIR.parent / default_filename,
        Path(default_filename),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    return from_env or default_filename

VEHICLE_MODEL_PATH = _resolve_model_path("VEHICLE_MODEL_PATH", "yolo11n.pt")
PLATE_MODEL_PATH = _resolve_model_path("PLATE_MODEL_PATH", "plate_model.pt")
OCR_MIN_CONFIDENCE = float(os.getenv("OCR_MIN_CONFIDENCE", "0.22"))
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(100 * 1024 * 1024)))
ALLOWED_ORIGINS = os.getenv(
    "ANPR_ALLOWED_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
).split(",")

DATA_DIR.mkdir(parents=True, exist_ok=True)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
