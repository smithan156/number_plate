# SMART ANPR

A full-stack ANPR starter application with a React + TypeScript dashboard and a FastAPI service. It supports image and video uploads, webcam snapshots over WebSockets, persistent detection history, filters, CSV export, and real-time statistics.

> **Model requirement:** vehicle and plate results are generated only when you configure real model weights. This repository does not include trained weights. If either detector is unavailable, the API reports an explicit error; it never creates sample plate numbers or successful detections.

## Project structure

```text
number_plate/
├── backend/
│   ├── app/
│   │   ├── services/anpr.py
│   │   ├── config.py
│   │   ├── database.py
│   │   ├── main.py
│   │   ├── package-lock.json
│   └──  models.py
│   ├── data/uploads/          # Created at runtime; ignored by git
│   └── requirements.txt
├── frontend/
│   ├── src/App.tsx
│   ├── src/index.css
│   ├── src/main.tsx
│   ├── index.html
│   ├── package.json
│   ├── postcss.config.js
│   ├── tailwind.config.js
│   ├── tsconfig.json
│   └── vite.config.ts
└── .gitignore
```

## Requirements

- Python 3.10–3.12
- Node.js 20+
- A trained Ultralytics YOLO vehicle model and a dedicated number-plate detector
- OpenCV-compatible image/video codecs

The default vehicle model is `yolo11n.pt`, downloaded by Ultralytics on first use when network access is available. The plate detector is domain-specific and must be supplied separately.

## Install and configure

### Backend

From the repository root, create and activate a virtual environment, then install:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r backend\requirements.txt
```

The models are already located under `backend/models/`:
- `yolo11n.pt` (Vehicle detector)
- `plate_model.pt` (Number plate detector)

Start the FastAPI backend with:

```powershell
.\.venv\Scripts\Activate.ps1
uvicorn app.main:app --app-dir backend --reload --port 8000
```

The backend listens on `http://localhost:8000`; interactive API docs are at `http://localhost:8000/docs`. Its SQLite database and uploaded images are created under `backend/data/`.

To use PostgreSQL instead, install a PostgreSQL driver (for example `psycopg[binary]`), set `ANPR_DATABASE_URL` to a SQLAlchemy URL such as `postgresql+psycopg://user:password@localhost:5432/anpr`, and create the database first.

### Frontend

In another terminal:

```powershell
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`). Set `VITE_API_URL` if the API is hosted elsewhere; the default is `http://localhost:8000`.

## Models and training

1. Use Ultralytics to train a vehicle detector on a dataset whose labels include the vehicle classes you need. Standard COCO checkpoints cover car, motorcycle, bus, truck, bicycle, and boat; they do not reliably distinguish scooter, auto rickshaw, lorry, tractor, or ship. Add representative labeled data and train custom classes for those distinctions.
2. Train a separate plate detector on representative Indian plates, including varied lighting, tilt, blur, occlusion, and regional formats. Export its best checkpoint as a YOLO-compatible `.pt` file.
3. Configure `VEHICLE_MODEL_PATH` and `PLATE_MODEL_PATH` to the checkpoints. Ultralytics class names are used to map detections to the supported vehicle categories.
4. Evaluate both detectors and OCR on a held-out, permissioned dataset before production. Recognition rate in the dashboard is the proportion of saved detections with a validated, sufficiently confident OCR result; it is **not** a measured character-level accuracy.

Example training commands (adjust dataset YAML and hyperparameters for your data):

```powershell
yolo detect train model=yolo11n.pt data=vehicle_dataset.yaml imgsz=640 epochs=100
yolo detect train model=yolo11n.pt data=plate_dataset.yaml imgsz=960 epochs=150
```

EasyOCR reads plate crops from the dedicated detector. The service enhances crops with OpenCV, applies Indian registration-format validation, and withholds the plate text when OCR confidence is below the configured threshold. Tune and validate this threshold on your own evaluation data; it is not a substitute for human review.

## Features and API

- `POST /api/detections/image` — analyze one uploaded image.
- `POST /api/detections/video` — analyze every tenth frame from an uploaded video, up to 30,000 frames per file.
- `GET /api/detections` — actual, paginated database records; supports `q`, `vehicle_type`, `date_from`, `date_to`, `page`, `page_size`, `sort_by`, and `order`.
- `GET /api/detections/{id}` — retrieve one detection.
- `GET /api/detections/export.csv` — export matching database records.
- `GET /api/stats` — live totals, daily counts, and hourly distribution derived from stored detection rows.
- `GET /api/health` — API and model configuration status.
- `WS /api/ws/live` — send JSON frames as `{ "image": "<base64 JPEG>" }`; receive detection results or explicit errors. Detected vehicles are saved at most once every five seconds per camera connection.

Image, video, and webcam analysis need both model files configured. The webcam uses `getUserMedia` and sends JPEG snapshots through the WebSocket; allow camera access in the browser. The current webcam view is snapshot-based rather than a continuous high-frame-rate stream.

## Production notes

This is a development-ready foundation, not a hardened public surveillance deployment. Before deployment, add authentication and authorization, HTTPS/WSS, upload quotas and retention policy, model-serving capacity limits, database migrations, audit logging, observability, and privacy/legal review. Keep weight files and captured images outside version control.
