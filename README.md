# Face Detection & Recognition (Jetson-ready)

## Highlights
- GPU-aware model loading with CUDA/TensorRT/DirectML when available; CPU fallback is automatic.
- Improved cosine-similarity recognition for better tolerance to slight side profiles.
- Preloaded models and tuned JPEG encoding to reduce latency and memory churn.
- Production-friendly logging with `APP_LOG_LEVEL` (INFO by default).
- Refreshed NiceGUI layout with clear status chips, directory view, and upload tester.

## Install

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

### Jetson Orin Nano (TensorRT)
- If you want GPU execution, replace the CPU wheel with the TensorRT-enabled one that matches your JetPack:
	- `pip uninstall -y onnxruntime`
	- `pip install --extra-index-url https://pypi.ngc.nvidia.com onnxruntime-gpu==<jetpack-matched-version>`
- Ensure `libopencv-dev` / `libopenblas` are available (JetPack usually provides them).

## Run

```bash
python main.py
```

## Configuration
Set via environment variables or edit `config.py`:
- `CAMERA_INDEX` (default 0)
- `FRAME_WIDTH` / `FRAME_HEIGHT` (default 1280x720)
- `RECOGNITION_THRESHOLD` (cosine similarity, default 0.35)
- `FACES_REFRESH_SECONDS` (DB sync cadence, default 2s)
- `APP_LOG_LEVEL` (INFO/DEBUG/WARN)
- `JPEG_QUALITY` (default 85)

## Project Layout

```
├── main.py              # NiceGUI app, camera loop, upload/registration
├── face_utils.py        # Provider selection, detection, recognition
├── db.py                # SQLite helpers for embeddings
├── config.py            # Runtime tunables
├── requirements.txt     # Minimal deps; swap onnxruntime for GPU build on Jetson
├── models/              # InsightFace models (auto)
├── storage/             # Saved snapshots (auto)
└── embeddings.db        # SQLite store (auto)
```

## Notes
- InsightFace downloads the chosen model into `models/` if not present.
- Recognition uses cosine similarity; keep faces reasonably frontal or with light yaw for best accuracy.
- The UI displays model info and current provider; logs show which provider was selected at startup.
