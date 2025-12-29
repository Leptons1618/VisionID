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


### Ubuntu: CUDA (Recommended for NVIDIA GPUs)

1. **Install NVIDIA Drivers**
	- Make sure you have the latest drivers for your GPU:
	  ```bash
	  sudo apt update
	  sudo apt install nvidia-driver-535  # Or latest for your GPU
	  sudo reboot
	  ```

2. **Install CUDA Toolkit (11.8 recommended)**
	- Download from https://developer.nvidia.com/cuda-toolkit-archive
	- Follow NVIDIA's instructions for your Ubuntu version.
	- Add CUDA to your PATH and LD_LIBRARY_PATH (usually done by the installer):
	  ```bash
	  export PATH=/usr/local/cuda/bin:$PATH
	  export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
	  ```

3. **(Optional) Install cuDNN**
	- Download cuDNN for CUDA 11.x from https://developer.nvidia.com/cudnn
	- Extract and copy the files to your CUDA folders as per NVIDIA's instructions.

4. **Install ONNX Runtime GPU**
	- In your Python environment:
	  ```bash
	  pip uninstall -y onnxruntime
	  pip install onnxruntime-gpu
	  ```

5. **Verify GPU is available**
	- Run:
	  ```python
	  import onnxruntime
	  print(onnxruntime.get_available_providers())
	  ```
	- You should see `CUDAExecutionProvider` in the output.

6. **(Optional) TensorRT for Advanced Users**
	- Install TensorRT from https://developer.nvidia.com/tensorrt
	- Add the TensorRT `lib` directory (containing `nvinfer*.so`) to your `LD_LIBRARY_PATH`.
	- Most users do not need TensorRT; CUDA is fast and stable.

**Provider selection:** By default, this app uses CUDA if available, then CPU. You can change this in `config.py` via `PROVIDER_PRIORITY`.

### Windows: CUDA + cuDNN

1. **Install NVIDIA Drivers**
	- Get drivers from: https://www.nvidia.com/Download/index.aspx

2. **Install CUDA Toolkit**
	- Install a CUDA toolkit supported by your `onnxruntime-gpu` wheel (11.8 or 12.x commonly).
	- Ensure `cudart64_*.dll` is on your `PATH` (CUDA installer normally does this).

3. **Install cuDNN**
	- Download cuDNN matching your CUDA version from NVIDIA (developer account required).
	- Copy `cudnn64_*.dll` into your CUDA `bin` folder (for example `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin`) or add the cuDNN `bin` folder to your `PATH`.

4. **Install ONNX Runtime GPU wheel**
	```powershell
	pip uninstall -y onnxruntime
	pip install onnxruntime-gpu
	```

5. **Verify providers**
	```python
	import onnxruntime
	print(onnxruntime.get_available_providers())
	```
	- If `CUDAExecutionProvider` is present, GPU should be usable. If you see errors about missing `cudnn64_*.dll`, ensure cuDNN is installed and its `bin` is on `PATH`.

If you prefer TensorRT on Windows, install TensorRT and add its `bin` directory (containing `nvinfer_*.dll`) to `PATH`.

## Run

```bash
python main.py
```

## Configuration
Set via environment variables or edit `config.py`:
- `IP_CAMERA_URL` (e.g. `rtsp://user:pass@host:554/stream`; if set, overrides camera index)
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
