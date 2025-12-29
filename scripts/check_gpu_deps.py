"""Simple GPU dependency checker for Windows/Linux for ONNX Runtime providers.

Run this on the host where you run the app. It searches PATH and common CUDA/TensorRT locations
for the DLL/shared objects that ONNX Runtime needs (cudart, cuDNN, TensorRT).

Usage:
    python scripts/check_gpu_deps.py

It prints findings and next-step suggestions.
"""
import os
import sys
import glob
from pathlib import Path

COMMON_CUDA_DIRS_WINDOWS = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
]
COMMON_TENSORRT_DIRS_WINDOWS = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\TensorRT\bin",
]

COMMON_CUDA_DIRS_LINUX = [
    "/usr/local/cuda/bin",
]
COMMON_TENSORRT_DIRS_LINUX = [
    "/usr/lib/x86_64-linux-gnu/", 
    "/usr/lib/nvidia-*/",
]

patterns = {
    'cudart': ['cudart64_*.dll', 'libcudart*.so*'],
    'cudnn': ['cudnn64_*.dll', 'libcudnn*.so*'],
    'nvinfer': ['nvinfer_*.dll', 'libnvinfer*.so*'],
}


def search_patterns(paths, pats):
    for p in paths:
        if not p:
            continue
        for pat in pats:
            try:
                for match in glob.glob(os.path.join(p, pat)):
                    if Path(match).exists():
                        return match
            except Exception:
                pass
    return None


def find_on_path(pat):
    paths = os.environ.get('PATH', '').split(os.pathsep)
    return search_patterns(paths, [pat])


def check_windows():
    print('Checking Windows-style locations and PATH...')
    paths = os.environ.get('PATH', '').split(os.pathsep)
    for key, pats in patterns.items():
        found = search_patterns(paths + COMMON_CUDA_DIRS_WINDOWS + COMMON_TENSORRT_DIRS_WINDOWS, pats)
        print(f"{key}:", found or 'NOT FOUND')


def check_linux():
    print('Checking Linux-style locations and PATH...')
    paths = os.environ.get('PATH', '').split(os.pathsep)
    for key, pats in patterns.items():
        found = search_patterns(paths + COMMON_CUDA_DIRS_LINUX + COMMON_TENSORRT_DIRS_LINUX, pats)
        print(f"{key}:", found or 'NOT FOUND')


def main():
    print('Python:', sys.executable)
    if os.name == 'nt':
        check_windows()
        print('\nSuggestions:')
        print('- Ensure CUDA bin (e.g. C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\bin) is on PATH')
        print('- Install cuDNN and copy cudnn64_*.dll into the CUDA bin folder or add cuDNN bin to PATH')
        print('- If using TensorRT, add its bin folder (containing nvinfer_*.dll) to PATH')
    else:
        check_linux()
        print('\nSuggestions:')
        print('- Ensure /usr/local/cuda/lib64 and /usr/local/cuda/bin are on LD_LIBRARY_PATH and PATH')
        print('- If using TensorRT, ensure libnvinfer*.so is on LD_LIBRARY_PATH')


if __name__ == '__main__':
    main()
