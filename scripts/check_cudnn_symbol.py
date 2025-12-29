"""Checks for cuDNN DLLs and the presence of the `cudnnCreate` symbol on Windows/Linux.

Run:
    python scripts/check_cudnn_symbol.py

This will search PATH and common CUDA install locations, try to load cuDNN with ctypes,
and report whether the expected symbol is available.
"""
import os
import sys
import glob
from pathlib import Path
import ctypes

SEARCH_DIRS_WINDOWS = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.8\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.0\bin",
]
SEARCH_DIRS_LINUX = ["/usr/local/cuda/lib64", "/usr/lib/x86_64-linux-gnu"]

patterns = ["cudnn64_*.dll", "libcudnn*.so*"]


def find_candidates():
    paths = os.environ.get('PATH', '').split(os.pathsep)
    candidates = []
    if os.name == 'nt':
        paths = paths + SEARCH_DIRS_WINDOWS
    else:
        paths = paths + SEARCH_DIRS_LINUX
    for p in paths:
        if not p:
            continue
        for pat in patterns:
            try:
                for match in glob.glob(os.path.join(p, pat)):
                    if Path(match).exists():
                        candidates.append(match)
            except Exception:
                pass
    return sorted(set(candidates))


def try_load_and_check_symbol(path):
    try:
        if os.name == 'nt':
            dll = ctypes.WinDLL(path)
        else:
            dll = ctypes.CDLL(path)
    except Exception as e:
        return False, f"LOAD FAIL: {e}"
    # Check for cudnnCreate symbol
    try:
        sym = getattr(dll, 'cudnnCreate')
        return True, 'SYMBOL FOUND'
    except AttributeError:
        # Some newer cuDNN names or ABI incompatibilities may exist
        return False, 'SYMBOL MISSING (cudnnCreate not found)'
    except Exception as e:
        return False, f'ERROR checking symbol: {e}'


def main():
    print('Python executable:', sys.executable)
    print('OS:', os.name)
    print('\nSearching for cuDNN candidates...')
    cand = find_candidates()
    if not cand:
        print('No cuDNN candidates found on PATH or common locations.')
        print('\nSuggestion: install cuDNN matching your CUDA and ensure bin/lib paths are on PATH or LD_LIBRARY_PATH.')
        return
    for p in cand:
        print('\nCandidate:', p)
        ok, msg = try_load_and_check_symbol(p)
        print('  ->', msg)

    print('\nIf symbols are missing, likely causes:')
    print('- Wrong cuDNN version for the installed ONNX Runtime CUDA wheel (ABI mismatch).')
    print('- Multiple CUDA/cuDNN installs on PATH causing wrong DLL to be loaded.\n')
    print('Next steps:')
    print('- Run `pip show onnxruntime-gpu` and paste its version here.')
    print('- Run `where cudnn64_*.dll` (PowerShell) or `ls /usr/local/cuda/lib64 | grep cudnn` (Linux) to show exact files.')

if __name__ == "__main__":
    main()
