import logging
import multiprocessing
import os
import threading
import warnings

import numpy as np
import onnxruntime
from insightface.app import FaceAnalysis

from config import (
    AVAILABLE_MODELS,
    DEFAULT_MODEL,
    MODELS_DIR,
    PROVIDER_PRIORITY,
    RECOGNITION_THRESHOLD,
    UNKNOWN_SIMILARITY,
)

# Global variables for singleton pattern
_face_app = None
_initialization_lock = threading.Lock()
_initialized_processes = set()

logger = logging.getLogger(__name__)


def _normalize_embeddings(emb: np.ndarray) -> np.ndarray:
    """L2-normalize embeddings to stabilize cosine similarity."""
    norm = np.linalg.norm(emb) + 1e-9
    return emb / norm


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity helper for reuse across known/unknown matching."""
    a_n = _normalize_embeddings(a.astype(np.float32))
    b_n = _normalize_embeddings(b.astype(np.float32))
    return float(np.dot(a_n, b_n))

def get_providers():
    """Return the best-matching provider list based on availability and configured priority."""
    available = onnxruntime.get_available_providers()
    ordered = [p for p in PROVIDER_PRIORITY if p in available]
    logger.info(
        "ONNX providers available=%s selected=%s",
        ",".join(available),
        ",".join(ordered) if ordered else "<none>",
    )
    return ordered if ordered else available


def _get_ctx_id(selected_providers):
    """Return ctx id: 0 for GPU-capable provider, -1 for CPU."""
    gpu_providers = {"CUDAExecutionProvider", "TensorrtExecutionProvider", "DmlExecutionProvider"}
    return 0 if any(p in gpu_providers for p in selected_providers) else -1

def check_and_setup_models(model_name=None):
    """Check if models exist locally, if not download them"""
    if model_name is None:
        model_name = DEFAULT_MODEL
    
    models_dir = MODELS_DIR
    
    # Create models directory if it doesn't exist
    os.makedirs(models_dir, exist_ok=True)
    
    # Check if the specific model exists
    model_path = os.path.join(models_dir, model_name)
    
    if os.path.exists(model_path) and os.listdir(model_path):
        logger.info("Model '%s' found in %s", model_name, model_path)
    else:
        logger.warning("Model '%s' missing. Will download to %s", model_name, model_path)
    
    # Suppress ONNX Runtime warnings about providers
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CUDA.*")
        warnings.filterwarnings("ignore", message=".*TensorRT.*")
        
        # Use existing model - set root to current directory so InsightFace uses our models folder
        providers = get_providers()
        face_app = FaceAnalysis(name=model_name, root='.', providers=providers)
        # Persist chosen providers for logging even if insightface does not expose attribute
        face_app._providers = providers
    
    return face_app

def get_face_app():
    """Lazy initialization of face analysis app with controlled initialization"""
    global _face_app
    
    if _face_app is None:
        with _initialization_lock:
            if _face_app is not None:
                return _face_app

            current_process = multiprocessing.current_process()
            process_name = current_process.name
            
            skip_init = os.environ.get('SKIP_MODEL_INIT', '').lower() == 'true'
            if skip_init:
                logger.warning("Model initialization skipped (process=%s)", process_name)
                return None
            
            logger.info("Initializing face models (process=%s)", process_name)
            _face_app = check_and_setup_models()
            providers = getattr(_face_app, "_providers", get_providers())
            ctx_id = _get_ctx_id(providers)
            # Wider det_size helps profile/side detections on Jetson, still light enough for realtime.
            _face_app.prepare(ctx_id=ctx_id, det_size=(640, 640), det_thresh=0.45)
            logger.info("Face models ready (providers=%s ctx_id=%s)", providers, ctx_id)
    
    return _face_app

def detect_faces(img):
    face_app = get_face_app()
    if face_app is None:
        logger.warning("Face recognition not available in this process")
        return []
    try:
        return face_app.get(img)
    except Exception as exc:
        logger.exception("Face detection failed: %s", exc)
        return []

def recognize_face(embedding, known_db, threshold=RECOGNITION_THRESHOLD):
    """Recognize a face using cosine similarity for better side-profile tolerance."""
    if not known_db:
        return "Unknown"

    query = _normalize_embeddings(embedding.astype(np.float32))
    best_name = "Unknown"
    best_score = -1.0

    for name, db_embed in known_db:
        known = _normalize_embeddings(db_embed.astype(np.float32))
        score = float(np.dot(query, known))
        if score > best_score:
            best_score = score
            best_name = name

    if best_score >= threshold:
        logger.debug("Recognition matched '%s' (similarity=%.3f, thr=%.3f)", best_name, best_score, threshold)
        return best_name

    logger.debug("Recognition unknown (best=%.3f, thr=%.3f)", best_score, threshold)
    return "Unknown"

def get_available_models():
    """List all available models in the models directory"""
    if not os.path.exists(MODELS_DIR):
        return []
    
    models = []
    for item in os.listdir(MODELS_DIR):
        item_path = os.path.join(MODELS_DIR, item)
        if os.path.isdir(item_path):
            models.append(item)
    return models

def get_model_info():
    """Get information about the currently loaded model and available models"""
    available_models = get_available_models()
    current_model = DEFAULT_MODEL
    
    return {
        'current_model': current_model,
        'available_models': available_models,
        'models_directory': os.path.abspath(MODELS_DIR)
    }

def preload_models():
    """Preload models in the main process to avoid multiple initializations"""
    logger.info("Preloading face recognition models...")
    get_face_app()
    logger.info("Models preloaded successfully")
