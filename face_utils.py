from insightface.app import FaceAnalysis
import numpy as np
import os
import onnxruntime
import warnings
import multiprocessing
import threading
from config import DEFAULT_MODEL, MODELS_DIR

# Global variables for singleton pattern
_face_app = None
_initialization_lock = threading.Lock()
_initialized_processes = set()

def get_providers():
    """Get available ONNX Runtime providers, prioritizing CPU to avoid CUDA warnings"""
    available_providers = onnxruntime.get_available_providers()
    
    # Prioritize CPU provider to avoid CUDA warnings if CUDA is not available
    preferred_providers = []
    if 'CPUExecutionProvider' in available_providers:
        preferred_providers.append('CPUExecutionProvider')
    if 'CUDAExecutionProvider' in available_providers:
        preferred_providers.append('CUDAExecutionProvider')
    
    return preferred_providers if preferred_providers else available_providers

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
        print(f"✅ Model '{model_name}' found in {model_path}")
    else:
        print(f"📥 Model '{model_name}' not found. Downloading...")
        print(f"✅ Model '{model_name}' will be downloaded to {model_path}")
    
    # Suppress ONNX Runtime warnings about providers
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*CUDA.*")
        warnings.filterwarnings("ignore", message=".*TensorRT.*")
        
        # Use existing model - set root to current directory so InsightFace uses our models folder
        face_app = FaceAnalysis(name=model_name, root='.', providers=get_providers())
    
    return face_app

def get_face_app():
    """Lazy initialization of face analysis app with controlled initialization"""
    global _face_app
    
    if _face_app is None:
        current_process = multiprocessing.current_process()
        process_name = current_process.name
        
        # Check if model initialization is disabled via environment variable
        skip_init = os.environ.get('SKIP_MODEL_INIT', '').lower() == 'true'
        
        if skip_init:
            print(f"⚠️ Model initialization skipped due to SKIP_MODEL_INIT env var (Process: {process_name})")
            return None
        
        # Initialize models
        print(f"🔧 Initializing face recognition models... (Process: {process_name})")
        _face_app = check_and_setup_models()
        _face_app.prepare(ctx_id=-1)  # Use CPU context to avoid CUDA issues
        print(f"✅ Face recognition models ready for process: {process_name}")
    
    return _face_app

def detect_faces(img):
    face_app = get_face_app()
    if face_app is None:
        print("⚠️ Face recognition not available in this process")
        return []
    return face_app.get(img)

def recognize_face(embedding, known_db, threshold=16.0):
    """Recognize a face by comparing embeddings with known faces."""
    if not known_db:
        return "Unknown"
    
    min_distance = float('inf')
    best_match = "Unknown"
    best_name = "Unknown"
    
    for name, db_embed in known_db:
        dist = np.linalg.norm(embedding - db_embed)
        if dist < min_distance:
            min_distance = dist
            best_name = name
    
    # Only accept the match if it's below the threshold
    if min_distance < threshold:
        best_match = best_name
    
    # Debug output
    print(f"Recognition: best match '{best_match}' with distance {min_distance:.3f} (threshold: {threshold})")
    return best_match

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
    print("🚀 Preloading face recognition models...")
    get_face_app()
    print("🎯 Models preloaded successfully")
