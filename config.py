import os

# Model configuration
DEFAULT_MODEL = "buffalo_l"
MODELS_DIR = "models"
AVAILABLE_MODELS = [
    "buffalo_l",
    "buffalo_m",
    "buffalo_s",
    "buffalo_sc",
]

# Execution configuration
# Ordered preference for inference providers; the first available will be used.
PROVIDER_PRIORITY = [
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

# Detection and recognition tuning
RECOGNITION_THRESHOLD = float(os.getenv("RECOGNITION_THRESHOLD", 0.35))
FACES_REFRESH_SECONDS = float(os.getenv("FACES_REFRESH_SECONDS", 2.0))
ENABLE_FLIP_AUGMENT = os.getenv("ENABLE_FLIP_AUGMENT", "1") == "1"
UNKNOWN_SIMILARITY = float(os.getenv("UNKNOWN_SIMILARITY", 0.60))

# Capture settings
CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))
IP_CAMERA_URL = os.getenv("IP_CAMERA_URL", "").strip() or None
FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 1280))
FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 720))
JPEG_QUALITY = int(os.getenv("JPEG_QUALITY", 85))

# Logging
LOG_LEVEL = os.getenv("APP_LOG_LEVEL", "INFO")

# Storage settings
STORAGE_DIR = "storage"
UNKNOWN_DIR = os.path.join(STORAGE_DIR, "unknown")
DATABASE_PATH = "embeddings.db"
