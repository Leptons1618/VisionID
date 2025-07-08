# Model Configuration
# This file contains settings for face recognition models

# Default model to use
DEFAULT_MODEL = "buffalo_l"

# Models directory
MODELS_DIR = "models"

# Available models for InsightFace
AVAILABLE_MODELS = [
    "buffalo_l",    # High accuracy, larger size
    "buffalo_m",    # Medium accuracy, medium size  
    "buffalo_s",    # Lower accuracy, smaller size
    "buffalo_sc",   # Compact version
]

# Model download URLs (handled automatically by InsightFace)
# Models will be downloaded from InsightFace model zoo when needed

# Storage settings
STORAGE_DIR = "storage"
DATABASE_PATH = "embeddings.db"
