# Face Detection & Recognition System

## Features

- **Live face detection and recognition** using webcam
- **Face registration** with name storage
- **Image upload** for face detection
- **Model management** with automatic downloading
- **Local storage** of face embeddings and images

## Model Management

### Models Directory
All face recognition models are stored in the `models/` directory. The system will automatically:
- Create the models directory if it doesn't exist
- Check for existing models before downloading
- Download models only when needed

### Configuration
Model settings can be configured in `config.py`:
- `DEFAULT_MODEL`: The default model to use (buffalo_l)
- `MODELS_DIR`: Directory where models are stored
- `AVAILABLE_MODELS`: List of supported models

### Available Models
- **buffalo_l**: High accuracy, larger size (default)
- **buffalo_m**: Medium accuracy, medium size
- **buffalo_s**: Lower accuracy, smaller size
- **buffalo_sc**: Compact version

## File Structure

```
├── main.py              # Main application with UI
├── face_utils.py        # Face detection and recognition utilities
├── db.py               # Database operations for face embeddings
├── config.py           # Configuration settings
├── requirements.txt    # Python dependencies
├── test_models.py      # Model setup testing script
├── models/             # Face recognition models (auto-created)
├── storage/            # Stored face images (auto-created)
└── embeddings.db       # SQLite database for face embeddings
```

## Usage

### Running the Application
```bash
python main.py
```

### Testing Model Setup
```bash
python test_models.py
```

### First Run
On the first run, the application will:
1. Create the `models/` directory
2. Download the default model (`buffalo_l`) if not present
3. Create the `storage/` directory for face images
4. Initialize the SQLite database

### Model Information
The application displays current model information in the admin panel:
- Current model in use
- Models directory path
- List of available models

## Dependencies

Key dependencies include:
- `nicegui`: Web-based UI framework
- `insightface`: Face recognition library
- `opencv-python`: Computer vision operations
- `numpy`: Numerical operations
- `sqlite3`: Database operations (built-in)

## Storage

- **Face images**: Stored in `storage/` directory
- **Face embeddings**: Stored in SQLite database (`embeddings.db`)
- **Models**: Stored in `models/` directory

## Notes

- Models are downloaded from the InsightFace model zoo
- The system checks for existing models before downloading to avoid redundant downloads
- Model files can be large (100MB+), so ensure adequate disk space
- GPU acceleration is supported if available (CUDA)
