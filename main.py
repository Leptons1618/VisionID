from nicegui import ui, app, core, run, Client
import cv2
import numpy as np
import os
from datetime import datetime
import base64
from PIL import Image
import signal
import time
from fastapi import Response
from db import init_db, insert_face, get_all_faces
from face_utils import detect_faces, recognize_face, get_model_info, preload_models
from config import STORAGE_DIR

# Initialize database and preload models
init_db()
preload_models()  # Preload models in the main process
known_faces = get_all_faces()
current_frame = None
video_capture = None
last_faces_update = time.time()  # Track when faces were last updated

# Placeholder image for when webcam is not available
black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')

def save_face_image(name, face_img):
    path = f"{STORAGE_DIR}/{name}_{datetime.now().strftime('%H%M%S')}.jpg"
    os.makedirs(STORAGE_DIR, exist_ok=True)
    cv2.imwrite(path, face_img)
    return path

# Add debug logging for video frame capture
def convert_frame(frame: np.ndarray) -> bytes:
    """Converts a frame from OpenCV to a JPEG image with face detection."""
    global known_faces, current_frame, last_faces_update
    
    try:
        # Refresh known_faces every 2 seconds or when forced
        current_time = time.time()
        if current_time - last_faces_update > 2.0:
            known_faces = get_all_faces()
            last_faces_update = current_time
        
        # Perform face detection and recognition
        faces = detect_faces(frame)
        for face in faces:
            # Convert bbox to integers to prevent slicing errors
            bbox = face.bbox.astype(int)
            name = recognize_face(face.embedding, known_faces)
            cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), (0, 255, 0), 2)
            cv2.putText(frame, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        
        # Store current frame globally for face registration
        current_frame = frame.copy()
        
        # Convert to JPEG
        _, imencode_image = cv2.imencode('.jpg', frame)
        return imencode_image.tobytes()
    
    except Exception as e:
        print(f"Error in convert_frame: {e}")
        # Return a simple frame if processing fails
        _, imencode_image = cv2.imencode('.jpg', frame)
        return imencode_image.tobytes()

@app.get('/video/frame')
async def grab_video_frame() -> Response:
    """FastAPI route to get the latest video frame."""
    global video_capture
    if not video_capture or not video_capture.isOpened():
        return placeholder
    
    # Read frame in a separate thread to avoid blocking
    _, frame = await run.io_bound(video_capture.read)
    if frame is None:
        return placeholder
    
    # Process frame with face detection in a separate process
    jpeg = await run.cpu_bound(convert_frame, frame)
    return Response(content=jpeg, media_type='image/jpeg')

def register_new_user(name):
    """Register a new user using the current frame."""
    global current_frame, video_capture
    
    if not name or not name.strip():
        ui.notify("Please enter a valid name.")
        return
    
    # If no current frame, try to capture one directly
    if current_frame is None:
        if video_capture and video_capture.isOpened():
            ret, frame = video_capture.read()
            if ret:
                current_frame = frame.copy()
            else:
                ui.notify("No video feed available.")
                return
        else:
            ui.notify("Camera not initialized.")
            return
        
    img = current_frame.copy()
    faces = detect_faces(img)
    if faces:
        face = faces[0]
        # Convert bbox to integers to fix the slicing error
        bbox = face.bbox.astype(int)
        face_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        path = save_face_image(name.strip(), face_img)
        insert_face(name.strip(), face.embedding.astype(np.float32), path)
        
        # Force refresh faces immediately for real-time recognition
        force_refresh_faces()
        
        ui.notify(f"{name.strip()} registered successfully!")
        # Refresh the admin panel
        if refresh_admin_panel:
            refresh_admin_panel()
    else:
        ui.notify("No face detected in current frame.")

def upload_image(file):
    """Handle uploaded image for face detection."""
    try:
        content = file.content.read()
        arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            ui.notify("Invalid image format", type='warning')
            return
        
        faces = detect_faces(img)
        for face in faces:
            # Convert bbox to integers to prevent slicing errors
            bbox = face.bbox.astype(int)
            name = recognize_face(face.embedding, known_faces)
            cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), (255, 0, 0), 2)
            cv2.putText(img, name, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Convert BGR to RGB for display
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_img)
        
        # Display result in a modal or container
        with ui.dialog() as dialog, ui.card().classes('w-auto max-w-4xl'):
            ui.label(f"Detection Results: {len(faces)} face(s) found").classes('text-lg font-semibold mb-4')
            ui.image(pil_image).classes('max-w-full h-auto')
            ui.button('Close', on_click=dialog.close).classes('mt-4')
        
        dialog.open()
        
        if faces:
            ui.notify(f"Detected {len(faces)} face(s)", type='positive')
        else:
            ui.notify("No faces detected in uploaded image", type='info')
            
    except Exception as e:
        ui.notify(f"Error processing image: {str(e)}", type='negative')

# Global variable to store refresh function
refresh_admin_panel = None

def force_refresh_faces():
    """Force refresh of known faces from database."""
    global known_faces, last_faces_update
    known_faces = get_all_faces()
    last_faces_update = time.time()
    print(f"Forced refresh: Loaded {len(known_faces)} known faces")

def setup_app():
    """Setup the application with video capture and UI."""
    global video_capture, refresh_admin_panel
    
    # Add custom CSS for better styling
    ui.add_head_html('''
    <style>
        .video-container {
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 300px;
            background: #f5f5f5;
            border-radius: 8px;
        }
        
        @media (max-width: 768px) {
            .mobile-stack {
                flex-direction: column !important;
            }
        }
        
        .admin-panel {
            max-height: 300px;
            overflow-y: auto;
        }
        
        .model-info {
            font-family: 'Monaco', 'Menlo', 'Ubuntu Mono', monospace;
            font-size: 0.8rem;
        }
    </style>
    ''')
    
    # Initialize video capture
    video_capture = cv2.VideoCapture(0)
    
    # Main container with responsive design
    with ui.column().classes('w-full max-w-6xl mx-auto p-4 space-y-6'):
        
        # Header
        ui.label("🎥 Live Face Detection & Recognition").classes('text-3xl font-bold text-center mb-4')
        
        # Video section with responsive sizing
        with ui.card().classes('w-full'):
            with ui.row().classes('w-full items-center justify-between mb-2'):
                ui.label("📹 Live Video Feed").classes('text-xl font-semibold')
                # Camera status indicator
                camera_status = ui.label().classes('text-sm')
                ui.timer(1.0, lambda: camera_status.set_text(
                    "🟢 Camera Connected" if video_capture and video_capture.isOpened() else "🔴 Camera Disconnected"
                ))
            
            with ui.row().classes('w-full justify-center video-container'):
                # Dynamic video display that adapts to screen size
                video_image = ui.interactive_image().classes('max-w-full h-auto border rounded-lg shadow-lg').style('max-height: 60vh; min-height: 300px; max-width: 800px;')
                ui.timer(interval=0.1, callback=lambda: video_image.set_source(f'/video/frame?{time.time()}'))
        
        # Controls section
        with ui.card().classes('w-full'):
            ui.label("👤 Face Registration").classes('text-xl font-semibold mb-2')
            with ui.row().classes('w-full gap-4 items-end'):
                name_input = ui.input("Enter your name", placeholder="Enter full name...").classes('flex-grow')
                ui.button("Register Face", on_click=lambda: [register_new_user(name_input.value), name_input.set_value("")]).classes('bg-blue-500 hover:bg-blue-600 text-white')

        # Upload section
        with ui.card().classes('w-full'):
            ui.label("📸 Upload Image for Detection").classes('text-xl font-semibold mb-2')
            ui.upload(on_upload=upload_image, max_file_size=10_000_000).classes('w-full')

        # Admin and info section side by side
        with ui.row().classes('w-full gap-4 mobile-stack'):
            # Admin Panel
            with ui.card().classes('flex-1 min-w-0'):
                ui.label("🧑‍💼 Admin Panel").classes('text-xl font-semibold mb-2')
                
                @ui.refreshable
                def admin_panel_content():
                    global known_faces
                    known_faces = get_all_faces()  # Refresh data
                    with ui.column().classes('admin-panel space-y-2'):
                        if known_faces:
                            for i, (name, _) in enumerate(known_faces, 1):
                                ui.label(f"{i}. 🟢 {name}").classes('text-green-600 font-medium text-sm')
                        else:
                            ui.label("No registered users yet").classes('text-gray-500 italic text-center py-4')
                
                admin_panel_content()
                
                # Make refresh function globally accessible
                refresh_admin_panel = admin_panel_content.refresh
                
                # Add refresh button
                ui.button("🔄 Refresh", on_click=lambda: refresh_admin_panel()).classes('mt-2 text-xs bg-gray-500 hover:bg-gray-600 text-white')

            # Model Information
            with ui.card().classes('flex-1 min-w-0'):
                ui.label("🤖 Model Information").classes('text-xl font-semibold mb-2')
                with ui.column().classes('space-y-2 model-info'):
                    model_info = get_model_info()
                    ui.label(f"📊 Current Model: {model_info['current_model']}").classes('text-sm font-semibold')
                    ui.label(f"📁 Directory: {model_info['models_directory']}").classes('text-xs text-gray-600 break-all')
                    available_models = ', '.join(model_info['available_models']) if model_info['available_models'] else 'None'
                    ui.label(f"📦 Available: {available_models}").classes('text-sm')

    # Setup cleanup and signal handling
    async def disconnect() -> None:
        """Disconnect all clients from current running server."""
        for client_id in Client.instances:
            await core.sio.disconnect(client_id)

    async def cleanup() -> None:
        """Cleanup function to release webcam and disconnect clients."""
        await disconnect()
        if video_capture:
            video_capture.release()

    app.on_shutdown(cleanup)
    
    # Simplified signal handling
    def handle_sigint(signum, frame) -> None:
        import asyncio
        import os
        os._exit(0)  # Force exit without UI cleanup issues
    
    signal.signal(signal.SIGINT, handle_sigint)

# Setup the application when server starts
app.on_startup(setup_app)

ui.run(port=8080, reload=False)
