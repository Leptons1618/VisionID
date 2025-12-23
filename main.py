from nicegui import ui, app, core, run, Client
import base64
import logging
import os
import signal
import time
from datetime import datetime

import cv2
import numpy as np
from PIL import Image
from fastapi import Response

from config import (
    CAMERA_INDEX,
    FACES_REFRESH_SECONDS,
    FRAME_HEIGHT,
    FRAME_WIDTH,
    JPEG_QUALITY,
    LOG_LEVEL,
    RECOGNITION_THRESHOLD,
    STORAGE_DIR,
    UNKNOWN_DIR,
    UNKNOWN_SIMILARITY,
)
from db import (
    delete_unknown,
    get_all_faces,
    get_all_unknowns,
    init_db,
    insert_face,
    insert_unknown,
    promote_unknown,
    touch_unknown,
)
from face_utils import cosine_similarity, detect_faces, get_model_info, preload_models, recognize_face

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# Initialize database and preload models
init_db()
preload_models()
known_faces = get_all_faces()
logger.info("Loaded %s known faces from DB", len(known_faces))
unknown_faces = get_all_unknowns()
logger.info("Loaded %s unknown faces from DB", len(unknown_faces))
current_frame = None
video_capture = None
last_faces_update = 0.0  # Track when faces were last updated
last_unknown_insert = 0.0

# Utility to merge highly similar unknown embeddings so UI shows one card per person
def merge_unknown_clusters(rows):
    clusters = []
    for unk_id, embedding, path, first_seen, last_seen, seen_count in rows:
        matched = None
        for cluster in clusters:
            if cosine_similarity(embedding, cluster['embedding']) >= UNKNOWN_SIMILARITY:
                matched = cluster
                break
        if matched:
            matched['seen_count'] += seen_count
            # Keep the most recent snapshot details
            if last_seen > matched['last_seen']:
                matched['id'] = unk_id
                matched['path'] = path
                matched['last_seen'] = last_seen
            matched['members'].append(unk_id)
        else:
            clusters.append({
                'id': unk_id,
                'embedding': embedding,
                'path': path,
                'first_seen': first_seen,
                'last_seen': last_seen,
                'seen_count': seen_count,
                'members': [unk_id],
            })
    return clusters

# Placeholder image for when webcam is not available
black_1px = 'iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAAAXNSR0IArs4c6QAAAA1JREFUGFdjYGBg+A8AAQQBAHAgZQsAAAAASUVORK5CYII='
placeholder = Response(content=base64.b64decode(black_1px.encode('ascii')), media_type='image/png')


def init_video_capture():
    """Initialize the camera with tuned buffer and resolution for Jetson/desktop."""
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_ANY)
    if not cap or not cap.isOpened():
        logger.error("Camera index %s not available", CAMERA_INDEX)
        return None

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
    logger.info("Camera ready index=%s resolution=%sx%s", CAMERA_INDEX, FRAME_WIDTH, FRAME_HEIGHT)
    return cap

def save_face_image(name, face_img):
    path = f"{STORAGE_DIR}/{name}_{datetime.now().strftime('%H%M%S')}.jpg"
    os.makedirs(STORAGE_DIR, exist_ok=True)
    cv2.imwrite(path, face_img)
    logger.info("Saved face image for '%s' to %s", name, path)
    return path


def save_unknown_image(face_img):
    os.makedirs(UNKNOWN_DIR, exist_ok=True)
    path = f"{UNKNOWN_DIR}/unknown_{datetime.now().strftime('%H%M%S')}.jpg"
    cv2.imwrite(path, face_img)
    logger.info("Saved unknown face snapshot to %s", path)
    return path

# Add debug logging for video frame capture
def convert_frame(frame: np.ndarray) -> bytes:
    """Converts a frame from OpenCV to a JPEG image with face detection."""
    global known_faces, unknown_faces, current_frame, last_faces_update, last_unknown_insert, refresh_unknown_panel
    
    try:
        # Refresh caches on a cadence
        current_time = time.monotonic()
        if current_time - last_faces_update > FACES_REFRESH_SECONDS:
            known_faces = get_all_faces()
            refresh_unknowns()
            last_faces_update = current_time
            logger.debug("Refreshed caches: known=%s unknown=%s", len(known_faces), len(unknown_faces))
        
        # Perform face detection and recognition
        faces = detect_faces(frame)
        for face in faces:
            bbox = face.bbox.astype(int)
            name = recognize_face(face.embedding, known_faces)
            color = (0, 255, 0)
            label = name

            if name == "Unknown":
                # Try to match against stored unknowns for reappearance highlighting
                best_sim = -1.0
                best_unknown_id = None
                for unk_id, unk_emb, _, _, _, _ in unknown_faces:
                    sim = cosine_similarity(face.embedding, unk_emb)
                    if sim > best_sim:
                        best_sim = sim
                        best_unknown_id = unk_id

                if best_unknown_id is not None and best_sim >= UNKNOWN_SIMILARITY:
                    label = f"Unknown #{best_unknown_id}"
                    color = (255, 165, 0)  # orange highlight for returning unknown
                    touch_unknown(best_unknown_id)
                else:
                    # Throttle inserts to avoid spamming DB on every frame
                    if current_time - last_unknown_insert > 2.0:
                        face_img = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        path = save_unknown_image(face_img)
                        new_id = insert_unknown(face.embedding.astype(np.float32), path)
                        refresh_unknowns()
                        if refresh_unknown_panel:
                            refresh_unknown_panel()
                        label = f"Unknown #{new_id}"
                        color = (0, 0, 255)  # red for brand new unknown
                        last_unknown_insert = current_time
                    else:
                        label = "Unknown"
                        color = (0, 0, 255)

            cv2.rectangle(frame, tuple(bbox[:2]), tuple(bbox[2:]), color, 2)
            cv2.putText(frame, label, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Store current frame globally for face registration
        current_frame = frame.copy()
        
        # Convert to JPEG
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
        _, imencode_image = cv2.imencode('.jpg', frame, encode_params)
        return imencode_image.tobytes()
    
    except Exception as e:
        logger.exception("Error in convert_frame: %s", e)
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
        bbox = face.bbox.astype(int)
        face_img = img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
        path = save_face_image(name.strip(), face_img)
        insert_face(name.strip(), face.embedding.astype(np.float32), path)
        logger.info("Registered new face '%s'", name.strip())
        
        force_refresh_faces()
        
        ui.notify(f"{name.strip()} registered successfully!")
        if refresh_admin_panel:
            refresh_admin_panel()
    else:
        logger.warning("Face registration failed: no face in frame")
        ui.notify("No face detected in current frame.")

def upload_image(file):
    """Handle uploaded image for face detection."""
    try:
        content = file.content.read()
        arr = np.frombuffer(content, np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        
        if img is None:
            logger.warning("Upload rejected: invalid image payload")
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
        logger.exception("Error processing uploaded image: %s", e)
        ui.notify(f"Error processing image: {str(e)}", type='negative')

# Global variables to store refresh functions
refresh_admin_panel = None
refresh_unknown_panel = None

def force_refresh_faces():
    """Force refresh of known faces from database."""
    global known_faces, last_faces_update
    known_faces = get_all_faces()
    last_faces_update = time.monotonic()
    logger.info("Forced refresh: loaded %s known faces", len(known_faces))


def refresh_unknowns():
    """Refresh unknown faces cache."""
    global unknown_faces
    unknown_faces = get_all_unknowns()
    logger.debug("Refreshed unknown faces: %s entries", len(unknown_faces))


def handle_promote_unknown(unk_id, name, refresh_callbacks):
    if not name or not name.strip():
        ui.notify("Enter a name to label this person", type='warning')
        return
    success = promote_unknown(unk_id, name.strip())
    if not success:
        ui.notify("Unknown entry no longer exists", type='warning')
        return
    force_refresh_faces()
    refresh_unknowns()
    for cb in refresh_callbacks:
        if cb:
            cb()
    ui.notify(f"Labeled as {name.strip()} and moved to known", type='positive')


def handle_delete_unknown(unk_id, refresh_callbacks):
    delete_unknown(unk_id)
    refresh_unknowns()
    for cb in refresh_callbacks:
        if cb:
            cb()
    ui.notify("Removed unknown entry", type='info')

def setup_app():
    """Setup the application with video capture and UI."""
    global video_capture, refresh_admin_panel
    
    ui.add_head_html('''
    <style>
        :root {
            --bg: #0b1221;
            --card: #0f172a;
            --accent: #0ea5e9;
            --accent-2: #22d3ee;
            --text: #e5e7eb;
        }
        body {
            background: radial-gradient(circle at 20% 20%, rgba(34,211,238,0.08), transparent 25%),
                        radial-gradient(circle at 80% 0%, rgba(14,165,233,0.08), transparent 35%),
                        var(--bg);
            color: var(--text);
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
        }
        .app-shell {
            width: 100%;
            max-width: 1400px;
            margin: 0 auto;
            padding: 24px 18px 36px;
            gap: 16px;
        }
        .panel {
            background: var(--card);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 14px;
            box-shadow: 0 20px 45px rgba(0,0,0,0.35);
        }
        .panel h2 {
            margin: 0;
            font-size: 1.05rem;
            font-weight: 700;
            letter-spacing: 0.01em;
        }
        .status-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            display: inline-block;
            margin-right: 8px;
        }
        .status-dot.on { background: #22c55e; box-shadow: 0 0 10px #22c55e; }
        .status-dot.off { background: #ef4444; }
        .video-frame {
            width: 100%;
            max-height: 75vh;
            min-height: 420px;
            border-radius: 12px;
            border: 1px solid rgba(255,255,255,0.08);
            background: #0b1221;
        }
        .grid-2 { display: grid; grid-template-columns: 2fr 1fr; gap: 16px; width: 100%; }
        .grid-monitor { display: grid; grid-template-columns: 3fr 1.1fr; gap: 18px; width: 100%; align-items: start; }
        .grid-3 { display: grid; grid-template-columns: 1.2fr 1fr 1fr; gap: 16px; width: 100%; }
        .card-scroll { max-height: 32rem; overflow-y: auto; padding-right: 8px; }
        .unknown-scroll { max-height: 36rem; overflow-y: auto; padding-right: 6px; }
        .unknown-grid { display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; width: 100%; }
        .pill { padding: 6px 10px; border-radius: 20px; font-size: 12px; background: rgba(14,165,233,0.12); color: var(--text); }
        @media (max-width: 960px) {
            .grid-2, .grid-3, .grid-monitor { grid-template-columns: 1fr; }
        }
    </style>
    ''')
    
    video_capture = init_video_capture()
    
    with ui.column().classes('app-shell'):
        with ui.row().classes('items-center justify-between mb-2'):
            ui.label("Vision Access Console").classes('text-3xl font-bold tracking-tight text-gray-100')
            with ui.row().classes('gap-2'):
                ui.button("App Settings", on_click=lambda: settings_dialog.open()).classes('bg-sky-600 text-white hover:bg-sky-500')
                ui.button("About", on_click=lambda: about_dialog.open()).classes('bg-gray-600 text-white hover:bg-gray-500')
        ui.label("Realtime detection, recognition, and enrollment. Tuned for Jetson.").classes('text-sm text-gray-200')

        live_tab = 'Live Monitor'
        manage_tab = 'Manage & Data'
        with ui.tabs().classes('mt-4 text-gray-100') as tabs:
            ui.tab(live_tab)
            ui.tab(manage_tab)

        with ui.tab_panels(tabs, value=live_tab).classes('w-full mt-3 text-gray-100'):
            with ui.tab_panel(live_tab):
                with ui.element('div').classes('grid-monitor'):
                    with ui.card().classes('panel p-4 space-y-3'):
                        with ui.row().classes('items-center justify-between'):
                            ui.html('<h2>Live Video</h2>').classes('text-gray-100')
                            camera_status = ui.label().classes('pill')

                        video_image = ui.interactive_image().classes('video-frame')
                        ui.timer(interval=0.1, callback=lambda: video_image.set_source(f'/video/frame?{time.time()}'))

                        def update_camera_status():
                            online = video_capture and video_capture.isOpened()
                            camera_status.set_text("Camera online" if online else "Camera offline")
                            camera_status.classes(replace='pill text-green-200' if online else 'pill text-red-300')

                        ui.timer(1.0, update_camera_status)

                    with ui.card().classes('panel p-4 space-y-3'):
                        ui.html('<h2>Quick Enroll</h2>').classes('text-gray-100')
                        with ui.column().classes('space-y-2'):
                            name_input = ui.input("Full name", placeholder="Enter name as it should appear").classes('w-full text-gray-900')
                            ui.button(
                                "Save to database",
                                on_click=lambda: [register_new_user(name_input.value), name_input.set_value("")],
                            ).classes('bg-sky-500 text-white w-full hover:bg-sky-600')
                            ui.label("Use frontal or slight profile angles; the model accepts light yaw for stability.").classes('text-xs text-gray-300')

            with ui.tab_panel(manage_tab):
                with ui.element('div').classes('grid-2'):
                    # Unknowns manager
                    with ui.card().classes('panel p-4 space-y-3'):
                        ui.html('<h2>Unknowns</h2>').classes('text-gray-100')

                        @ui.refreshable
                        def unknown_panel_content():
                            global unknown_faces
                            unknown_faces = get_all_unknowns()
                            clusters = merge_unknown_clusters(unknown_faces)
                            with ui.element('div').classes('unknown-scroll w-full'):
                                if clusters:
                                    with ui.element('div').classes('unknown-grid'):
                                        for cluster in clusters:
                                            unk_id = cluster['id']
                                            path = cluster['path']
                                            last_seen = cluster['last_seen']
                                            seen_count = cluster['seen_count']
                                            members = cluster['members']
                                            with ui.card().classes('p-3 bg-slate-800 border border-slate-700 space-y-2 h-full'):
                                                member_hint = f" (group of {len(members)})" if len(members) > 1 else ""
                                                ui.label(f"Unknown #{unk_id}{member_hint}").classes('text-sm font-semibold text-gray-100')
                                                ui.label(f"Seen {seen_count}× • Last: {last_seen}").classes('text-xs text-gray-300')
                                                if len(members) > 1:
                                                    ui.label(f"Merged IDs: {', '.join(str(m) for m in members)}").classes('text-[11px] text-gray-400')
                                                with ui.row().classes('items-start gap-3'):
                                                    ui.image(path).classes('w-24 h-24 object-cover rounded border border-slate-700')
                                                    with ui.column().classes('space-y-2 flex-1'):
                                                        ui.label(f"Snapshot: {path}").classes('text-xs text-gray-400 break-all')
                                                        ui.button(
                                                            icon='zoom_in',
                                                            on_click=lambda p=path: show_snapshot(p),
                                                        ).classes('bg-slate-600 text-white text-xs px-2 py-1 hover:bg-slate-500 w-max')
                                                name_input = ui.input("Label as", placeholder="Enter name").classes('w-full text-gray-900')
                                                with ui.row().classes('gap-2'):
                                                    ui.button(
                                                        "Promote",
                                                        on_click=lambda u=unk_id, inp=name_input: handle_promote_unknown(u, inp.value, [admin_panel_content.refresh, unknown_panel_content.refresh]),
                                                    ).classes('bg-emerald-600 text-white text-xs px-3 py-1 hover:bg-emerald-500')
                                                    ui.button(
                                                        "Delete",
                                                        on_click=lambda u=unk_id: handle_delete_unknown(u, [admin_panel_content.refresh, unknown_panel_content.refresh]),
                                                    ).classes('bg-gray-600 text-white text-xs px-3 py-1 hover:bg-gray-500')
                                else:
                                    ui.label("No unknowns recorded yet").classes('text-gray-400 italic text-center py-4')

                        unknown_panel_content()
                        refresh_unknown_panel = unknown_panel_content.refresh

                    # Right column stack
                    with ui.column().classes('space-y-3'):
                        with ui.card().classes('panel p-4 space-y-3'):
                            ui.html('<h2>Upload Test Image</h2>').classes('text-gray-100')
                            ui.upload(on_upload=upload_image, max_file_size=10_000_000).classes('w-full')
                            ui.label("PNG or JPEG up to 10 MB. Faces will be detected and labeled.").classes('text-xs text-gray-300')

                        with ui.card().classes('panel p-4 space-y-3'):
                            ui.html('<h2>Directory</h2>').classes('text-gray-100')

                            @ui.refreshable
                            def admin_panel_content():
                                global known_faces
                                known_faces = get_all_faces()
                                with ui.column().classes('space-y-1 max-h-72 overflow-y-auto'):
                                    if known_faces:
                                        for i, (name, _) in enumerate(known_faces, 1):
                                            with ui.row().classes('items-center justify-between text-sm'):
                                                ui.label(f"{i}. {name}").classes('text-gray-100')
                                                ui.label("Registered").classes('text-xs text-gray-400')
                                    else:
                                        ui.label("No registered users yet").classes('text-gray-400 italic text-center py-4')

                            admin_panel_content()
                            refresh_admin_panel = admin_panel_content.refresh
                            ui.button("Refresh", on_click=lambda: [refresh_admin_panel(), refresh_unknown_panel() if refresh_unknown_panel else None]).classes('bg-gray-600 text-white w-full hover:bg-gray-500')

                        with ui.card().classes('panel p-4 space-y-3'):
                            ui.html('<h2>Model & Runtime</h2>').classes('text-gray-100')
                            info = get_model_info()
                            ui.label(f"Active model: {info['current_model']}").classes('text-sm text-gray-100')
                            ui.label(f"Models path: {info['models_directory']}").classes('text-xs text-gray-300 break-all')
                            available_models = ', '.join(info['available_models']) if info['available_models'] else 'None'
                            ui.label(f"Available: {available_models}").classes('text-sm text-gray-100')
                            ui.label("Providers and device selection happen automatically; CUDA/TensorRT used when available.").classes('text-xs text-gray-300')

        with ui.dialog() as settings_dialog, ui.card().classes('w-full max-w-xl p-4 space-y-3 panel'):
            ui.label("App Settings").classes('text-xl font-semibold text-gray-100')
            ui.label("Runtime values are sourced from environment variables or config.py. Restart required after changes.").classes('text-sm text-gray-300')
            ui.label(f"Camera index: {CAMERA_INDEX}").classes('text-sm text-gray-200')
            ui.label(f"Frame: {FRAME_WIDTH}x{FRAME_HEIGHT}").classes('text-sm text-gray-200')
            ui.label(f"Recognition threshold: {RECOGNITION_THRESHOLD}").classes('text-sm text-gray-200')
            ui.label(f"Unknown similarity (grouping): {UNKNOWN_SIMILARITY}").classes('text-sm text-gray-200')
            ui.label(f"Faces refresh (s): {FACES_REFRESH_SECONDS}").classes('text-sm text-gray-200')
            ui.label(f"JPEG quality: {JPEG_QUALITY}").classes('text-sm text-gray-200')
            ui.label(f"Log level: {LOG_LEVEL}").classes('text-sm text-gray-200')
            ui.button("Close", on_click=settings_dialog.close).classes('bg-gray-600 text-white w-full hover:bg-gray-500')

        with ui.dialog() as about_dialog, ui.card().classes('w-full max-w-lg p-4 space-y-3 panel'):
            ui.label("About").classes('text-xl font-semibold text-gray-100')
            ui.label("Edge-ready face recognition with InsightFace and NiceGUI.").classes('text-sm text-gray-200')
            ui.label("Optimized for CUDA/TensorRT when available; falls back to CPU.").classes('text-sm text-gray-200')
            ui.label("For best results, enroll with neutral lighting and slight yaw tolerance.").classes('text-sm text-gray-200')
            ui.button("Close", on_click=about_dialog.close).classes('bg-gray-600 text-white w-full hover:bg-gray-500')

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
    
    # Graceful signal handling for Ctrl+C
    def handle_sigint(signum, frame) -> None:
        import asyncio
        logger.info("SIGINT received, shutting down gracefully...")
        try:
            if video_capture:
                video_capture.release()
            loop = asyncio.get_event_loop()
            loop.create_task(app.shutdown())
        except Exception:
            pass
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_sigint)

    def show_snapshot(path: str):
        with ui.dialog() as dlg, ui.card().classes('p-3 panel w-full max-w-3xl'):
            ui.label(path).classes('text-xs text-gray-300 break-all')
            ui.image(path).classes('w-full h-auto rounded border border-slate-700')
            ui.button('Close', on_click=dlg.close).classes('bg-gray-600 text-white text-xs px-3 py-1 hover:bg-gray-500 mt-2')
        dlg.open()

# Setup the application when server starts
app.on_startup(setup_app)

ui.run(port=8080, reload=False, title="Vision Sentinel Console")
