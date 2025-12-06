import os
import sys
from pathlib import Path # ADDED THIS LINE
# Fix for Linux Headless servers crashing with Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"

# Workaround for GCC 11 + NVCC compatibility issue
# Try to force C++14 standard (might break if gsplat needs 17)
# Or try to silence the specific error?
os.environ["TORCH_NVCC_FLAGS"] = "-std=c++14"
os.environ["CFLAGS"] = "-std=c++14"
os.environ["CXXFLAGS"] = "-std=c++14"

# Ensure .venv/bin is in PATH so torch can find ninja
venv_bin = Path(__file__).parent / ".venv" / "bin"
if venv_bin.exists():
    os.environ["PATH"] = str(venv_bin) + os.pathsep + os.environ.get("PATH", "")

import logging
import asyncio
import json
import socket
import time
from pathlib import Path
from typing import List
from fastapi import FastAPI, UploadFile, File, Form, WebSocket, BackgroundTasks
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse # Updated import
import viser

# Import Rich for beautiful console output
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from backend.engine import TrainerEngine

console = Console()

# Setup
app = FastAPI(title="OpenSplat Server")
logger = logging.getLogger("OpenSplat")

# Determine LAN IP
try:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.connect(("8.8.8.8", 80))
    LAN_IP = s.getsockname()[0]
    s.close()
except Exception:
    LAN_IP = "127.0.0.1"

# Initialize Viser (Silence it slightly by handling the print later)
import io
import contextlib
with contextlib.redirect_stdout(io.StringIO()):
    viser_server = viser.ViserServer(host="0.0.0.0", port=8080)
    viser_server.gui.configure_theme(dark_mode=True) # Fix: Dark Mode

# Initialize Engine
engine = TrainerEngine(viser_server)

app.mount("/static", StaticFiles(directory="templates"), name="static")
app.mount("/images", StaticFiles(directory="workspace/data/images"), name="images")
app.mount("/data", StaticFiles(directory="workspace/data"), name="data")

@app.get("/")
async def get_index():
    index_path = Path("templates/index.html")
    if index_path.exists():
        return HTMLResponse(content=index_path.read_text(), status_code=200)
    return {"error": "Frontend not found"}

@app.get("/download")
async def download_ply():
    ply_path = Path("workspace/data/checkpoints/final.ply")
    if ply_path.exists():
        return FileResponse(ply_path, media_type="application/octet-stream", filename="vibe_splat.ply")
    return {"error": "Model not trained yet."}

@app.post("/upload")
async def upload_video(
    background_tasks: BackgroundTasks, 
    file: UploadFile = File(None),
    files: List[UploadFile] = File(None),
    fps: int = Form(2),
    max_steps: int = Form(3000),
    quality: str = Form("medium"),
    blur_filter: bool = Form(False),
    smart_selection: bool = Form(False),
    colmap_matcher: str = Form("sequential")
):
    workspace = Path("workspace")
    workspace.mkdir(exist_ok=True)
    
    input_path = None
    
    # Handle multiple image files
    if files and len(files) > 0 and files[0].filename:
        images_dir = workspace / "data" / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        
        # Clear previous images
        for old in images_dir.glob("*"):
            old.unlink()
        
        for i, f in enumerate(files):
            ext = Path(f.filename).suffix
            dest = images_dir / f"{i:04d}{ext}"
            with open(dest, "wb") as buffer:
                buffer.write(await f.read())
        
        console.print(f"[bold green]Images Uploaded:[/bold green] {len(files)} files")
        input_path = images_dir  # Signal that extraction should be skipped
    elif file and file.filename:
        safe_filename = Path(file.filename).name
        input_path = workspace / safe_filename
        
        with open(input_path, "wb") as buffer:
            buffer.write(await file.read())
        
        console.print(f"[bold green]File Uploaded:[/bold green] {safe_filename} (FPS: {fps})")
    else:
        return {"status": "error", "message": "No file uploaded"}

    def pipeline_task():
        try:
            success = engine.process_video(
                input_path, 
                workspace / "data", 
                fps=fps,
                smart_selection=smart_selection,
                blur_filter=blur_filter,
                matcher_type=colmap_matcher
            )
            if success:
                engine.start_training(workspace / "data", max_steps=max_steps)
            else:
                if engine.status == "cancelled":
                    engine.log("Pipeline cancelled successfully.")
                else:
                    engine.log("Pipeline failed during processing.")
        except Exception as e:
            engine.log(f"Critical Pipeline Error: {e}")
            engine.status = "error"
            
    background_tasks.add_task(pipeline_task)
    
    return {"status": "Processing started"}

@app.post("/train")
async def train_model(
    background_tasks: BackgroundTasks,
    max_steps: int = Form(3000),
    resume: bool = Form(False)
):
    workspace = Path("workspace")
    def train_task():
        engine.start_training(workspace / "data", max_steps=max_steps, resume=resume)
    
    background_tasks.add_task(train_task)
    return {"status": "Training started/resumed"}

@app.post("/control")
async def control_engine(data: dict):
    action = data.get("action")
    if action:
        console.print(f"[bold yellow]User Action:[/bold yellow] {action}")
        engine.control(action)
        return {"status": "ok", "action": action}
    return {"status": "error", "message": "No action provided"}

@app.post("/set_view")
async def set_animation_view():
    # Capture current client camera from Viser
    # Viser doesn't store client state on server unless we ask or track it.
    # Using the first connected client as the "director"
    clients = list(viser_server.get_clients().values())
    if not clients:
        return {"status": "error", "message": "No clients connected"}
    
    client = clients[0]
    engine.animation_pose = (client.camera.position, client.camera.wxyz)
    engine.log("Animation view saved!")
    return {"status": "ok", "message": "View saved"}

@app.post("/render")
async def render_animation(background_tasks: BackgroundTasks):
    if not hasattr(engine, "animation_pose") or engine.animation_pose is None:
        return {"status": "error", "message": "No view set. Use /set_view first."}
    
    def render_task():
        engine.log("Starting Render...")
        # TODO: Implement render loop in engine
        pass
    
    background_tasks.add_task(render_task)
    return {"status": "Render started"}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Drain log queue
            logs = []
            while not engine.log_queue.empty():
                msg = engine.log_queue.get()
                logs.append(msg)
                # Print to server console too for verbosity
                # Don't print every COLMAP line to console if it's too spammy, but user asked for verbosity.
                # We'll style it a bit.
                if "[COLMAP]" in msg:
                    console.print(f"[dim]{msg}[/dim]")
                else:
                    console.print(f"[cyan]{msg}[/cyan]")
            
            elapsed = 0
            if engine.start_time:
                elapsed = int(time.time() - engine.start_time)

            stats = {
                "step": engine.current_step,
                "max_steps": engine.max_steps,
                "status": engine.status,
                "checklist": engine.checklist,
                "image_count": f"{engine.used_images_count} / {engine.extracted_images_count}" if engine.extracted_images_count > 0 else "-",
                "extracted_count": engine.extracted_images_count,
                "used_count": engine.used_images_count,
                "colmap_progress": getattr(engine, 'colmap_progress', {"stage": "", "current": 0, "total": 0}),
                "registered_cameras": len(engine.train_cameras) if engine.train_cameras else 0,
                "pipeline_message": engine.pipeline_message,
                "elapsed_time": elapsed,
                "is_paused": engine.paused,
                "logs": logs
            }
            await websocket.send_text(json.dumps(stats))
            await asyncio.sleep(0.5) 
    except Exception:
        pass

def print_banner():
    banner_text = Text()
    banner_text.append("\n  ✨ OpenSplat Pro ✨\n", style="bold blue")
    banner_text.append("  Gaussian Splatting Appliance\n\n", style="italic white")
    banner_text.append(f"  🖥️  UI Dashboard:   ", style="bold green")
    banner_text.append(f"http://{LAN_IP}:8000\n", style="underline blue")
    banner_text.append(f"  🔮 Viser 3D View:  ", style="bold green")
    banner_text.append(f"http://{LAN_IP}:8080\n", style="underline blue")
    banner_text.append("\n  Ready for jobs...", style="dim white")
    
    panel = Panel(banner_text, border_style="blue", expand=False)
    console.print(panel)

if __name__ == "__main__":
    import uvicorn
    print_banner()
    # Disable standard access log to keep console clean for our custom logs
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")