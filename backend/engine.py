import math
import struct
import time
import logging
import subprocess
import shutil
import queue
import re
import threading
import zipfile
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Dict, Tuple, NamedTuple

import torch
import torch.nn.functional as F
import numpy as np
import cv2
import viser

# gsplat modern API (1.4.0)
from gsplat import rasterization, DefaultStrategy

logger = logging.getLogger("OpenSplat")
logging.basicConfig(level=logging.INFO)

# ... (Helpers & Loaders - Keep same as before) ...
def qvec2rotmat(qvec):
    return np.array([
        [1 - 2 * qvec[2]**2 - 2 * qvec[3]**2,
         2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
         2 * qvec[1] * qvec[3] + 2 * qvec[0] * qvec[2]],
        [2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
         1 - 2 * qvec[1]**2 - 2 * qvec[3]**2,
         2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1]],
        [2 * qvec[1] * qvec[3] - 2 * qvec[0] * qvec[2],
         2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
         1 - 2 * qvec[1]**2 - 2 * qvec[2]**2]])

class ColmapLoader:
    # COLMAP camera model ID -> number of parameters
    # https://github.com/colmap/colmap/blob/main/src/colmap/sensor/models.h
    CAMERA_MODEL_PARAMS = {
        0: 3,   # SIMPLE_PINHOLE: f, cx, cy
        1: 4,   # PINHOLE: fx, fy, cx, cy
        2: 4,   # SIMPLE_RADIAL: f, cx, cy, k
        3: 5,   # RADIAL: f, cx, cy, k1, k2
        4: 8,   # OPENCV: fx, fy, cx, cy, k1, k2, p1, p2
        5: 9,   # OPENCV_FISHEYE: fx, fy, cx, cy, k1, k2, k3, k4
        6: 12,  # FULL_OPENCV: fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6
        7: 5,   # FOV: fx, fy, cx, cy, omega
        8: 5,   # SIMPLE_RADIAL_FISHEYE: f, cx, cy, k
        9: 6,   # RADIAL_FISHEYE: f, cx, cy, k1, k2
        10: 4,  # THIN_PRISM_FISHEYE: fx, fy, cx, cy (+ more)
    }
    
    def __init__(self, model_path: Path, images_dir: Path):
        self.model_path = model_path
        self.images_dir = images_dir
        self.cameras = {}
        self.images = {}
        self.points3D = {}
        
        if not (self.model_path / "cameras.bin").exists():
            raise FileNotFoundError(f"COLMAP binary files not found in {self.model_path}")

        self._read_cameras_binary()
        self._read_images_binary()
        self._read_points3D_binary()

    def _read_cameras_binary(self):
        with open(self.model_path / "cameras.bin", "rb") as f:
            num_cameras = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_cameras):
                camera_id = struct.unpack("<i", f.read(4))[0]
                model_id = struct.unpack("<i", f.read(4))[0]
                width = struct.unpack("<Q", f.read(8))[0]
                height = struct.unpack("<Q", f.read(8))[0]
                
                # Get the correct number of params for this camera model
                params_cnt = self.CAMERA_MODEL_PARAMS.get(model_id, 4)  # Default to 4
                params = struct.unpack(f"<{params_cnt}d", f.read(8 * params_cnt))
                
                K = np.eye(3)
                # Handle different camera models for intrinsics extraction
                if model_id == 0:  # SIMPLE_PINHOLE
                    focal, cx, cy = params[0], params[1], params[2]
                    K[0, 0] = focal; K[1, 1] = focal; K[0, 2] = cx; K[1, 2] = cy
                elif model_id == 1:  # PINHOLE
                    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                    K[0, 0] = fx; K[1, 1] = fy; K[0, 2] = cx; K[1, 2] = cy
                elif model_id in [2, 3, 8, 9]:  # SIMPLE_RADIAL, RADIAL, *_FISHEYE
                    focal, cx, cy = params[0], params[1], params[2]
                    K[0, 0] = focal; K[1, 1] = focal; K[0, 2] = cx; K[1, 2] = cy
                elif model_id in [4, 5, 6, 7, 10]:  # OPENCV variants, FOV, THIN_PRISM
                    fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                    K[0, 0] = fx; K[1, 1] = fy; K[0, 2] = cx; K[1, 2] = cy
                else:
                    # Unknown model, try to extract something reasonable
                    if len(params) >= 4:
                        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
                        K[0, 0] = fx; K[1, 1] = fy; K[0, 2] = cx; K[1, 2] = cy
                    else:
                        focal = params[0] if params else max(width, height)
                        K[0, 0] = focal; K[1, 1] = focal; K[0, 2] = width/2; K[1, 2] = height/2
                        
                self.cameras[camera_id] = {"K": K, "w": width, "h": height, "model_id": model_id}

    def _read_images_binary(self):
        with open(self.model_path / "images.bin", "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
            for _ in range(num_images):
                binary = f.read(64) 
                image_id, qw, qx, qy, qz, tx, ty, tz, camera_id = struct.unpack("<idddddddi", binary)
                name = b""
                while True:
                    char = f.read(1)
                    if char == b"\x00": break
                    name += char
                name = name.decode("utf-8")
                num_points2d = struct.unpack("<Q", f.read(8))[0]
                f.seek(24 * num_points2d, 1)
                qvec = np.array([qw, qx, qy, qz])
                R = qvec2rotmat(qvec)
                T = np.array([tx, ty, tz])
                self.images[image_id] = {"R": R, "T": T, "camera_id": camera_id, "name": name}

    def _read_points3D_binary(self):
        with open(self.model_path / "points3D.bin", "rb") as f:
            num_points = struct.unpack("<Q", f.read(8))[0]
            xyzs = []
            rgbs = []
            for _ in range(num_points):
                binary = f.read(43) 
                data = struct.unpack("<QdddBBBd", binary)
                xyz = np.array(data[1:4])
                rgb = np.array(data[4:7])
                track_len = struct.unpack("<Q", f.read(8))[0]
                f.seek(8 * track_len, 1)
                xyzs.append(xyz)
                rgbs.append(rgb)
            self.points3D["xyz"] = np.array(xyzs) if xyzs else np.zeros((0, 3))
            self.points3D["rgb"] = np.array(rgbs) if rgbs else np.zeros((0, 3))

    def get_training_data(self, device="cuda"):
        cameras = []
        gt_images = []
        skipped_no_cam = 0
        skipped_no_img = 0
        
        for img_id, img_data in self.images.items():
            camera_id = img_data["camera_id"]
            if camera_id not in self.cameras:
                skipped_no_cam += 1
                continue  # Skip images with missing camera info
            cam_info = self.cameras[camera_id]
            img_path = self.images_dir / img_data["name"]
            img = cv2.imread(str(img_path))
            if img is None:
                skipped_no_img += 1
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img).float() / 255.0
            gt_images.append(img_tensor.to(device))
            R = torch.tensor(img_data["R"], dtype=torch.float32, device=device)
            T = torch.tensor(img_data["T"], dtype=torch.float32, device=device)
            W2C = torch.eye(4, device=device)
            W2C[:3, :3] = R
            W2C[:3, 3] = T
            K = torch.tensor(cam_info["K"], dtype=torch.float32, device=device)
            cameras.append({"viewmat": W2C, "K": K, "width": cam_info["w"], "height": cam_info["h"], "name": img_data["name"]})
        
        # Log any skipped cameras for debugging
        if skipped_no_cam > 0 or skipped_no_img > 0:
            logger.warning(f"ColmapLoader: Skipped {skipped_no_cam} (no camera), {skipped_no_img} (no image file)")
        
        logger.info(f"ColmapLoader: Loaded {len(cameras)} cameras, {len(self.cameras)} camera models, {len(self.images)} images in model")
        
        return cameras, gt_images, self.points3D

# ... (SSIM - No Change) ...
def create_window(window_size, channel):
    _1D_window = torch.tensor([math.exp(-(x - window_size // 2) ** 2 / float(2 * 1.5 ** 2)) for x in range(window_size)])
    _1D_window = _1D_window.unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel).to(img1.device)
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2
    C1 = 0.01 ** 2; C2 = 0.03 ** 2
    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return ssim_map.mean() if size_average else ssim_map.mean(1).mean(1).mean(1)

# --- Engine (Modern API for 1.4.0) ---

class TrainerEngine:
    def __init__(self, viser_server):
        self.viser_server = viser_server
        self.paused = False
        self.stop_requested = False
        self.current_step = 0
        self.max_steps = 3000
        self.status = "idle"
        self.log_queue = queue.Queue()
        self.executor = ThreadPoolExecutor(max_workers=1)
        self.training_future = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.current_process = None
        self.checklist = {
            "extract": "pending", 
            "filter": "pending", 
            "colmap": "pending", 
            "train": "pending"
        }
        self.image_count = 0
        self.start_time = None
        self.pipeline_message = "Ready"
        
        # Training State
        self.params = None
        self.optimizers = None
        self.strategy = None
        self.strategy_state = None
        self.train_cameras = None
        self.train_cameras = None
        self.animation_pose = None # (position, wxyz)
        self.extracted_images_count = 0
        self.used_images_count = 0
        self.colmap_progress = {"stage": "", "current": 0, "total": 0}
        self.current_data_dir = None

    def log(self, msg: str):
        logger.info(msg)
        self.log_queue.put(msg)

    def render_video(self, data_dir: Path, fps=30):
        frames_dir = data_dir / "animation_frames"
        if not frames_dir.exists():
            self.log("No animation frames found.")
            return False
        
        output_video = data_dir / "training_evolution.mp4"
        self.log(f"Rendering video to {output_video}...")
        
        cmd = [
            "ffmpeg", "-y", 
            "-framerate", str(fps), 
            "-i", str(frames_dir / "%05d.jpg"), 
            "-c:v", "libx264", 
            "-pix_fmt", "yuv420p", 
            str(output_video)
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.log("Video rendering complete!")
        return True

    def cancel_job(self):
        self.log("Stopping job...")
        self.stop_requested = True
        if self.current_process:
            try:
                self.current_process.terminate()
                self.current_process.wait(timeout=5)
            except Exception:
                self.current_process.kill()
            self.current_process = None
        self.status = "cancelled"
        self.pipeline_message = "Cancelled by user"
        for k in self.checklist:
            if self.checklist[k] == "running": self.checklist[k] = "pending"

    def process_video(self, input_path: Path, output_dir: Path, fps: int = 2, 
                      smart_selection: bool = False, blur_filter: bool = False, 
                      matcher_type: str = "sequential"):
        if self.stop_requested: return False
        
        # Reset checklist for new run
        for k in self.checklist:
            self.checklist[k] = "pending"
        self.extracted_images_count = 0
        self.used_images_count = 0
        
        images_dir = output_dir / "images"
        
        # If input is a directory, images are already there (multi-file upload)
        if input_path.is_dir():
            self.log(f"Using pre-uploaded images from {input_path.name}")
            images_dir = input_path
            self.checklist["extract"] = "completed"
        else:
            # Full cleanup for fresh start
            if output_dir.exists():
                self.log("Cleaning up previous run data...")
                try:
                    shutil.rmtree(output_dir)
                except Exception as e:
                    self.log(f"Warning: Failed to clean cleanup {output_dir}: {e}")
            
            output_dir.mkdir(parents=True, exist_ok=True)
            images_dir.mkdir(exist_ok=True)

            self.start_time = time.time()
            self.status = "extracting"
            self.pipeline_message = "Processing Input..."
            self.checklist["extract"] = "running"
            
            # 1. Extraction / Input Handling
            if input_path.suffix.lower() == ".zip":
                self.log(f"Extracting ZIP: {input_path.name}")
                try:
                    with zipfile.ZipFile(input_path, 'r') as zip_ref:
                        zip_ref.extractall(images_dir)
                    # Flatten if nested
                    for root, dirs, files in os.walk(images_dir):
                        for file in files:
                            if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                src = Path(root) / file
                                dst = images_dir / file
                                if src != dst: shutil.move(str(src), str(dst))
                except Exception as e:
                    self.log(f"ZIP Extraction failed: {e}")
                    return False
            else:
                # Smart selection extracts at 10fps (dense) then filters for diversity
                extract_fps = 10 if smart_selection else fps
                self.log(f"Extracting frames from video: {input_path.name} at {extract_fps} FPS" + (" (Smart)" if smart_selection else ""))
                cmd = ["ffmpeg", "-y", "-i", str(input_path), "-qscale:v", "1", "-qmin", "1", "-vf", f"fps={extract_fps}", str(images_dir / "%04d.jpg")]
                try:
                    self.current_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    self.current_process.wait()
                    if self.stop_requested: return False
                    if self.current_process.returncode != 0: raise Exception("FFmpeg failed")
                finally:
                    self.current_process = None
            
            self.checklist["extract"] = "completed"
        
        all_images = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.png")) + list(images_dir.glob("*.jpeg")))
        self.extracted_images_count = len(all_images)
        self.checklist["extract"] = "completed"

        if not all_images:
            self.log("Error: No images found.")
            return False

        # 2. Filtering (Smart & Blur)
        if self.stop_requested: return False
        self.checklist["filter"] = "running"
        self.pipeline_message = "Filtering Images..."
        
        kept_images = []
        last_hash = None
        
        self.log(f"Filtering {len(all_images)} images (Smart: {smart_selection}, Blur: {blur_filter})...")
        
        for i, img_path in enumerate(all_images):
            if self.stop_requested: return False
            if i % 10 == 0: self.pipeline_message = f"Filtering {i}/{len(all_images)}..."
            
            img = cv2.imread(str(img_path))
            if img is None: continue
            
            # Blur Check
            if blur_filter:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                variance = cv2.Laplacian(gray, cv2.CV_64F).var()
                if variance < 100: # Threshold
                    img_path.unlink()
                    continue

            # Smart Selection (Cohesion/Similarity)
            if smart_selection:
                # Simple perceptual hash: resize to 8x8, gray, compare
                # Or just use histogram correlation for speed/robustness
                # Let's use Histogram
                hist = cv2.calcHist([img], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
                hist = cv2.normalize(hist, hist).flatten()
                
                if last_hash is not None:
                    score = cv2.compareHist(last_hash, hist, cv2.HISTCMP_CORREL)
                    # If too similar (> 0.95), skip (duplicate/static)
                    # If too different (< 0.3), maybe keep (cut)? 
                    # User wants "cohesion", so maybe we want to avoid blurry transitions?
                    # Actually, usually we want to avoid DUPES.
                    if score > 0.98:
                        img_path.unlink()
                        continue
                last_hash = hist

            kept_images.append(img_path)

        self.used_images_count = len(kept_images)
        self.log(f"Filtering complete. Used {self.used_images_count} / {self.extracted_images_count} images.")
        self.checklist["filter"] = "completed"
        
        if self.used_images_count < 10:
             self.log("Warning: Very few images remaining. Reconstruction might fail.")

        # 3. COLMAP
        if self.stop_requested: return False
        self.status = "colmap"
        self.checklist["colmap"] = "running"
        self.pipeline_message = "COLMAP Reconstruction..."
        self.log(f"Starting COLMAP ({matcher_type})...")
        
        # Feature Extraction
        extract_cmd = [
            "colmap", "feature_extractor",
            "--database_path", str(output_dir / "database.db"),
            "--image_path", str(images_dir),
            "--ImageReader.camera_model", "OPENCV",
            "--SiftExtraction.use_gpu", "1"
        ]
        self._run_colmap_command(extract_cmd, "Extracting Features")
        if self.stop_requested: return False

        # Matching
        matcher_cmd = ["colmap", f"{matcher_type}_matcher", 
                       "--database_path", str(output_dir / "database.db"),
                       "--SiftMatching.use_gpu", "1"]
        self._run_colmap_command(matcher_cmd, "Matching Features")
        if self.stop_requested: return False

        # Mapper
        output_dir.joinpath("sparse").mkdir(exist_ok=True)
        mapper_cmd = [
            "colmap", "mapper",
            "--database_path", str(output_dir / "database.db"),
            "--image_path", str(images_dir),
            "--output_path", str(output_dir / "sparse"),
            "--Mapper.ba_global_function_tolerance", "0.000001"
        ]
        self._run_colmap_command(mapper_cmd, "Reconstructing 3D")
        if self.stop_requested: return False

        # Post-processing: Select the best COLMAP model (most registered images)
        sparse_dir = output_dir / "sparse"
        
        # Handle case where COLMAP outputs directly to sparse/ instead of sparse/0/
        if (sparse_dir / "cameras.bin").exists():
            sparse_dest = sparse_dir / "0"
            sparse_dest.mkdir(exist_ok=True)
            for f in ["cameras.bin", "images.bin", "points3D.bin"]:
                src = sparse_dir / f
                if src.exists(): shutil.move(str(src), str(sparse_dest / f))
        
        # Find the best model (most registered images)
        best_model = self._select_best_colmap_model(sparse_dir)
        if best_model is None:
            self.log("CRITICAL: No valid COLMAP reconstruction found!")
            return False
        
        # If best model is not sparse/0, move it there for consistency
        if best_model.name != "0":
            self.log(f"Selected model '{best_model.name}' as best (has most registered images)")
            target = sparse_dir / "0"
            backup = sparse_dir / "0_backup"
            if target.exists():
                shutil.move(str(target), str(backup))
            shutil.move(str(best_model), str(target))
        
        # Log how many images were registered
        with open(sparse_dir / "0" / "images.bin", "rb") as f:
            num_images = struct.unpack("<Q", f.read(8))[0]
        self.log(f"COLMAP finished successfully. Registered {num_images} images.")
        self.checklist["colmap"] = "completed"
        return True
    
    def _select_best_colmap_model(self, sparse_dir: Path) -> Optional[Path]:
        """Find the COLMAP model with the most registered images.
        
        Also considers models with at least 20% of input images as viable,
        and logs all model statistics for debugging.
        """
        models_info = []
        
        for model_dir in sparse_dir.iterdir():
            if not model_dir.is_dir():
                continue
            images_bin = model_dir / "images.bin"
            points_bin = model_dir / "points3D.bin"
            if not images_bin.exists():
                continue
            
            try:
                with open(images_bin, "rb") as f:
                    num_images = struct.unpack("<Q", f.read(8))[0]
                
                num_points = 0
                if points_bin.exists():
                    with open(points_bin, "rb") as f:
                        num_points = struct.unpack("<Q", f.read(8))[0]
                
                models_info.append({
                    "dir": model_dir,
                    "images": num_images,
                    "points": num_points,
                    "name": model_dir.name
                })
            except Exception as e:
                self.log(f"Warning: Could not read {images_bin}: {e}")
        
        if not models_info:
            return None
        
        # Sort by number of images (primary) and points (secondary)
        models_info.sort(key=lambda x: (x["images"], x["points"]), reverse=True)
        
        # Log all models
        self.log(f"Found {len(models_info)} COLMAP reconstruction(s):")
        for i, m in enumerate(models_info):
            coverage = int(100 * m["images"] / self.used_images_count) if self.used_images_count > 0 else 0
            marker = "→" if i == 0 else " "
            self.log(f"  {marker} Model '{m['name']}': {m['images']} images ({coverage}%), {m['points']} points")
        
        # Consider models with at least 20% coverage as "viable"
        min_images_threshold = max(5, int(0.2 * self.used_images_count))
        viable_models = [m for m in models_info if m["images"] >= min_images_threshold]
        
        if not viable_models:
            # No good models, use the best available
            self.log(f"Warning: No model has 20%+ coverage, using best available")
            return models_info[0]["dir"]
        
        # Return the best viable model (most images)
        best = viable_models[0]
        if len(viable_models) > 1:
            self.log(f"Selected model '{best['name']}' ({len(viable_models)} viable models found)")
        
        return best["dir"]

    def _run_colmap_command(self, cmd, stage_name):
        self.pipeline_message = f"COLMAP: {stage_name}..."
        self.colmap_progress = {"stage": stage_name, "current": 0, "total": 0}
        self.current_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        while True:
            if self.stop_requested:
                self.current_process.terminate()
                return
            line = self.current_process.stdout.readline()
            if not line and self.current_process.poll() is not None: break
            if line:
                line = line.strip()
                # Parse progress
                # "Processed file 5 / 100"
                if "Processed file" in line:
                     parts = line.split()
                     if len(parts) >= 4 and parts[-2] == "/":
                         try:
                             current = int(parts[-3])
                             total = int(parts[-1])
                             self.colmap_progress = {"stage": stage_name, "current": current, "total": total}
                             pct = int(100 * current / total) if total > 0 else 0
                             self.pipeline_message = f"{stage_name}: {current}/{total} ({pct}%)"
                         except ValueError:
                             pass
                elif "Registering image" in line:
                     # Registering image #5 (5)
                     parts = line.split("#")
                     if len(parts) > 1:
                         img_num = parts[1].split('(')[0].strip()
                         self.pipeline_message = f"{stage_name}: Registering #{img_num}"
                elif "Elapsed time" in line:
                     # COLMAP reports elapsed time at the end of each stage
                     self.log(f"[COLMAP] {line}")
                
                if "warning" in line.lower() or "error" in line.lower():
                    self.log(f"[COLMAP] {line}")
        
        if self.current_process.returncode != 0:
            raise Exception(f"COLMAP {stage_name} failed.")

    def start_training(self, data_dir: Path, max_steps: int = 3000, resume: bool = False):
        if self.training_future and not self.training_future.done(): return
        
        self.stop_requested = False
        self.paused = False
        
        if not resume:
            self.current_step = 0
        
        self.max_steps = max_steps
        self.status = "training"
        self.checklist["train"] = "running"
        self.pipeline_message = "Training Gaussian Splats..."
        self.training_future = self.executor.submit(self._training_loop, data_dir, resume)

    def _training_loop(self, data_dir: Path, resume: bool):
        self.log(f"Initializing Training (gsplat 1.4.0 engine) - Resume: {resume}...")
        self.current_data_dir = data_dir
        try:
            if not resume or self.params is None:
                sparse_path = data_dir / "sparse/0"
                loader = ColmapLoader(sparse_path, data_dir / "images")
                self.train_cameras, train_images, points3d = loader.get_training_data(self.device)
                
                if not self.train_cameras: raise Exception("No cameras found.")

                # Initialization Logic (KNN-based scale estimation)
                self.log("Computing initial scales via KNN...")
                means = torch.tensor(points3d["xyz"], dtype=torch.float32, device=self.device, requires_grad=True)
                
                # Efficient KNN for scale initialization
                with torch.no_grad():
                    # If too many points, use a subset for estimation to avoid OOM
                    # But for exact per-point scale, we need neighbor.
                    # Simple chunked approach for N^2 avoidance?
                    # For simple "VibeSplat", let's use a simplified heuristic if N is huge, or cdist if small.
                    # Actually, let's use the standard pytorch3d/simple-knn approach if available? No.
                    # Let's use a simple KDTree via scipy if available, or torch brute force in chunks.
                    # Since we don't have scipy, let's use torch cdist in chunks.
                    
                    scales_init = torch.ones((len(means), 3), dtype=torch.float32, device=self.device)
                    
                    # Chunked KNN (K=3, average distance)
                    chunk_size = 1024
                    N = len(means)
                    min_dists = []
                    
                    for i in range(0, N, chunk_size):
                        end = min(i + chunk_size, N)
                        batch = means[i:end]
                        # Compute distance to all other points (heavy!) -> Actually we only need local.
                        # If N > 10k, global cdist is too big.
                        # Fallback: Use a fixed decent scale relative to scene extent.
                        if N > 5000: 
                            # Heuristic: Scene extent / 100
                            # This is safer than crashing OOM
                            scene_extent = (means.max(dim=0)[0] - means.min(dim=0)[0]).mean().item()
                            avg_dist = max(scene_extent / 1000.0, 0.0001)
                            min_dists.extend([avg_dist] * (end - i))
                        else:
                            dists = torch.cdist(batch, means) # [B, N]
                            # Set self-distance to inf
                            dists.fill_diagonal_(float('inf')) # Only works if batch is view of means? No.
                            # We need to mask out the identity if batch is means.
                            # If batch is subset, we mask [i:end] cols.
                            for j in range(end-i):
                                dists[j, i+j] = float('inf')
                            
                            k_dists, _ = dists.topk(k=3, dim=1, largest=False)
                            avg = k_dists.mean(dim=1)
                            min_dists.append(avg)
                    
                    if isinstance(min_dists, list) and isinstance(min_dists[0], torch.Tensor):
                        dist_tensor = torch.cat(min_dists)
                    else:
                        dist_tensor = torch.tensor(min_dists, device=self.device)
                        
                    scales_init = torch.log(torch.clamp(dist_tensor, min=1e-7)).unsqueeze(1).repeat(1, 3)

                scales = scales_init.clone().requires_grad_(True)
                quats = torch.randn((len(means), 4), dtype=torch.float32, device=self.device); quats = F.normalize(quats, dim=-1); quats.requires_grad = True
                opacities = torch.zeros((len(means)), dtype=torch.float32, device=self.device); opacities.requires_grad = True
                rgbs = torch.tensor(points3d["rgb"], dtype=torch.float32, device=self.device) / 255.0
                colors = (rgbs - 0.5) / 0.28209; colors = colors.unsqueeze(1); colors.requires_grad = True

                self.params = {"means": means, "quats": quats, "scales": scales, "opacities": opacities, "colors": colors} 
                
                self.strategy = DefaultStrategy(verbose=False)
                
                self.optimizers = {
                    "means": torch.optim.Adam([means], lr=1.6e-4, eps=1e-15),
                    "quats": torch.optim.Adam([quats], lr=1e-3, eps=1e-15),
                    "scales": torch.optim.Adam([scales], lr=5e-3, eps=1e-15),
                    "opacities": torch.optim.Adam([opacities], lr=5e-2, eps=1e-15),
                    "colors": torch.optim.Adam([colors], lr=2.5e-3, eps=1e-15),
                }
                
                self.strategy_state = self.strategy.initialize_state(scene_scale=1.0)
            else:
                # We need to reload images if we are just resuming from memory but didn't persist images/cameras?
                # In this simple implementation, we assume self.train_cameras is still valid if params is valid.
                # If we restarted the server, we can't resume from memory. 
                # For now, 'Resume' only works if the python process didn't die.
                if self.train_cameras is None:
                     sparse_path = data_dir / "sparse/0"
                     loader = ColmapLoader(sparse_path, data_dir / "images")
                     self.train_cameras, train_images, _ = loader.get_training_data(self.device)
                else:
                     # We need train_images again as we don't store them (too big for self?)
                     # Let's just reload loader for images
                     sparse_path = data_dir / "sparse/0"
                     loader = ColmapLoader(sparse_path, data_dir / "images")
                     _, train_images, _ = loader.get_training_data(self.device)

            self.log(f"Starting Loop from step {self.current_step} to {self.max_steps}...")
            
            # Pre-compute black background (Batch size 1)
            bg_color = torch.zeros((1, 3), dtype=torch.float32, device=self.device)

            for step in range(self.current_step, self.max_steps):
                if self.stop_requested: 
                    self.status = "cancelled"; break
                if self.paused: time.sleep(0.1); continue

                cam_idx = torch.randint(0, len(self.train_cameras), (1,)).item()
                cam = self.train_cameras[cam_idx]; gt_image = train_images[cam_idx]

                # Modern gsplat API
                render_colors, render_alphas, info = rasterization(
                    means=self.params["means"], 
                    quats=self.params["quats"], 
                    scales=torch.exp(self.params["scales"]),
                    opacities=torch.sigmoid(self.params["opacities"]),
                    colors=self.params["colors"],
                    viewmats=cam["viewmat"].unsqueeze(0), 
                    Ks=cam["K"].unsqueeze(0), 
                    width=cam["width"], 
                    height=cam["height"],
                    sh_degree=0,
                    backgrounds=bg_color # Fix: Shape (1, 3)
                )
                
                render_colors = render_colors.squeeze(0)
                
                # Pre-backward
                self.strategy.step_pre_backward(self.params, self.optimizers, self.strategy_state, step, info)

                l1 = F.l1_loss(render_colors, gt_image)
                loss = l1
                if step % 10 == 0:
                     render_permuted = render_colors.permute(2, 0, 1).unsqueeze(0)
                     gt_permuted = gt_image.permute(2, 0, 1).unsqueeze(0)
                     loss += 0.2 * (1.0 - ssim(render_permuted, gt_permuted))

                loss.backward()
                
                for opt in self.optimizers.values():
                    opt.step()
                    opt.zero_grad(set_to_none=True)
                
                # Fix info tensor shapes
                if "radii" in info and info["radii"].ndim == 1:
                    info["radii"] = info["radii"].unsqueeze(0)
                if "means2d" in info and info["means2d"].ndim == 2:
                    m2d = info["means2d"]
                    m2d_view = m2d.view(1, *m2d.shape)
                    if m2d.grad is not None:
                        m2d_view.grad = m2d.grad.view(1, *m2d.grad.shape)
                    info["means2d"] = m2d_view

                self.strategy.step_post_backward(self.params, self.optimizers, self.strategy_state, step, info)
                
                self.current_step = step
                
                if step % 50 == 0: 
                    self.update_viser_scene(self.params)
                    
                    # Animation Capture
                    if self.animation_pose:
                        pos, wxyz = self.animation_pose
                        # Convert to viewmat (W2C)
                        # Viser gives wxyz, position
                        R = qvec2rotmat(np.array([wxyz[0], wxyz[1], wxyz[2], wxyz[3]]))
                        # W2C = [R^T, -R^T * t]
                        # Actually simpler: make C2W then invert
                        C2W = np.eye(4)
                        C2W[:3, :3] = R
                        C2W[:3, 3] = pos
                        W2C = np.linalg.inv(C2W)
                        
                        # Assume simple K from first camera or default
                        if self.train_cameras:
                            K = self.train_cameras[0]["K"].unsqueeze(0)
                            W = self.train_cameras[0]["width"]
                            H = self.train_cameras[0]["height"]
                        else:
                            # Fallback
                            K = torch.tensor([[800, 0, 400], [0, 800, 400], [0, 0, 1]], device=self.device).unsqueeze(0).float()
                            W, H = 800, 800

                        viewmat = torch.tensor(W2C, device=self.device, dtype=torch.float32).unsqueeze(0)
                        
                        with torch.no_grad():
                             anim_color, _, _ = rasterization(
                                means=self.params["means"], 
                                quats=self.params["quats"], 
                                scales=torch.exp(self.params["scales"]),
                                opacities=torch.sigmoid(self.params["opacities"]),
                                colors=self.params["colors"],
                                viewmats=viewmat, 
                                Ks=K, 
                                width=W, 
                                height=H,
                                sh_degree=0,
                                backgrounds=torch.zeros((1, 3), device=self.device) # Fix: Shape (1, 3)
                            )
                             img = anim_color.squeeze(0).cpu().numpy()
                             img = (np.clip(img, 0, 1) * 255).astype(np.uint8)
                             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                             
                             anim_dir = data_dir / "animation_frames"
                             anim_dir.mkdir(exist_ok=True)
                             cv2.imwrite(str(anim_dir / f"{step:05d}.jpg"), img)

                if step % 500 == 0: self.save_ply(data_dir / f"checkpoints/step_{step}.ply", self.params)

            if not self.stop_requested:
                self.status = "completed"
                self.checklist["train"] = "completed"
                self.pipeline_message = "Done!"
                self.log("Training Finished Successfully.")
                self.save_ply(data_dir / "checkpoints/final.ply", self.params)
                self.render_video(data_dir) # Auto-render


        except Exception as e:
            self.log(f"Training CRASHED: {e}")
            logger.exception("Training failed")
            self.status = "error"

    def update_viser_scene(self, params):
        if not self.viser_server: return
        
        # Get images directory for loading thumbnails
        images_dir = None
        if self.current_data_dir:
            images_dir = self.current_data_dir / "images"
        
        # Update Cameras - only on first update (step 0)
        # This avoids re-adding cameras every 50 steps which can cause flicker
        if self.train_cameras and self.current_step == 0:
            num_cams = len(self.train_cameras)
            self.log(f"Displaying {num_cams} cameras in viewport...")
            
            # Preload all thumbnails once
            thumbnails = {}
            if images_dir:
                for i, cam in enumerate(self.train_cameras):
                    if not cam.get("name"):
                        continue
                    img_path = images_dir / cam["name"]
                    if img_path.exists():
                        try:
                            thumb = cv2.imread(str(img_path))
                            if thumb is not None:
                                thumb = cv2.cvtColor(thumb, cv2.COLOR_BGR2RGB)
                                # Resize for performance but keep decent quality
                                h, w = thumb.shape[:2]
                                target_size = 512
                                scale_factor = min(target_size / w, target_size / h)
                                new_w = int(w * scale_factor)
                                new_h = int(h * scale_factor)
                                thumb = cv2.resize(thumb, (new_w, new_h))
                                thumbnails[i] = thumb
                        except Exception:
                            pass
            
            for i, cam in enumerate(self.train_cameras):
                try:
                    # cam["viewmat"] is W2C (4x4)
                    # We need C2W for position and quaternion
                    c2w = torch.inverse(cam["viewmat"]).cpu().numpy()
                    R = c2w[:3, :3]
                    t = c2w[:3, 3]
                    
                    # Calculate FoV
                    fov_y = 2 * math.atan(0.5 * cam["height"] / cam["K"][1,1].item())
                    aspect = cam["width"] / cam["height"]
                    
                    # Calculate camera orientation
                    wxyz = viser.transforms.SO3.from_matrix(R).wxyz
                    
                    # Frustum scale - determines the "depth" of the frustum visualization
                    frustum_scale = 0.25  # Larger for visibility
                    
                    # Add Frustum with image
                    frustum = self.viser_server.scene.add_camera_frustum(
                        f"/cameras/cam_{i}",
                        fov=fov_y,
                        aspect=aspect,
                        scale=frustum_scale,
                        wxyz=wxyz,
                        position=t,
                        color=(255, 200, 50),
                        image=thumbnails.get(i)  # Viser can display image directly on frustum!
                    )
                    
                    # Make frustum clickable - teleport to this camera view
                    def make_click_handler(pos, orientation, img_name):
                        def handler(event):
                            event.client.camera.position = pos
                            event.client.camera.wxyz = orientation
                            self.log(f"Jumped to camera: {img_name}")
                        return handler
                    
                    frustum.on_click(make_click_handler(t.copy(), wxyz, cam.get("name", f"cam_{i}")))
                    
                    # Set initial camera position and animation pose from FIRST camera
                    if i == 0:
                        # Set all connected clients to first camera view
                        for client in self.viser_server.get_clients().values():
                            client.camera.position = t
                            client.camera.wxyz = wxyz
                        
                        # Auto-set animation pose if not already set
                        if self.animation_pose is None:
                            self.animation_pose = (t.copy(), wxyz)
                            self.log("Auto-set animation view from first camera")
                    
                except Exception as e:
                    self.log(f"Warning: Failed to add camera {i}: {e}")


        with torch.no_grad():
            means = params["means"]
            quats = params["quats"]
            scales = torch.exp(params["scales"])
            opacities = torch.sigmoid(params["opacities"])
            colors = params["colors"].squeeze(1) * 0.28209 + 0.5
            colors = torch.clamp(colors, 0, 1)

            # Normalize quaternions (w, x, y, z)
            quats = F.normalize(quats, dim=-1)
            w, x, y, z = quats.unbind(-1)
            
            # Rotation matrix
            R = torch.stack([
                1 - 2*y**2 - 2*z**2, 2*x*y - 2*w*z, 2*x*z + 2*w*y,
                2*x*y + 2*w*z, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*w*x,
                2*x*z - 2*w*y, 2*y*z + 2*w*x, 1 - 2*x**2 - 2*y**2
            ], dim=-1).reshape(-1, 3, 3)
            
            # Scale matrix
            S = torch.diag_embed(scales)
            
            # Covariance = R * S * S * R.T
            M = torch.bmm(R, S)
            covs = torch.bmm(M, M.transpose(1, 2))

            self.viser_server.scene.add_gaussian_splats(
                "/gaussians", 
                centers=means.cpu().numpy(), 
                covariances=covs.cpu().numpy(), 
                rgbs=colors.cpu().numpy(), 
                opacities=opacities.unsqueeze(-1).cpu().numpy()
            )

    def save_ply(self, path: Path, params):
        path.parent.mkdir(parents=True, exist_ok=True)
        with torch.no_grad():
            m = params["means"].cpu().numpy()
            # Keep scales as log-encoded (SuperSplat expects log scale)
            s = params["scales"].cpu().numpy()
            # Normalize quaternions and ensure w-first format
            q = F.normalize(params["quats"], dim=-1).cpu().numpy()
            o = torch.sigmoid(params["opacities"]).cpu().numpy()
            # SH coefficient DC term - convert from our encoding back to standard
            c = params["colors"].squeeze(1).cpu().numpy() * 0.28209 + 0.5
            c = np.clip(c, 0, 1)  # Ensure valid color range
            
            # SuperSplat-compatible PLY format
            header = f"""ply
format binary_little_endian 1.0
element vertex {len(m)}
property float x
property float y
property float z
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
"""
            with open(path, "wb") as f:
                f.write(header.encode('ascii'))
                # Write binary data for each gaussian
                for i in range(len(m)):
                    # Position (x, y, z)
                    f.write(struct.pack('<fff', m[i,0], m[i,1], m[i,2]))
                    # SH DC coefficients (f_dc_0, f_dc_1, f_dc_2)
                    f.write(struct.pack('<fff', c[i,0], c[i,1], c[i,2]))
                    # Opacity (already sigmoid-ed)
                    f.write(struct.pack('<f', o[i]))
                    # Log-encoded scales
                    f.write(struct.pack('<fff', s[i,0], s[i,1], s[i,2]))
                    # Quaternion (w, x, y, z) - SuperSplat uses rot_0=w
                    f.write(struct.pack('<ffff', q[i,0], q[i,1], q[i,2], q[i,3]))

    def control(self, action: str):
        if action == "pause": self.paused = True
        elif action == "resume": self.paused = False
        elif action == "cancel": self.cancel_job()