import math
import struct
import time
import logging
import subprocess
import shutil
import queue
import re
import threading
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
                params_cnt = {0: 3, 1: 4, 2: 4}.get(model_id, 3)
                params = struct.unpack(f"<{params_cnt}d", f.read(8 * params_cnt))
                K = np.eye(3)
                if model_id == 0 or model_id == 2: 
                    focal, cx, cy = params[0], params[1], params[2]
                    K[0, 0] = focal; K[1, 1] = focal; K[0, 2] = cx; K[1, 2] = cy
                elif model_id == 1: 
                    fx, fy, cx, cy = params
                    K[0, 0] = fx; K[1, 1] = fy; K[0, 2] = cx; K[1, 2] = cy
                self.cameras[camera_id] = {"K": K, "w": width, "h": height}

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
        for img_id, img_data in self.images.items():
            cam_info = self.cameras[img_data["camera_id"]]
            img_path = self.images_dir / img_data["name"]
            img = cv2.imread(str(img_path))
            if img is None: continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img).float() / 255.0
            gt_images.append(img_tensor.to(device))
            R = torch.tensor(img_data["R"], dtype=torch.float32, device=device)
            T = torch.tensor(img_data["T"], dtype=torch.float32, device=device)
            W2C = torch.eye(4, device=device)
            W2C[:3, :3] = R
            W2C[:3, 3] = T
            K = torch.tensor(cam_info["K"], dtype=torch.float32, device=device)
            cameras.append({"viewmat": W2C, "K": K, "width": cam_info["w"], "height": cam_info["h"]})
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
        self.animation_pose = None # (position, wxyz)

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

    def process_video(self, video_path: Path, output_dir: Path, fps: int = 2):
        if self.stop_requested: return False
        output_dir.mkdir(parents=True, exist_ok=True)
        images_dir = output_dir / "images"
        images_dir.mkdir(exist_ok=True)

        self.start_time = time.time()
        self.status = "extracting"
        self.pipeline_message = "Extracting Frames..."
        self.checklist["extract"] = "running"
        self.log(f"Processing video: {video_path.name} at {fps} FPS")

        cmd = ["ffmpeg", "-y", "-i", str(video_path), "-qscale:v", "1", "-qmin", "1", "-vf", f"fps={fps}", str(images_dir / "%04d.jpg")]
        try:
            self.current_process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            self.current_process.wait()
            if self.stop_requested: return False
            if self.current_process.returncode != 0: raise Exception("FFmpeg failed")
        finally:
            self.current_process = None
        self.checklist["extract"] = "completed"

        if self.stop_requested: return False
        self.checklist["filter"] = "running"
        self.pipeline_message = "Filtering Blurry Images..."
        self.log("Running blur filter...")
        images = list(images_dir.glob("*.jpg"))
        if not images:
            self.log("Error: No images extracted.")
            return False

        count = 0
        for img_path in images:
            if self.stop_requested: return False
            img = cv2.imread(str(img_path))
            if img is None: continue
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if variance < 100:
                img_path.unlink()
                count += 1
        
        self.image_count = len(list(images_dir.glob("*.jpg")))
        self.log(f"Removed {count} blurry images. Using {self.image_count} images.")
        self.checklist["filter"] = "completed"

        if self.stop_requested: return False
        self.status = "colmap"
        self.checklist["colmap"] = "running"
        self.pipeline_message = "COLMAP Reconstruction..."
        self.log("Starting COLMAP reconstruction...")
        
        colmap_cmd = [
            "colmap", "automatic_reconstructor", 
            "--workspace_path", str(output_dir), 
            "--image_path", str(images_dir), 
            "--quality", "medium",
            "--use_gpu", "1"
        ]
        
        self.current_process = subprocess.Popen(colmap_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        while True:
            if self.stop_requested:
                self.current_process.terminate()
                return False
            line = self.current_process.stdout.readline()
            if not line and self.current_process.poll() is not None: break
            if line:
                line = line.strip()
                if "Processed file" in line or "Matching block" in line: continue
                self.log(f"[COLMAP] {line}")
                if "Extracting features" in line: self.pipeline_message = "COLMAP: Extracting..."
                elif "Matching features" in line: self.pipeline_message = "COLMAP: Matching..."
                elif "Reconstruction" in line: self.pipeline_message = "COLMAP: Reconstructing..."
        
        ret = self.current_process.poll()
        self.current_process = None
        if ret != 0:
            if self.stop_requested: return False
            self.log(f"COLMAP failed with return code {ret}.")
            return False

        if not (output_dir / "sparse/0/cameras.bin").exists():
             if (output_dir / "sparse/cameras.bin").exists():
                sparse_dest = output_dir / "sparse/0"
                sparse_dest.mkdir(exist_ok=True)
                for f in ["cameras.bin", "images.bin", "points3D.bin"]:
                    src = output_dir / "sparse" / f
                    if src.exists(): shutil.move(str(src), str(sparse_dest / f))
             else:
                self.log("CRITICAL: COLMAP failed to produce sparse/0/cameras.bin")
                raise Exception("COLMAP Reconstruction Failed.")

        self.log("COLMAP finished successfully.")
        self.checklist["colmap"] = "completed"
        return True

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
        try:
            if not resume or self.params is None:
                sparse_path = data_dir / "sparse/0"
                loader = ColmapLoader(sparse_path, data_dir / "images")
                self.train_cameras, train_images, points3d = loader.get_training_data(self.device)
                
                if not self.train_cameras: raise Exception("No cameras found.")

                means = torch.tensor(points3d["xyz"], dtype=torch.float32, device=self.device, requires_grad=True)
                scales = torch.ones((len(means), 3), dtype=torch.float32, device=self.device) * np.log(0.1) # Better init?
                scales.requires_grad = True
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
            
            # Pre-compute black background
            bg_color = torch.zeros(3, dtype=torch.float32, device=self.device)

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
                    backgrounds=bg_color # Fix: Dark background
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
                                backgrounds=torch.zeros(3, device=self.device)
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

        except Exception as e:
            self.log(f"Training CRASHED: {e}")
            logger.exception("Training failed")
            self.status = "error"

    def update_viser_scene(self, params):
        if not self.viser_server: return
        
        # Update Cameras
        if self.train_cameras:
             for i, cam in enumerate(self.train_cameras):
                  # Viser expects standard [R|t]
                  # cam["viewmat"] is W2C (4x4)
                  # We need C2W for position and quaternion
                  c2w = torch.inverse(cam["viewmat"]).cpu().numpy()
                  # viser.scene.add_camera_frustum uses wxyz orientation and position
                  # But converting 4x4 to pos/quat is easier with utilities, or manually
                  # c2w is [ [R, t], [0, 1] ]
                  R = c2w[:3, :3]
                  t = c2w[:3, 3]
                  
                  # Convert R to wxyz
                  # Or just use add_camera_frustum(..., wxyz=..., position=...)
                  # We can use trimesh or scipy, but we have qvec2rotmat... need reverse?
                  # Let's just use a simple frustum visualizer or just points for now if complex.
                  # Actually, viser has a nice helper if we just pass the pose.
                  self.viser_server.scene.add_camera_frustum(
                      f"/cameras/cam_{i}",
                      fov=2 * math.atan(0.5 * cam["height"] / cam["K"][1,1].item()),
                      aspect=cam["width"] / cam["height"],
                      scale=0.1,
                      wxyz=viser.transforms.SO3.from_matrix(R).wxyz,
                      position=t,
                      color=(255, 255, 0)
                  )

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
            s = torch.exp(params["scales"]).cpu().numpy()
            q = params["quats"].cpu().numpy()
            o = torch.sigmoid(params["opacities"]).cpu().numpy()
            c = params["colors"].squeeze(1).cpu().numpy() * 0.28209 + 0.5
            
            header = f"""ply
format ascii 1.0
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
            with open(path, "w") as f:
                f.write(header)
                for i in range(len(m)):
                    f.write(f"{m[i,0]} {m[i,1]} {m[i,2]} {c[i,0]} {c[i,1]} {c[i,2]} {o[i]} {s[i,0]} {s[i,1]} {s[i,2]} {q[i,0]} {q[i,1]} {q[i,2]} {q[i,3]}\n")

    def control(self, action: str):
        if action == "pause": self.paused = True
        elif action == "resume": self.paused = False
        elif action == "cancel": self.cancel_job()