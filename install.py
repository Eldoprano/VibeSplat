import sys
import shutil
import subprocess
import platform
import os
from pathlib import Path

def run_command(cmd, shell=False):
    """Runs a shell command and prints output."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        env = os.environ.copy()
        env["UV_HTTP_TIMEOUT"] = "600"
        subprocess.check_call(cmd, shell=shell, env=env)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        sys.exit(1)

def check_uv():
    if not shutil.which("uv"):
        print("[ERROR] 'uv' not found. Please install uv first.")
        print("curl -LsSf https://astral.sh/uv/install.sh | sh")
        sys.exit(1)
    print("[OK] uv found.")

def install_system_colmap():
    """Attempts to install COLMAP using system package managers."""
    if shutil.which("colmap"):
        print("[OK] COLMAP is already installed.")
        return

    print("[INFO] COLMAP not found. Attempting installation...")
    system = platform.system()
    
    if system == "Linux":
        if shutil.which("apt-get"):
            print("Detected Debian/Ubuntu. Running apt-get...")
            run_command(["sudo", "apt-get", "update"])
            run_command(["sudo", "apt-get", "install", "-y", "colmap", "ffmpeg"])
        elif shutil.which("pacman"):
            print("Detected Arch Linux. Running pacman...")
            run_command(["sudo", "pacman", "-S", "--noconfirm", "colmap", "ffmpeg"])
        elif shutil.which("dnf"):
            print("Detected Fedora. Running dnf...")
            run_command(["sudo", "dnf", "install", "-y", "colmap", "ffmpeg"])
        else:
            print("[WARNING] Could not detect package manager. Please install COLMAP manually.")
    elif system == "Windows":
        print("[WARNING] On Windows, please download COLMAP and add it to your PATH.")
        print("Download: https://github.com/colmap/colmap/releases")

def setup_python_deps():
    """Installs Python dependencies with specific fixes."""
    print("[INFO] Syncing dependencies via uv...")
    
    # 1. Sync basic env
    run_command(["uv", "sync"])

    # 2. Fix OpenCV (Qt Crash Fix)
    print("[INFO] Enforcing opencv-python-headless...")
    run_command(["uv", "pip", "uninstall", "opencv-python"])
    run_command(["uv", "pip", "install", "opencv-python-headless"])

    # 3. Install PyTorch 2.1.2 (Stable for gsplat wheels)
    print("[INFO] Installing PyTorch 2.1.2 (CUDA 12.1)...")
    run_command([
        "uv", "pip", "install", 
        "torch==2.1.2", "torchvision==0.16.2", 
        "--index-url", "https://download.pytorch.org/whl/cu121"
    ])

    print("[INFO] Installing gsplat (pre-built)...")
    # Use gsplat wheel for PyTorch 2.1 + CUDA 12.1
    run_command([
        "uv", "pip", "install", "gsplat==1.4.0",
        "--extra-index-url", "https://docs.gsplat.studio/whl/pt21cu121"
    ])

def main():
    print("=== OpenSplat Robust Installer ===")
    check_uv()
    install_system_colmap()
    setup_python_deps()
    
    print("\n=== Installation Complete ===")
    print("Run the server with:")
    print("  uv run server.py")

if __name__ == "__main__":
    main()