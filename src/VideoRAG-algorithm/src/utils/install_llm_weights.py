import importlib.util
import subprocess
import sys
from pathlib import Path

try:
    from .file_locator import FileLocator
except (ImportError, ValueError):
    from file_locator import FileLocator
from huggingface_hub import snapshot_download


class LLMWeightsInstaller:
    def __init__(self, models_path: str | Path | None = None):
        self.fl = FileLocator(check_paths=False)
        if models_path:
            self.MODEL_PATH = Path(models_path).resolve()
        else:
            self.MODEL_PATH = self.fl.ModelPath

        # paths check
        self.fl.check_all_paths([self.fl.ModelPath], create_if_not_exists=True)

    def install_package(
        self, package_name: str, import_name: str, pip_install_cmd=None
    ):
        """
        Generic function to check if a python package is installed.
        If not, installs it using pip.
        """
        print(f"\n🔍 Checking library: {package_name}")

        # Check if package is already installed
        spec = importlib.util.find_spec(import_name)
        if spec is not None:
            print(f"✅ {package_name} is already installed.")
            return

        print(f"⬇️  {package_name} not found. Installing...")

        # Determine the install command (default is just the package name)
        if pip_install_cmd is None:
            cmd = [sys.executable, "-m", "pip", "install", package_name]
        else:
            # For complex installs like git urls
            cmd = [sys.executable, "-m", "pip", "install"] + pip_install_cmd.split()

        try:
            subprocess.check_call(cmd)
            print(f"✅ {package_name} installed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"❌ Error installing {package_name}: {e}")

    def download_if_missing(self, repo_id: str, folder_path: Path):
        """
        Downloads model weights if config.json is missing in the target folder.
        """
        check_file = folder_path / "config.json"

        print(f"\n🔍 Checking model weights: {repo_id}")
        print(f"   Destination: {folder_path}")

        if folder_path.exists() and check_file.exists():
            print("✅ Found existing weights. Skipping download.")
        else:
            print("⬇️  Weights not found. Downloading...")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=folder_path,
                    local_dir_use_symlinks=False,
                )
                print("✅ Download completed.")
            except Exception as e:
                print(f"❌ Error downloading weights: {e}")


if __name__ == "__main__":
    installer: LLMWeightsInstaller = LLMWeightsInstaller()
    print(f"📂 Model path detected: {installer.MODEL_PATH}")
    # 1. Install Faster Whisper (Required for faster-distil-whisper-large-v3)
    #    Pip name: faster-whisper, Import name: faster_whisper
    installer.install_package(
        package_name="faster-whisper", import_name="faster_whisper"
    )

    # 2. Install ImageBind (Required for video embeddings)
    #    Installed from Git URL
    installer.install_package(
        package_name="ImageBind",
        import_name="imagebind",
        pip_install_cmd="git+https://github.com/facebookresearch/ImageBind.git",
    )

    # --- PHASE 2: DOWNLOAD MODEL WEIGHTS ---
    # Download Model Weights (Audio)
    installer.download_if_missing(
        repo_id="Systran/faster-distil-whisper-large-v3",
        folder_path=installer.fl.FasterDistil,
    )

    # Download Model Weights (Vision)
    installer.download_if_missing(
        repo_id="openbmb/MiniCPM-V-2_6-int4", folder_path=installer.fl.MiniCPM
    )

    print("\n🎉 All installations and downloads completed.")
