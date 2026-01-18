import importlib.util
import subprocess
import sys
from pathlib import Path

from huggingface_hub import snapshot_download


class LLMWeightsInstaller:
    def __init__(
        self,
        current_script_path: str | Path | None = None,
        installer_dir: str | Path | None = None,
        project_root: str | Path | None = None,
    ):
        if current_script_path:
            self.CURRENT_SCRIPT = Path(current_script_path).resolve()
        else:
            self.CURRENT_SCRIPT = Path(__file__).resolve()

        if installer_dir:
            self.INSTALLER_DIR = Path(installer_dir).resolve()
        else:
            self.INSTALLER_DIR = self.CURRENT_SCRIPT.parent

        if project_root:
            self.PROJECT_ROOT = Path(project_root).resolve()
        else:
            self.PROJECT_ROOT = self.INSTALLER_DIR.parent.parent.parent / "models"

        # paths check
        self.check_path_exists(
            [self.PROJECT_ROOT, self.INSTALLER_DIR, self.CURRENT_SCRIPT]
        )

    def check_path_exists(self, paths: list[Path]):
        """
        Utility function to check if a given path exists.
        """
        for p in paths:
            if not p.exists():
                raise FileNotFoundError(f"Path not found: {p}")

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

    def download_if_missing(self, repo_id: str, folder_name: str):
        """
        Downloads model weights if config.json is missing in the target folder.
        """
        dest_path = self.PROJECT_ROOT / folder_name
        check_file = dest_path / "config.json"

        print(f"\n🔍 Checking model weights: {repo_id}")
        print(f"   Destination: {dest_path}")

        if dest_path.exists() and check_file.exists():
            print("✅ Found existing weights. Skipping download.")
        else:
            print("⬇️  Weights not found. Downloading...")
            try:
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=dest_path,
                    local_dir_use_symlinks=False,
                )
                print("✅ Download completed.")
            except Exception as e:
                print(f"❌ Error downloading weights: {e}")


if __name__ == "__main__":
    installer: LLMWeightsInstaller = LLMWeightsInstaller()
    print(f"📂 Project Root detected: {installer.PROJECT_ROOT}")
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

    # 3. Download Model Weights (Audio)
    installer.download_if_missing(
        repo_id="Systran/faster-distil-whisper-large-v3",
        folder_name="faster-distil-whisper-large-v3",
    )

    # 4. Download Model Weights (Vision)
    installer.download_if_missing(
        repo_id="openbmb/MiniCPM-V-2_6-int4", folder_name="MiniCPM-V-2_6-int4"
    )

    print("\n🎉 All installations and downloads completed.")
