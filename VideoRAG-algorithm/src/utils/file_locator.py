from pathlib import Path


class FileLocator:
    def __init__(self):
        self.ModelPath = Path(__file__).parent.parent.parent.parent / "models"
        self.MiniCPM = self.ModelPath / "MiniCPM-V-2_6-int4"
        self.FasterDistil = self.ModelPath / "faster-distil-whisper-large-v3"
        self.paths: list[Path] = [self.MiniCPM, self.FasterDistil]

        # Check all paths
        self.check_all_path()

    def check_all_path(self):
        for path in self.paths:
            if not path.exists():
                raise FileNotFoundError(f"File path not found: {path.resolve()}")
