from pathlib import Path


class FileLocator:
    def __init__(self, check_paths: bool = True):
        self.ProjectRoot: Path = Path(__file__).parent.parent.parent.parent.parent
        self.ModelPath = self.ProjectRoot / "models"
        self.MiniCPM = self.ModelPath / "MiniCPM-V-2_6-int4"
        self.FasterDistil = self.ModelPath / "faster-distil-whisper-large-v3"
        self.paths: list[Path] = [self.MiniCPM, self.FasterDistil]

        # Check all paths
        if check_paths:
            self.check_all_paths(create_if_not_exists=False)

    def check_all_paths(
        self, paths_list: list[Path] = [], create_if_not_exists: bool = False
    ):
        paths: list[Path] = paths_list if paths_list else self.paths
        for path in paths:
            try:
                if not path.exists() and create_if_not_exists:
                    path.mkdir(parents=True, exist_ok=True)
                elif not path.exists():
                    raise FileNotFoundError(f"File path not found: {path.resolve()}")
            except PermissionError:
                raise PermissionError(f"Permission denied to create directory: {path}")
