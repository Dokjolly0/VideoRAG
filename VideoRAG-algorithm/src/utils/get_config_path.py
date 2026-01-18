import json
from pathlib import Path
from typing import Optional, Union


def get_config_path(
    key: str,
    config_file: Union[str, Path] = "config.json",
) -> Optional[Path]:
    """
    Return the absolute Path starting from the directory of the config.json.
    If the key does not exist, return None.
    """
    config_path = Path(config_file)

    if not config_path.is_absolute():
        config_path = Path.cwd() / config_path

    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} non trovato")

    with config_path.open(encoding="utf-8") as f:
        config = json.load(f)

    value = config.get(key)
    if value is None:
        return None

    return (config_path.parent / value).resolve()
