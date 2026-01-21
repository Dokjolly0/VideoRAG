import argparse
import os
import subprocess
from pathlib import Path

# Costanti per i percorsi
ROOT_DIR = Path(__file__).parent.resolve()
ALGO_DIR = ROOT_DIR / "VideoRAG-algorithm"


def get_conda_python(env_name: str) -> str:
    """Restituisce il percorso dell'eseguibile python dell'ambiente specifico."""
    try:
        output = subprocess.check_output(["conda", "info", "--envs"], encoding="utf-8")
        for line in output.splitlines():
            if env_name in line:
                # Prende il path dell'ambiente (ultima colonna)
                env_path = line.split()[-1]
                if os.name == "nt":
                    return str(Path(env_path) / "python.exe")
                return str(Path(env_path) / "bin" / "python")
    except Exception:
        pass
    return "python"  # Fallback


def check_conda_env_exists(env_name: str) -> bool:
    try:
        output = subprocess.check_output(["conda", "env", "list"], encoding="utf-8")
        return any(
            line.startswith(env_name) or f" {env_name} " in line
            for line in output.splitlines()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def init_project():
    print("--- Init VideoRAG Project ---")
    env_name = "videorag"

    # 1. Creazione ambiente se non esiste
    if not check_conda_env_exists(env_name):
        print(f"Creating conda environment '{env_name}'...")
        subprocess.run(
            ["conda", "create", "-n", env_name, "python=3.10", "-y"], check=True
        )

    # 2. Otteniamo il Python corretto dell'ambiente
    python_env = get_conda_python(env_name)
    print(f"Using environment python: {python_env}")

    # 3. Installazione dipendenze
    req_path = ALGO_DIR / "requirements.txt"
    if req_path.exists():
        print("Installing dependencies...")
        subprocess.run(
            [python_env, "-m", "pip", "install", "-r", str(req_path)], check=True
        )

    # 4. Check Torch (usando il python dell'ambiente)
    torch_script = ALGO_DIR / "src" / "scripts" / "check_torch.py"
    if torch_script.exists():
        result = subprocess.run([python_env, str(torch_script)])
        if result.returncode != 0:
            print("⚠️ Warning: Torch check failed or CUDA not available.")

    # 5. Download pesi LLM
    print("Checking LLM Weights...")
    subprocess.run(
        [python_env, "-m", "src.utils.install_llm_weights"], cwd=str(ALGO_DIR)
    )

    # 6. Ollama Pull
    models = ["llama3:8b", "nomic-embed-text", "olmo2", "gemma2:latest"]
    for model in models:
        print(f"Ollama pull: {model}")
        subprocess.run(["ollama", "pull", model])


def main():
    parser = argparse.ArgumentParser(prog="VideoRAG")
    parser.add_argument(
        "--init", action="store_true", help="Initialize the project and environment"
    )
    parser.add_argument("--cmd", action="store_true", help="Start CLI")
    parser.add_argument("--desktop", action="store_true", help="Start Desktop UI")

    args = parser.parse_args()
    python_env = get_conda_python("videorag")

    if args.init:
        init_project()
    elif args.cmd:
        subprocess.run([python_env, "-m", "src.scripts.run_local"], cwd=str(ALGO_DIR))
    elif args.desktop:
        print("Starting desktop app (Electron)...")
        # subprocess.run(["npm", "start"], cwd=ROOT_DIR / "VIMO-Desktop")
    else:
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nAborted.")
    except Exception as e:
        print(f"Error: {e}")
