# Project: VideoRAG and Vimo Desktop

This is a monorepo containing two main projects: `Vimo-desktop`, a desktop application for video analysis, and `VideoRAG-algorithm`, the core AI engine that powers it.

## Project Overview

*   **Vimo-desktop**: An Electron-based desktop application with a React frontend. It provides a user-friendly interface for interacting with the `VideoRAG-algorithm`. Users can upload videos, ask questions about them, and get intelligent answers.
*   **VideoRAG-algorithm**: A Python-based, state-of-the-art AI framework that uses retrieval-augmented generation (RAG) to understand and analyze video content. It can process extremely long videos and answer complex questions about them.

## Building and Running

### Vimo-desktop

The desktop application consists of a React frontend and a Python backend.

**1. Set Up and Start the Backend Service:**

The backend is a Flask server that exposes the `VideoRAG-algorithm`'s functionality.

```bash
# Create and activate a conda environment
conda create --name vimo python=3.11
conda activate vimo

# Install dependencies from Vimo-desktop/python_backend
pip install -r Vimo-desktop/python_backend/requirements.txt # Assuming a requirements.txt exists or will be created

# Navigate to the backend directory and start the API server
cd Vimo-desktop/python_backend
python videorag_api.py
```

**2. Launch the Frontend Application:**

The frontend is a React application built with Vite.

```bash
# Navigate to the desktop app's root directory
cd Vimo-desktop

# Install dependencies
pnpm install

# Start the development server
pnpm dev
```

### VideoRAG-algorithm

The `VideoRAG-algorithm` can be run independently for development and testing.

**1. Set Up the Environment:**

```bash
# Create and activate a conda environment
conda create --name videorag python=3.11
conda activate videorag

# Install dependencies
pip install -r VideoRAG-algorithm/requirements.txt
```

**2. Download Model Checkpoints:**

```bash
# Ensure git-lfs is installed
git lfs install

# Download MiniCPM-V model
git lfs clone https://huggingface.co/openbmb/MiniCPM-V-2_6-int4

# Download Whisper model
git lfs clone https://huggingface.co/Systran/faster-distil-whisper-large-v3

# Download ImageBind checkpoint
mkdir .checkpoints
cd .checkpoints
wget https://dl.fbaipublicfiles.com/imagebind/imagebind_huge.pth
cd ..
```

**3. Run the Algorithm:**

The `run_local.py` script provides a quick way to test the algorithm.

```python
# In VideoRAG-algorithm/run_local.py
import os
import logging
import warnings
import multiprocessing

warnings.filterwarnings("ignore")
logging.getLogger("httpx").setLevel(logging.WARNING)

# Please enter your openai key
os.environ["OPENAI_API_KEY"] = ""

from videorag._llm import openai_4o_mini_config
from videorag import VideoRAG, QueryParam


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    # Please enter your video file path in this list; there is no limit on the length.
    # Here is an example; you can use your own videos instead.
    video_paths = [
        'movies/Iron-Man.mp4',
        'movies/Spider-Man.mkv',
    ]
    videorag = VideoRAG(llm=openai_4o_mini_config, working_dir=f"./videorag-workdir")
    videorag.insert_video(video_path_list=video_paths)
```

## Development Conventions

*   **Monorepo Structure**: The project is organized as a monorepo, with the frontend and backend in separate directories.
*   **Python Backend**: The AI/ML logic is implemented in Python using PyTorch. The backend for the desktop app is a Flask server.
*   **React Frontend**: The desktop app's UI is built with React and TypeScript.
*   **Dependency Management**: The Python projects use `pip` and `conda` for dependency management. The frontend uses `pnpm`.
*   **Testing**: The `VideoRAG-algorithm` directory includes a `tests` directory, suggesting that there are tests for the algorithm. The `Vimo-desktop` project does not have an obvious top-level `tests` directory.
*   **Configuration**: The projects use a combination of Python files (e.g., `config.py`) and environment variables (e.g., `OPENAI_API_KEY`) for configuration.
