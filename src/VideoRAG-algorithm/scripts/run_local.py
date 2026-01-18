import asyncio
import os
from pathlib import Path

import nest_asyncio

from ..videorag.llm import ollama_config
from ..videorag.videorag import QueryParam, VideoRAG

nest_asyncio.apply()


async def main():
    base_project_path = Path(__file__).resolve().parents[3]
    storage_path = base_project_path / "data" / "videorag_storage"

    # Ollama setup
    custom_llm = ollama_config
    custom_llm.model_name = "llama3:8b"
    custom_llm.embedding_model_name = "nomic-embed-text"
    os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
    os.environ["OPENAI_API_KEY"] = "ollama"  # Dummy value for Ollama

    # VideoRAG setup
    vrag = VideoRAG(working_dir=str(storage_path), llm=custom_llm)

    # Visual analysis model loading
    # Load the visual analysis model before proceeding
    # If you haven't downloaded MiniCPM locally, set debug=True to skip it
    print("Caricamento modelli di analisi visiva...")
    vrag.load_caption_model(debug=False)

    video_input = input("Inserisci il percorso del video: ").strip()
    video_path = str(Path(video_input).resolve())
    if not os.path.exists(video_path):
        print("Errore: File non trovato.")
        return

    print(f"--- Inizio analisi: {video_path} ---")
    await vrag.insert_video(video_path_list=[video_path])

    print("--- Analisi completata ---")
    query = input("Cosa vuoi sapere sul video?: ")

    # QueryParam setup
    param = QueryParam()
    param.mode = "videorag"

    print("\nGenerazione risposta in corso...")
    try:
        risposta = await vrag.aquery(query=query, param=param)
        print(f"\nRISPOSTA:\n{risposta}")
    except Exception as e:
        print(f"\nErrore durante la query: {e}")


if __name__ == "__main__":
    asyncio.run(main())
