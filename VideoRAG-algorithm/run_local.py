import os
import asyncio
import nest_asyncio
from videorag import VideoRAG, QueryParam
from videorag._llm import ollama_config

nest_asyncio.apply()

def main():
    # Ollama setup
    custom_llm = ollama_config
    custom_llm.model_name = "llama3:8b" # Assicurati che il nome sia identico a quello in 'ollama list'
    custom_llm.embedding_model_name = "nomic-embed-text"
    os.environ["OPENAI_API_BASE"] = "http://localhost:11434/v1"
    os.environ["OPENAI_API_KEY"] = "ollama"

    # VideoRAG setup
    vrag = VideoRAG(
        working_dir="./videorag_storage",
        llm=custom_llm
    )

    # Visual analysis model loading
    # Load the visual analysis model before proceeding
    # If you haven't downloaded MiniCPM locally, set debug=True to skip it
    print("Caricamento modelli di analisi visiva...")
    vrag.load_caption_model(debug=False) 

    video_path = input("Inserisci il percorso del video: ").strip()
    if not os.path.exists(video_path):
        print("Errore: File non trovato.")
        return

    print(f"--- Inizio analisi: {video_path} ---")
    vrag.insert_video(video_path_list=[video_path])
    
    print("--- Analisi completata ---")
    query = input("Cosa vuoi sapere sul video?: ")

    # QueryParam setup
    param = QueryParam()
    param.mode = "videorag" 
    
    print("\nGenerazione risposta in corso...")
    try:
        risposta = vrag.query(query=query, param=param)
        print(f"\nRISPOSTA:\n{risposta}")
    except Exception as e:
        print(f"\nErrore durante la query: {e}")

if __name__ == "__main__":
    main()