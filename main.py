import os
import argparse
import subprocess


def main():
    parser = argparse.ArgumentParser(
        prog="VideoRAG",
        description="VideoRAG: A Video Retrieval and Analysis Generator",
    )
    # Entrambi devono avere i trattini per essere opzionali (flags)
    parser.add_argument(
        "--cmd", action="store_true", help="Start the command line interface"
    )
    parser.add_argument(
        "--desktop", action="store_true", help="Start the desktop interface"
    )
    parser.add_argument(
        "--install-llm-weights", action="store_true", help="Install LLM weights"
    )

    args = parser.parse_args()
    algo_path = os.path.join(os.getcwd(), "VideoRAG-algorithm")

    if args.cmd and args.desktop:
        raise ValueError("Only one of 'cmd' or 'desktop' can be specified.")

    if args.install_llm_weights:
        # Eseguiamo il comando partendo dall'interno di VideoRAG-algorithm
        subprocess.run(
            ["python", "-m", "src.installer.install_llm_weights"],
            cwd=algo_path
        )
    elif args.cmd:
        # Eseguiamo il comando partendo dall'interno di VideoRAG-algorithm
        subprocess.run(
            ["python", "-m", "src.scripts.run_local"],
            cwd=algo_path
        )
    elif args.desktop:
        print("Start desktop app...")
        # Qui probabilmente dovrai avviare Electron dalla cartella VIMO-Desktop
    else:
        print("Please specify --cmd, --desktop, or --install-llm-weights")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as ex:
        print(f"An error occurred: {ex}")
