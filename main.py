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

    # Logica di controllo
    if args.cmd and args.desktop:
        raise ValueError("Only one of 'cmd' or 'desktop' can be specified.")

    if args.install_llm_weights:
        subprocess.run(
            ["python", "-m", "src.VideoRAG-algorithm.installer.install_llm_weights"]
        )
    elif args.cmd:
        subprocess.run(["python", "-m", "src.VideoRAG-algorithm.scripts.run_local"])
    elif args.desktop:
        print("Start desktop app...")
    else:
        raise ValueError("Either '--cmd' or '--desktop' must be specified.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nExiting...")
    except Exception as ex:
        print(f"An error occurred: {ex}")
