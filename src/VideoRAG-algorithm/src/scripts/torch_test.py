import sys

import torch

is_installed = torch.__version__ is not None
is_nvidia_gpu_available = torch.cuda.is_available()


def check_torch():
    print("Checking Torch installation...")
    if is_installed:
        print("Torch is installed.")
        print("Torch version:", torch.__version__)
        print("CUDA version:", torch.version.cuda)
        sys.exit(0)
    else:
        print("Torch is not installed.")
        sys.exit(1)


check_torch()
