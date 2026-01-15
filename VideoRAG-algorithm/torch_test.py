import torch

is_installed = torch.__version__ is not None
is_nvidia_gpu_available = torch.cuda.is_available()

print(f"Torch installed: {is_installed}")
print(f"NVIDIA GPU available: {is_nvidia_gpu_available}")
print(f"Torch version: {torch.__version__}")
