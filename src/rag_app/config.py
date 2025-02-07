# config.py
import torch
import os

# GPU Configuration
gpu_config = {
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "gpu_layers": 32,
    "batch_size": 512
}

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

llm_config = {
    "max_tokens": 300,
    "n_ctx": 4096,
    "temperature": 0.7,
    "model_path": os.path.join(BASE_DIR, "models", "llama-2-7b.Q4_K_M.gguf"),
    **gpu_config
}