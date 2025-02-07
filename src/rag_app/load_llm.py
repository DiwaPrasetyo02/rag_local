# load_llm.py
import torch
from llama_cpp import Llama
from typing import Dict, Any
from .config import llm_config

class LocalLLM:
    def __init__(self):
        
        # Check GPU availability
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # GPU Configuration
        gpu_config = {
            "n_gpu_layers": 32,  # Total layers that will be runned by GPU
            "use_cuda": True,    # Activating CUDA
            "n_batch": 512,      # Batch size for inference
            "n_threads": 4,      # Thread for CPU operations
        }

        try:
            # Initializing model with GPU support
            self.llm = Llama(
                model_path=llm_config["model_path"],
                n_ctx=llm_config["n_ctx"],
                **gpu_config
            )
            print(f"Model loaded on: {self.device}")
            if self.device == "cuda":
                print(f"GPU Memory Allocated: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
        except Exception as e:
            print(f"Error initializing model: {str(e)}")
            raise

    def generate_response(self, prompt: str, **kwargs) -> Dict[str, Any]:
        """
        Generater response using Local LLama model's with GPU acceleration
        """
        try:
            # Set cuda stream for better GPU utilization
            with torch.cuda.stream(torch.cuda.Stream()):
                output = self.llm.create_completion(
                    prompt=prompt,
                    max_tokens=kwargs.get("max_tokens", llm_config["max_tokens"]),
                    temperature=kwargs.get("temperature", llm_config["temperature"]),
                    top_p=0.95,
                    top_k=40,
                    repeat_penalty=1.1,
                    stop=["Human:", "Assistant:"],
                )
            
            # Clear CUDA cache after inference
            if self.device == "cuda":
                torch.cuda.empty_cache()
            
            return {
                "success": True,
                "content": output['choices'][0]['text'].strip(),
                "raw_response": output,
                "device": self.device
            }
        except Exception as e:
            return {
                "success": False,
                "error": f"LLM Error: {str(e)}",
                "raw_error": str(e),
                "device": self.device
            }

    def get_gpu_info(self) -> Dict[str, Any]:
        """
        Get GPU information and memory usage
        """
        if self.device == "cuda":
            return {
                "gpu_name": torch.cuda.get_device_name(0),
                "memory_allocated": f"{torch.cuda.memory_allocated()/1024**2:.2f} MB",
                "memory_cached": f"{torch.cuda.memory_reserved()/1024**2:.2f} MB",
                "device": self.device
            }
        return {"device": "cpu"}