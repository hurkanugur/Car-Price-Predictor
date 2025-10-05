import torch

class DeviceManager:
    def __init__(self):
        self.device = self._select_device()

    def _select_device(self):
        """
        Selects the best available device: CUDA > MPS > CPU.
        """
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")

        print(f"• Selected device: {device}")
        return device
    
    def release_memory(self):
        """
        Releases GPU or MPS memory to help avoid out-of-memory issues.
        """
        try:
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
                print("🧹 Released CUDA memory cache.")
            elif self.device.type == "mps":
                torch.mps.empty_cache()
                print("🧹 Released MPS memory cache.")
            else:
                print("ℹ️ No GPU memory to release (CPU device).")
                
        except Exception as e:
            print(f"⚠️ Failed to release GPU memory: {e}")
