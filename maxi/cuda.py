try:
    import torch
    print("PyTorch is installed.")

    if torch.cuda.is_available():
        print("CUDA is available.")
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            print(f" - GPU {i}: {gpu_name}")
    else:
        print("CUDA is not available.")
except ImportError:
    print("PyTorch is not installed.")