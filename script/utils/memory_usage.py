import torch
import tracemalloc

def print_gpu_memory_info(output_file, device):

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)
    reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)
    max_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
    max_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)

    with open(output_file, "a") as out:
        out.write(f"[M] - Allocated GPU memory: {allocated:.2f} MB\n")
        out.write(f"[M] - Reserved GPU memory: {reserved:.2f} MB\n")
        out.write(f"[M] - Max GPU allocated memory: {max_allocated:.2f} MB\n")
        out.write(f"[M] - Max GPU reserved memory: {max_reserved:.2f} MB\n")

def print_cpu_memory_info(output_file):
    current, peak = tracemalloc.get_traced_memory()
    current_mb = current / (1024 ** 2)
    peak_mb = peak / (1024 ** 2)

    with open(output_file, "a") as out:
        out.write(f"[M] - Current CPU memory: {current_mb:.2f} MB\n")
        out.write(f"[M] - Max CPU memory: {peak_mb:.2f} MB\n")

def print_memory_info(output_file, device):
    if "cuda" in str(device):
        print_gpu_memory_info(output_file, device)

    print_cpu_memory_info(output_file)

def reset_gpu_memory_info():
    torch.cuda.reset_peak_memory_stats()

def reset_cpu_memory_info():
    tracemalloc.stop()
    tracemalloc.start()

def reset_memory_info(device):
    if "cuda" in str(device):
        reset_gpu_memory_info()

    reset_cpu_memory_info()