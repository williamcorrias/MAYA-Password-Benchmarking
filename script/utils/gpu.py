import subprocess
import time
import os

from filelock import FileLock

LOCK_PATH = "/tmp/gpu_selection.lock"

def select_gpu():
    while True:
        try:
            with FileLock(LOCK_PATH, timeout=0.1):
                chosen_gpu = get_optimal_gpu()
                if chosen_gpu is not None:
                    return chosen_gpu
                else:
                    time.sleep(1)
        except TimeoutError:
            time.sleep(10)


def get_visible_gpus():
    visible_gpus = os.getenv("CUDA_VISIBLE_DEVICES")
    if visible_gpus is not None:
        visible_gpus = visible_gpus.split(",")
    return visible_gpus

def get_optimal_gpu():
    visible_gpus = get_visible_gpus()

    result = subprocess.run(
        ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.free,memory.total,', '--format=csv,noheader,nounits'],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True)

    gpus_info = result.stdout.strip().split('\n')

    gpu_memory = []
    for gpu in gpus_info:
        id, gpu_utilization, memory_free, memory_total = map(int, gpu.split(','))

        if visible_gpus is not None:
            if str(id) not in visible_gpus:
                continue
            else:
                id = visible_gpus.index(str(id))

        percentage_free_memory = round((memory_free / memory_total * 100), 2)
        percentage_free_gpu = 100 - gpu_utilization

        if percentage_free_memory < 25 or percentage_free_gpu < 25:
            continue

        priority_value = (percentage_free_memory * percentage_free_gpu) / 2
        gpu_memory.append((id, priority_value))

    gpu_memory = sorted(gpu_memory, key=lambda x: x[1], reverse=True)
    return gpu_memory[0][0] if gpu_memory else None