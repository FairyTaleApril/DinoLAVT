import random
import torch
import os
import numpy as np

from utils.logger import info

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def print_gpu_info():
    info(f"cuda version: {torch.version.cuda}")
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        info(f"gpu: {gpu_name}")

        device = torch.device('cuda:0')
        total_memory = torch.cuda.get_device_properties(device).total_memory

        reserved_memory = torch.cuda.memory_reserved(device)
        allocated_memory = torch.cuda.memory_allocated(device)

        free_memory = reserved_memory - allocated_memory

        unreserved_memory = total_memory - reserved_memory
        info(f"Total Memory: {total_memory / (1024 ** 3):.2f} GB")
        info(f"Reserved Memory: {reserved_memory / (1024 ** 3):.2f} GB")
        info(f"Allocated Memory: {allocated_memory / (1024 ** 3):.2f} GB")
        info(f"Free Memory (within reserved): {free_memory / (1024 ** 3):.2f} GB")
        info(f"Unreserved Memory: {unreserved_memory / (1024 ** 3):.2f} GB")
    else:
        info("No GPU available.")