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

def computeIoU(pred_seg, gd_seg):
    I = np.sum(np.logical_and(pred_seg, gd_seg))
    U = np.sum(np.logical_or(pred_seg, gd_seg))

    return I, U

def overlay_davis(image, mask, colors=[[0, 0, 0], [255, 0, 0]], cscale=1, alpha=0.4):
    from scipy.ndimage.morphology import binary_dilation

    colors = np.reshape(colors, (-1, 3))
    colors = np.atleast_2d(colors) * cscale

    im_overlay = image.copy()
    object_ids = np.unique(mask)

    for object_id in object_ids[1:]:
        # Overlay color on  binary mask
        foreground = image*alpha + np.ones(image.shape)*(1-alpha) * np.array(colors[object_id])
        binary_mask = mask == object_id

        # Compose image
        im_overlay[binary_mask] = foreground[binary_mask]

        # countours = skimage.morphology.binary.binary_dilation(binary_mask) - binary_mask
        countours = binary_dilation(binary_mask) ^ binary_mask
        # countours = cv2.dilate(binary_mask, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))) - binary_mask
        im_overlay[countours, :] = 0

    return im_overlay.astype(image.dtype)