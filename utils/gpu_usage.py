from pynvml import *
from hurry.filesize import size
# import torch

def print_gpu_info(gpu_idx):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(gpu_idx)
    info = nvmlDeviceGetMemoryInfo(handle)
    print("{} Total memory: {}={} || Free memory: {}={} || Used memory: {}={} ||\n".format(gpu_idx, info.total, size(info.total), info.free, size(info.free), info.used, size(info.used)))

def main():
    num_gpus = 8 #torch.cuda.device_count()
    for i in range(num_gpus):
        print_gpu_info(i)

if __name__ == "__main__":
    main()