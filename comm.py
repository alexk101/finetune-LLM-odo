import os
import torch
import torch.distributed as dist
import datetime

def init_process_group():
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print(f"WORLD_SIZE: {world_size}")
    print(f"RANK: {os.getenv('RANK')}")
    print(f"LOCAL_RANK: {os.getenv('LOCAL_RANK')}")
    print(f"ROCR_VISIBLE_DEVICES: {os.getenv('ROCR_VISIBLE_DEVICES')}")

    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
    print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=5)  # Increased timeout for large-scale runs
        )
