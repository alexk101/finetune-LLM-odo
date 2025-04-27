import os
import torch
import torch.distributed as dist
import datetime

def init_process_group():
    world_size = int(os.getenv("WORLD_SIZE", 1))
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    
    if os.getenv("DEBUG") == "true":
        print(f"WORLD_SIZE: {world_size}")
        print(f"RANK: {rank}")
        print(f"LOCAL_RANK: {local_rank}")

        print(f"ROCR_VISIBLE_DEVICES: {os.getenv('ROCR_VISIBLE_DEVICES')}")
        print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        print(f"torch.cuda.current_device(): {torch.cuda.current_device()}")
        print(f"torch.cuda.device_count(): {torch.cuda.device_count()}")

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=world_size,
            rank=rank,
            timeout=datetime.timedelta(minutes=5)  # Increased timeout for large-scale runs
        )
        print(f"Rank {rank}: Process group initialized successfully")

    return rank, world_size, local_rank
