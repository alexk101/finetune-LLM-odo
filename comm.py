import os
import torch.distributed as dist
import datetime

def init_process_group():
    world_size = int(os.getenv("WORLD_SIZE", 1))

    if world_size > 1:
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            timeout=datetime.timedelta(minutes=5)  # Increased timeout for large-scale runs
        )
