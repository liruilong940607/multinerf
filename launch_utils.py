import os
import random
from typing import Any, Callable

import numpy as np
import torch


def _set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _find_free_port():
    import socket

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Binding to port 0 will cause the OS to find an available port for us
    sock.bind(("", 0))
    port = sock.getsockname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by other processes.
    return port


def _distributed_worker(
    local_rank: int,
    world_size: int,
    distributed: bool,
    fn: Callable,
) -> Any:
    print("Distributed worker: %d / %d" % (local_rank + 1, world_size))
    if distributed:
        torch.cuda.set_device(local_rank)
        torch.distributed.init_process_group(
            backend="nccl", world_size=world_size, rank=local_rank
        )
    _set_random_seed(1234 + local_rank)
    output = fn(local_rank)
    if world_size > 1:
        torch.distributed.barrier(device_ids=[local_rank])
        torch.distributed.destroy_process_group()
    print("Job Done for worker: %d / %d" % (local_rank + 1, world_size))
    return output


def launch(fn: Callable) -> float:
    assert torch.cuda.is_available(), "CUDA device is required!"
    world_size = torch.cuda.device_count()
    distributed = world_size > 1
    if distributed:
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = str(_find_free_port())
        world_size = torch.cuda.device_count()
        process_context = torch.multiprocessing.spawn(
            _distributed_worker,
            args=(world_size, distributed, fn),
            nprocs=world_size,
            join=False,
        )
        try:
            process_context.join()
        except KeyboardInterrupt:
            # this is important.
            # if we do not explicitly terminate all launched subprocesses,
            # they would continue living even after this main process ends,
            # eventually making the OD machine unusable!
            for i, process in enumerate(process_context.processes):
                if process.is_alive():
                    print("terminating process " + str(i) + "...")
                    process.terminate()
                process.join()
                print("process " + str(i) + " finished")
        return 1.0
    else:
        return _distributed_worker(
            local_rank=0, world_size=1, distributed=False, fn=fn
        )
