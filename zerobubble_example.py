import os
import torch
import torch.multiprocessing as mp
import argparse
import torch.distributed as dist
import torch.nn as nn
import time
import subprocess
from torch.cuda import Event
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleInterleavedZeroBubble, Schedule1F1B, ScheduleInterleaved1F1B
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from model import Transformer, LLAMA_DEBUG, LLAMA_3B, LLAMA_1B

def setup(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"

    backend = "nccl"
    dist.init_process_group(backend=backend, rank=rank, world_size=world_size)


def cleanup():
    dist.destroy_process_group()


def run_pipeline(rank, world_size, mb_size, num_mbs, seq_len, model_cfg=LLAMA_DEBUG):
    setup(rank, world_size)
    dist.barrier()
    device = torch.device(f"cuda:{rank}")
    print(f"[Rank {rank}] using torch device {device}")

    model = Transformer(model_cfg)
    model.to(device)

    # Helper loss function that automatically reshapes output and targets tensors
    def tokenwise_loss_fn(outputs, targets):
        loss_fn = nn.CrossEntropyLoss()
        outputs = outputs.reshape(-1, model_cfg.vocab_size)
        targets = targets.reshape(-1)
        return loss_fn(outputs, targets)

    # Partition the model into world_size stages and create the stage for this rank
    layers_per_rank = model_cfg.n_layers // world_size

    split_spec = {
        f"layers.{i * layers_per_rank}": SplitPoint.BEGINNING
        for i in range(1, world_size)
    }

    example_input = torch.randint(
        low=0,
        high=model_cfg.vocab_size,
        size=(mb_size, seq_len),
        dtype=torch.long,
        device=device
    )
    
    pipe = pipeline(
        module=model,
        mb_args=(example_input,),
        split_spec=split_spec,
    )
    
    stage = pipe.build_stage(rank, device=device) # Currently assumes one stage per rank  
    schedule = ScheduleInterleavedZeroBubble([stage], n_microbatches=num_mbs, loss_fn=tokenwise_loss_fn)

    # Print the pipeline schedule
    print(f"\n=== Pipeline order for rank {rank} ===")
    for step, action in enumerate(schedule.pipeline_order[rank]):
        if action is None:
            continue
        stage_idx, op, mb = action
        print(f"t={step:03d} | stage={stage_idx} | op={op} | microbatch={mb}")

    # Generate random training tensors
    x = torch.randint(
        low=0,
        high=model_cfg.vocab_size,
        size=(mb_size * num_mbs, seq_len),
        dtype=torch.long,
        device=device                 
    )

    y = torch.randint(
        low=0, 
        high=model_cfg.vocab_size, 
        size=(mb_size * num_mbs, seq_len), 
        dtype=torch.long,
        device=device
    )

    total_tokens = mb_size * num_mbs * seq_len

    warmup_steps = 2
    measure_steps = 4

    for _ in range(warmup_steps):
        if rank != world_size - 1:
            schedule.step(x)
        else:
            _losses = []
            _ = schedule.step(target=y, losses=_losses)
        torch.cuda.synchronize()
        dist.barrier()

    elapsed_times = []

    for _ in range(measure_steps):
        dist.barrier()
        torch.cuda.synchronize()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        
        start.record()

        if rank != world_size - 1:
            schedule.step(x)
        else:
            losses = []
            _ = schedule.step(target=y, losses=losses)

        end.record()
        torch.cuda.synchronize()
        dist.barrier()

        if rank == world_size - 1:
            elapsed_ms = start.elapsed_time(end)
            elapsed_times.append(elapsed_ms / 1000.0)  

    if rank == world_size - 1:
        avg_elapsed = sum(elapsed_times) / len(elapsed_times)
        tokens_per_sec = total_tokens / avg_elapsed
        print(f"[Rank {rank}] Avg elapsed: {avg_elapsed:.6f}s (CUDA events)")
        print(f"[Rank {rank}] Throughput: {tokens_per_sec:.2f} tokens/s")
        print(f"[Rank {rank}] final losses: {losses}")

    dist.barrier()
    cleanup()

   

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--microbatch_size', type=int, default=8)
    parser.add_argument('--num_microbatches', type=int, default=4)
    parser.add_argument('--seq_len', type=int, default=512)
    args = parser.parse_args()
    
    world_size = 2

    mp.set_start_method("spawn", force=True)
    manager = mp.Manager()

    mp.spawn(
        run_pipeline,
        args=(world_size, args.microbatch_size, args.num_microbatches, args.seq_len, LLAMA_3B),
        nprocs=world_size,
        join=True,
    )
