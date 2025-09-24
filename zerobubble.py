import os
import torch
import argparse
import time
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleInterleavedZeroBubble
from model import Transformer, LLAMA_3B

def run_pipeline(model_args, cmd_args):
   global world_size, rank, device
   microbatch_size = cmd_args.microbatch_size
   num_microbatches = cmd_args.num_microbatches
   
   model = Transformer(model_args)
   model.to(device)

   # Partition the model into two stages and create the stage for this rank
   layers_per_rank = model_args.n_layers // world_size

   split_spec = {
      f"layers.{i * layers_per_rank}": SplitPoint.BEGINNING
      for i in range(1, world_size)
   }

   example_input = torch.randint(
      low=0,
      high=model_args.vocab_size,
      size=(microbatch_size, model_args.max_seq_len),
      dtype=torch.long,
      device=device
   )
   
   pipe = pipeline(
      module=model,
      mb_args=(example_input,),
      split_spec=split_spec,
    )
   
   stage = pipe.build_stage(rank, device=device) # Currently assumes one stage per rank
   

   # Helper loss function that automatically reshapes output and targets tensors
   def tokenwise_loss_fn(outputs, targets):
      loss_fn = nn.CrossEntropyLoss()
      outputs = outputs.reshape(-1, model_args.vocab_size)
      targets = targets.reshape(-1)
      return loss_fn(outputs, targets)

   schedule = ScheduleInterleavedZeroBubble([stage], n_microbatches=num_microbatches, loss_fn=tokenwise_loss_fn)

   # Generate random training tensors
   x = torch.randint(
      low=0,
      high=model_args.vocab_size,
      size=(microbatch_size * num_microbatches, model_args.max_seq_len),
      dtype=torch.long,
      device=device                 
   )

   y = torch.randint(
      low=0, 
      high=model_args.vocab_size, 
      size=(microbatch_size * num_microbatches, model_args.max_seq_len), 
      dtype=torch.long,
      device=device
   )

   # Perform schedule step and compute throughput
   total_tokens_per_step = microbatch_size * num_microbatches * model_args.max_seq_len
   step_durations = []

   for step in range(cmd_args.num_steps):
      dist.barrier()
      start_time = time.perf_counter()

      if rank == 0:
         schedule.step(x)
      elif rank == 1:
         losses = []
         output = schedule.step(target=y, losses=losses)
         if step == cmd_args.num_steps - 1:
            print(f"final losses: {losses}")

      dist.barrier()
      end_time = time.perf_counter()
      step_durations.append(end_time - start_time)

   if rank == 1:
      avg_time = sum(step_durations) / len(step_durations)
      throughput = total_tokens_per_step / avg_time
      print(f"Average Throughput Over {cmd_args.num_steps} steps: {throughput:.2f} tokens/second")

   dist.destroy_process_group()

if __name__ == "__main__":
   # Parse command line arguments
   # May add more arguments (e.g., pipeline schedule or model config) in the future
   parser = argparse.ArgumentParser()
   parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
   parser.add_argument('--microbatch_size', type=int, default=64)
   parser.add_argument('--num_microbatches', type=int, default=256)
   parser.add_argument('--use_profiler', action="store_true")
   parser.add_argument('--num_steps', type=int, default=5)
   cmd_args = parser.parse_args()

   # Perform torch.distributed and device initialization
   global device, rank, world_size
   rank = int(os.environ["LOCAL_RANK"])
   world_size = int(os.environ["WORLD_SIZE"])
   device = torch.device(f"cuda:{rank}") if cmd_args.cuda else torch.device("cpu")
   backend = "nccl" if cmd_args.cuda else "gloo"
   dist.init_process_group(backend=backend, rank=rank, world_size=world_size)

   # Load model config
   model_args = LLAMA_3B

   # Run Pipeline Schedule
   run_pipeline(model_args, cmd_args)
