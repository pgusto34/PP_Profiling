import os
import torch
import argparse
import torch.distributed as dist
import torch.nn as nn
import subprocess
from torch.distributed.pipelining import pipeline, SplitPoint, ScheduleInterleavedZeroBubble
from torch.profiler import profile, ProfilerActivity, record_function, schedule
from model import Transformer, LLAMA_DEBUG, LLAMA_3B, LLAMA_1B

def wrap_stage_forward(stage, name):
    orig_forward = stage.submod.forward
    mb_counter = {"i": 0}

    def wrapped_forward(self, *args, **kwargs):
        i = mb_counter["i"]
        mb_counter["i"] += 1

        with record_function(f"{name}_F_mb{i}"):
            return orig_forward(self, *args, **kwargs)

    stage.submod.forward = wrapped_forward

def wrap_stage_backward(stage, name):
    orig_bwd_one = stage.__class__.backward_one_chunk
    def wrapped_bwd_one(self, *args, **kwargs):
        mb = kwargs.get("bwd_chunk_id", args[0] if args else "unk")
        with record_function(f"{name}_B_mb{mb}"):
            return orig_bwd_one(self, *args, **kwargs)
    stage.backward_one_chunk = wrapped_bwd_one.__get__(stage, stage.__class__)

    orig_bwd_w = stage.__class__.backward_weight_one_chunk
    def wrapped_bwd_w(self, *args, **kwargs):
       mb = kwargs.get("bwd_chunk_id", args[0] if args else "unk")
       with record_function(f"{name}_W_mb{mb}"):
            return orig_bwd_w(self, *args, **kwargs)
    stage.backward_weight_one_chunk = wrapped_bwd_w.__get__(stage, stage.__class__)


def run_pipeline(model_args, cmd_args):
   global world_size, rank, device
   microbatch_size = cmd_args.microbatch_size
   num_microbatches = cmd_args.num_microbatches

   # Initialize profiler activity based on device availability
   activities = [ProfilerActivity.CPU]
   if torch.cuda.is_available():
      device = "cuda"
      activities += [ProfilerActivity.CUDA]
   elif torch.xpu.is_available():
      device = "xpu"
      activities += [ProfilerActivity.XPU]
   
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
   
   stage = pipe.build_stage(rank, device=device) # Currently assumes one stage per rank\

   wrap_stage_forward(stage, f"stage{rank}")
   wrap_stage_backward(stage, f"stage{rank}")

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

   print(f"\n=== Pipeline order for rank {rank} ===")
   for step, action in enumerate(schedule.pipeline_order[rank]):
       if action is None:
           continue
       stage_idx, op, mb = action
       print(f"t={step:03d} | stage={stage_idx} | op={op} | microbatch={mb}")

   with profile(activities=activities, record_shapes=True) as prof:
      if rank == 0:
         schedule.step(x)
      elif rank == 1:
         losses = []
         output = schedule.step(target=y, losses=losses)
         print(f"final losses: {losses}")

   trace_dir = os.path.join(os.path.dirname(__file__), "traces")
   out_path = os.path.join(trace_dir, f"trace_{rank}.json")
   prof.export_chrome_trace(out_path)
   
   dist.barrier()
   if rank == 1:
      print(prof.key_averages().table())

      # Create traces folder if it doesn't exist
      os.makedirs(trace_dir, exist_ok=True)

      merged_out = os.path.join(trace_dir, "merged_trace.json")
      cmd = ["python", "filter_trace.py", os.path.join(trace_dir,"trace_0.json"), os.path.join(trace_dir,"user_only_trace_0.json")]
      subprocess.run(cmd, check=True)

      cmd = ["python", "filter_trace.py", os.path.join(trace_dir, "trace_1.json"),  os.path.join(trace_dir,"user_only_trace_1.json")]
      subprocess.run(cmd, check=True)

      cmd = ["python", "merge_traces.py", os.path.join(trace_dir,"user_only_trace_0.json"), os.path.join(trace_dir,"user_only_trace_1.json"), merged_out]
      subprocess.run(cmd, check=True)
      print(f"✅ Merged trace available at {merged_out}")

if __name__ == "__main__":
   # Parse command line arguments
   # May add more arguments (e.g., pipeline schedule or model config) in the future
   parser = argparse.ArgumentParser()
   parser.add_argument('--cuda', type=int, default=int(torch.cuda.is_available()))
   parser.add_argument('--microbatch_size', type=int, default=64)
   parser.add_argument('--num_microbatches', type=int, default=256)
   # parser.add_argument('--use_profiler', action="store_true")
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
   model_args = LLAMA_DEBUG

   # Run Pipeline Schedule
   run_pipeline(model_args, cmd_args)
   dist.destroy_process_group()
