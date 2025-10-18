import json
import glob
import pandas as pd
from pathlib import Path
import sys
import os

def load_logs(patterns):
    """Load timing logs matching one or more glob patterns."""
    if isinstance(patterns, str):
        patterns = [patterns]

    rows = []
    for pattern in patterns:
        expanded = glob.glob(pattern)
        if not expanded:
            print(f"[WARN] No files matched pattern: {pattern}")
        for path in expanded:
            # Extract rank number from filename (rank0_timing.jsonl → 0)
            rank = None
            stem = Path(path).stem
            if stem.startswith("rank"):
                try:
                    rank = int(stem.split("_")[0].replace("rank", ""))
                except ValueError:
                    pass

            with open(path) as f:
                for line in f:
                    if not line.strip():
                        continue
                    event = json.loads(line)
                    event.setdefault("rank", rank)
                    rows.append(event)

    if not rows:
        raise RuntimeError(
            f"No timing records found. Checked patterns: {patterns}\n"
            f"Working directory: {os.getcwd()}"
        )
    return pd.DataFrame(rows)

if __name__ == "__main__":
    patterns = sys.argv[1:] if len(sys.argv) > 1 else ["rank*_timing.jsonl"]

    df = load_logs(patterns)
    print(f"Loaded {len(df)} timing records from {df['rank'].nunique()} ranks.")
    print()

    # Sanity check for required columns
    for col in ["rank", "stage", "microbatch", "phase", "total_ms"]:
        if col not in df.columns:
            raise KeyError(f"Missing expected column '{col}' in timing logs")

    # --- Collapse duplicates by mean per (rank, stage, microbatch, phase) ---
    df = (
        df.groupby(["rank", "stage", "microbatch", "phase"], as_index=False)
        .agg(total_ms=("total_ms", "mean"), count=("total_ms", "size"))
    )
    df = df.sort_values(["rank", "stage", "microbatch", "phase"]).reset_index(drop=True)

    print("=== Per-Microbatch Stage Timing (averaged across repeats) ===")
    print(df[["rank", "stage", "microbatch", "phase", "total_ms"]].to_string(index=False))
    df.to_csv("timing_microbatch_breakdown.csv", index=False)
    print("\nWrote timing_microbatch_breakdown.csv")

    # --- Mean phase time per stage (across all microbatches) ---
    summary = (
        df.groupby(["rank", "stage", "phase"])
        .agg(mean_total_ms=("total_ms", "mean"),
             std_total_ms=("total_ms", "std"),
             count=("microbatch", "count"))
        .reset_index()
        .sort_values(["rank", "stage", "phase"])
    )

    print("\n=== Mean Phase Time Per Stage ===")
    print(summary.to_string(index=False))
    summary.to_csv("timing_stage_means.csv", index=False)
    print("\nWrote timing_stage_means.csv")
