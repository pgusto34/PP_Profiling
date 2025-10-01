import json, sys

META_NAMES = {"process_name", "thread_name", "process_sort_index"}

# GPT-generated script to merge traces from two ranks
def merge_traces(input_files, output_file):
    merged = {"traceEvents": [], "displayTimeUnit": "ns"}
    base_pid = 1000

    for idx, path in enumerate(input_files):
        with open(path, "r") as f:
            data = json.load(f)

        events = data.get("traceEvents", data)
        pid = base_pid + idx
        cleaned = []
        min_ts = None

        for ev in events:
            if not isinstance(ev, dict):
                continue

            # normalize ts / dur to numbers
            if "ts" in ev and isinstance(ev["ts"], str):
                try:
                    ev["ts"] = float(ev["ts"])
                except Exception:
                    continue
            if "dur" in ev and isinstance(ev["dur"], str):
                try:
                    ev["dur"] = float(ev["dur"])
                except Exception:
                    ev.pop("dur", None)

            if isinstance(ev.get("ts"), (int, float)):
                min_ts = ev["ts"] if min_ts is None else min(min_ts, ev["ts"])

            # drop incoming metadata to avoid pid/name collisions
            if ev.get("ph") == "M" and ev.get("name") in META_NAMES:
                continue

            # remap pid, normalize tid
            ev["pid"] = pid
            if not isinstance(ev.get("tid"), (int, float)):
                ev["tid"] = 0

            cleaned.append(ev)

        meta_ts = (min_ts - 1.0) if min_ts is not None else 0.0
        # inject rank labels + ordering
        merged["traceEvents"].extend([
            {"ph":"M","pid":pid,"tid":0,"ts":meta_ts,"name":"process_name","args":{"name": f"rank{idx}"}},
            {"ph":"M","pid":pid,"tid":0,"ts":meta_ts,"name":"process_sort_index","args":{"sort_index": idx}},
            {"ph":"M","pid":pid,"tid":0,"ts":meta_ts,"name":"thread_name","args":{"name":"main"}},
        ])

        merged["traceEvents"].extend(cleaned)

    with open(output_file, "w") as f:
        json.dump(merged, f)

    print(f"Merged trace written to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python merge_traces.py <in1.json> <in2.json> ... <out.json>")
        sys.exit(1)
    *inputs, output = sys.argv[1:]
    merge_traces(inputs, output)
