import json
import sys

# GPT-generated script to filter out non-user events from profiling traces
def filter_user_annotations(input_path, output_path):
    with open(input_path, "r") as f:
        trace = json.load(f)

    # Keep only events that are user annotations
    user_events = [
        ev for ev in trace.get("traceEvents", [])
        if ev.get("cat") == "user_annotation"
    ]

    # Preserve important metadata if present
    filtered_trace = {
        "schemaVersion": trace.get("schemaVersion", 1),
        "displayTimeUnit": trace.get("displayTimeUnit", "ms"),
        "baseTimeNanoseconds": trace.get("baseTimeNanoseconds", 0),
        "traceEvents": user_events,
    }

    # Keep device and distributed info if available
    if "deviceProperties" in trace:
        filtered_trace["deviceProperties"] = trace["deviceProperties"]
    if "distributedInfo" in trace:
        filtered_trace["distributedInfo"] = trace["distributedInfo"]

    with open(output_path, "w") as f:
        json.dump(filtered_trace, f, indent=2)

    print(f"✅ Wrote {len(user_events)} user annotations + metadata to {output_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python filter_trace.py input_trace.json output_trace.json")
        sys.exit(1)

    filter_user_annotations(sys.argv[1], sys.argv[2])
