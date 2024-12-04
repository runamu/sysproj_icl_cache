import re

# log_file_path = "/home/yedasong/sysproj_base/sysproj_icl_cache/24MLSYS-prompt-cache/outputs/riddle_no_cache_llama2.log"
log_file_path = "/home/yedasong/sysproj_base/sysproj_icl_cache/24MLSYS-prompt-cache/outputs/riddle_with_cache_llama2.log"

# Sample log data
with open(log_file_path, "r") as file:
    log_data = file.read()

# Regular expression to capture "Action Acc" and "Task"
latency_pattern = r"Prefill latency: ([\d\.]+)"

# Parse all "Action Acc" values and task names preserving order
latency_matches = re.findall(latency_pattern, log_data)

# Average latency
latency_avg = sum(map(float, latency_matches)) / len(latency_matches)

# Print results
print(f"Average Latency for {len(latency_matches)} samples: {latency_avg} ms")
