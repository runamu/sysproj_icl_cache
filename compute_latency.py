import re

# log_file_path = "24MLSYS-prompt-cache/outputs/riddle_no_cache_llama2.log"
log_file_path = "24MLSYS-prompt-cache/outputs/riddle_with_cache_llama2.log"

# log_file_path = "24MLSYS-prompt-cache/outputs/sst2_no_cache_llama2.log"
# log_file_path = "24MLSYS-prompt-cache/outputs/sst2_with_cache_llama2.log"


# Sample log data
with open(log_file_path, "r") as file:
    log_data = file.read()

# Regular expression to capture "Action Acc" and "Task"
latency_pattern = r"Prefill latency: ([\d\.]+)"
response_pattern = r"Assistant: Prefill latency: [\d\.]+ ms\n(.*?)\nprompt_text:"

# Parse all "Action Acc" values and task names preserving order
latency_matches = re.findall(latency_pattern, log_data)
response_matches = re.findall(response_pattern, log_data, re.DOTALL)

# Average latency
latency_avg = sum(map(float, latency_matches)) / len(latency_matches)

# Print results
print(f"Average Latency for {len(latency_matches)} samples: {latency_avg} ms")

# # Combine results
# parsed_data = list(zip(task_matches, response_matches))

# # Print results
# print(f"Total Action Acc: {action_acc_matches[-1]}")
# for task, action_acc in parsed_data:
#     print(f"Task: {task}, Action Acc: {action_acc}")
