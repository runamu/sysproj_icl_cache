import json
import re

def calculate_module_lengths(prompt):
    # Split the prompt into modules and system/user sections
    modules = {}

    # Find and calculate module lengths
    module_pattern = r'<module name="(.*?)">(.*?)</module>'
    for match in re.finditer(module_pattern, prompt, re.DOTALL):
        module_name = match.group(1)
        module_text = match.group(2).strip()
        modules[module_name] = len(module_text)
    # Calculate total length and average
    total_length = sum(modules.values())
    average_length = total_length / len(modules)

    # Calculate system section length
    system_start = prompt.find('<system>')
    system_end = prompt.find('</system>')
    system_text = prompt[system_start+8:system_end].strip()
    modules['system'] = len(system_text)

    # Calculate user section length
    user_start = prompt.find('<user>')
    user_end = prompt.find('</user>')
    user_text = prompt[user_start+6:user_end].strip()
    modules['user'] = len(user_text)

    # import ipdb; ipdb.set_trace()
    return modules, total_length, average_length

# Prompt to analyze
# data_path = "dataset/csqa/csqa_train.jsonl"
# data_path = "dataset/math/gsm8k.jsonl"
# data_path = "dataset/riddlesense/rs_train.jsonl"
# data_path = "dataset/sst2/sst2_train.jsonl"
data_paths = [
    "math.xml",
    "csqa.xml",
    "riddle_newprompt.xml",
    "sst2.xml",
]

for data_path in data_paths:
    with open(data_path, 'r') as f:
        prompt = f.read()

    # Calculate lengths
    module_lengths, total_length, average_length = calculate_module_lengths(prompt)

    # Print results
    print(data_path)
    print("Module Lengths:")
    for module, length in module_lengths.items():
        print(f"{module}: {length} characters")

    print(f"Total Length: {total_length} characters")
    print(f"Average Module Length: {average_length:.2f} characters")
    print()
