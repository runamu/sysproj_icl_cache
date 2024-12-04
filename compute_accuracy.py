import re
import ast
import json
from json_repair import repair_json

# result_path = "24MLSYS-prompt-cache/outputs/results/sst2_no_cache_llama2.json"
result_path = "24MLSYS-prompt-cache/outputs/results/sst2_with_cache_llama2.json"

def read_jsonl(file_path):
    """Reads a JSONL file and fetches a limited number of entries."""
    data = []
    with open(file_path, "r") as f:
        for idx, line in enumerate(f):
            # if idx >= limit:
            #     break
            data.append(json.loads(line))
    return data

results = read_jsonl(result_path)

def extract_last_label(response):
    # Pattern to match JSON-like objects
    json_pattern = re.compile(r'\{.*?\}')
    matches = json_pattern.findall(response)

    if matches:
        # Take the last JSON-like object
        last_json = matches[-1]
        # Normalize the JSON to ensure it can be parsed correctly
        normalized_json = last_json.replace("'", "\"").replace(" ", "").replace("label", "\"label\"")
        try:
            # Parse the JSON and extract the label
            parsed_json = json.loads(normalized_json)
            return parsed_json.get("label")
        except json.JSONDecodeError:
            return None
    return None

# parse the response
accuracies = []
for result in results:
    parsed_response = ast.literal_eval(repair_json(result['response']))
    answer = result['answers']
    try:
        prediction = parsed_response['label']
        # TODO: just use a pattern matching - use the last json if there are multiple jsons
        # prediction = extract_last_label(result['response'])
        correct = answer == prediction
    except:
        correct = False
        # continue
    # import ipdb; ipdb.set_trace()
    accuracies.append(correct)


accuracy = sum(accuracies) / len(accuracies)
print(f"Accuracy for {len(accuracies)} samples: {accuracy}")


