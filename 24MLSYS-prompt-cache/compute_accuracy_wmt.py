import re
import ast
import json
from json_repair import repair_json

from metrics import (
    # qa_f1_score,
    # rouge_zh_score,
    # qa_f1_zh_score,
    rouge_score,
    # classification_score,
    # retrieval_score,
    # retrieval_zh_score,
    # count_score,
    # code_sim_score
)


# result_path = "outputs/results_yx/wmt_no_cache_llama2.json"
result_path = "outputs/results_yx/wmt_with_cache_llama2.json"

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

def extract_last_French(response):
    # Pattern to match JSON-like objects
    json_pattern = re.compile(r'\{.*?\}')
    matches = json_pattern.findall(response)

    if matches:
        # Take the last JSON-like object
        last_json = matches[-1]
        # Normalize the JSON to ensure it can be parsed correctly
        normalized_json = last_json.replace("'", "\"").replace(" ", "").replace("French", "\"French\"")
        try:
            # Parse the JSON and extract the French
            parsed_json = json.loads(normalized_json)
            return parsed_json.get("French")
        except json.JSONDecodeError:
            return None
    return None

# parse the response
scores = []
for result in results:
    answer = result['answers']
    try:
        parsed_response = ast.literal_eval(repair_json(result['response']))
        prediction = parsed_response['French']
        # import ipdb; ipdb.set_trace()
        rouge_result = rouge_score(answer, prediction)
        scores.append(rouge_result)
    except:
        rouge_result = 0.0
        # continue
    # import ipdb; ipdb.set_trace()

    if len(scores) >= 90:
        break

mean_scores = sum(scores) / len(scores)
print("Result path:", result_path)
print(f"Average score for {len(scores)} samples: {mean_scores}")


