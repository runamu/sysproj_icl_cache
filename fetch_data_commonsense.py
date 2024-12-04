from datasets import load_dataset
import json

dataset = load_dataset("tau/commonsense_qa")

# import ipdb; ipdb.set_trace()

for split in dataset.keys():
    # Save to a JSONL file
    with open(f"dataset/csqa/csqa_{split}.jsonl", "w") as file:
        for item in dataset[split]:
            json.dump(item, file)  # Serialize the dictionary
            file.write("\n")