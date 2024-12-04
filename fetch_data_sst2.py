from datasets import load_dataset
import json

dataset = load_dataset("stanfordnlp/sst2")

# import ipdb; ipdb.set_trace()

for split in dataset.keys():
    # Save to a JSONL file
    with open(f"dataset/sst2/sst2_{split}.jsonl", "w") as file:
        for item in dataset[split]:
            json.dump(item, file)  # Serialize the dictionary
            file.write("\n")