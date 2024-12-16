from datasets import load_dataset
import json

dataset = load_dataset("Nicolas-BZRD/English_French_Webpages_Scraped_Translated")

# import ipdb; ipdb.set_trace()

cnt = 0
for split in dataset.keys():
    # Save to a JSONL file
    with open(f"dataset/wmt/wmt_{split}.jsonl", "w") as file:
        for item in dataset[split]:
            json.dump(item, file)  # Serialize the dictionary
            file.write("\n")
            cnt += 1

            if cnt >= 10000:
                break