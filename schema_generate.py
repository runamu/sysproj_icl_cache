import json
import os

# Input file and output file paths
input_file = "dataset/gsm8k/gsm8k.jsonl"
output_file = "math.xml"

def read_jsonl(file_path, limit=10):
    """Reads a JSONL file and fetches a limited number of entries."""
    data = []
    with open(file_path, "r") as f:
        for idx, line in enumerate(f):
            if idx >= limit:
                break
            data.append(json.loads(line))
    return data

def generate_pml_schema(schema_name, modules):
    """Generates a Prompt Markup Language (PML) schema with the given modules."""
    pml_template = f"""<schema name="{schema_name}">
  <system>
    You are a sophisticated language model assistant designed to read and understand the given math problems and their answer examples.
    Your current task is to analyze the provided math problem and generate a correct and well-explained answer based on the examples given.
    Ensure your responses are accurate, clear, and concise.
  </system>
  <user>
    Let's think step by step.
  </user>
"""
    for i, module in enumerate(modules):
        question = module.get("question", "N/A").replace('"', '&quot;')
        answer = module.get("answer", "N/A").replace('"', '&quot;')
        pml_template += f"""
  <module name="module_{i + 1}">
     Q:{question}
     A:{answer}.
  </module>
"""
    pml_template += "</schema>"
    return pml_template

def save_pml(file_path, content):
    """Saves the PML content to a file."""
    with open(file_path, "w") as f:
        f.write(content)

def main():
    # Ensure the input file exists
    if not os.path.exists(input_file):
        print(f"Input file does not exist: {input_file}")
        return

    # Fetch data
    print("Reading JSONL data...")
    data = read_jsonl(input_file)

    # Generate PML schema
    print("Generating PML schema...")
    pml_content = generate_pml_schema("math", data)

    # Save PML schema to file
    print(f"Saving PML schema to {output_file}...")
    save_pml(output_file, pml_content)
    print("Done.")

if __name__ == "__main__":
    main()
