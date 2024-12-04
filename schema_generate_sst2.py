import json
import os

# Input file and output file paths
input_file = "dataset/sst2/sst2_train.jsonl"
output_file = "sst2.xml"

def format_sst2(sst2_data, with_answer=True):
    # Create the formatted output using an f-string
    if with_answer:
        output = f"""Sentence: {sst2_data['sentence']}
    Label: {sst2_data['label']}"""
# Label: {'Negative' if sst2_data['label'] == 0 else 'Positive'}"""

    else:
        output = f"""Sentence: {sst2_data['sentence']}
    Label: """

    return output

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
        #### Task Description ####
        You are tasked with binary sentiment analysis. Analyze the provided examples carefully to understand the task. Each example consists of a sentence and its corresponding sentiment label. The label `1` represents positive sentiment, and the label `0` represents negative sentiment. After reviewing the examples, classify the LAST sentence based on its sentiment. Respond with a single word, either `0` or `1`.

        #### Response Format ####
        Respond in a JSON format with the following structure:
        {{"label": sentiment_label}}]
        Do NOT provide any explanations or additional text. Do NOT include the labels of the examples in your response.

        #### Examples ####
        Here are a few example interactions:
    </system>
    <user>
    </user>
"""
    for i, module in enumerate(modules):
        pml_template += f"""
  <module name="module_{i + 1}">
    {format_sst2(module)}
  </module>
"""
        # # Example module:
        # Question: A man is incarcerated in prison, and as his punishment he has to carry a one tonne bag of sand backwards and forwards across a field the size of a football pitch.  What is the one thing he can put in it to make it lighter?
        # Choices: (A) throw (B) bit (C) gallon (D) mouse (E) hole
        # Output: hole

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
    pml_content = generate_pml_schema("sst2", data)

    # Save PML schema to file
    print(f"Saving PML schema to {output_file}...")
    save_pml(output_file, pml_content)
    print("Done.")

if __name__ == "__main__":
    main()
