import json
import os

# Input file and output file paths
input_file = "dataset/riddlesense/rs_train.jsonl"
output_file = "riddle.xml"

def format_riddle(riddle_data, with_answer=True):
    # Extract question stem and choices from the dictionary
    stem = riddle_data['question']['stem']
    choices = riddle_data['question']['choices']
    answer_key = riddle_data['answerKey']

    # Find the text of the choice corresponding to the answer key
    answer_text = next(choice['text'] for choice in choices if choice['label'] == answer_key)

    # Create the formatted output using an f-string
    if with_answer:
        output = f"""Question: {stem}
Choices: {' '.join(f"({choice['label']}) {choice['text']}" for choice in choices)}
Output: {answer_text}"""
    else:
        output = f"""Question: {stem}
Choices: {' '.join(f"({choice['label']}) {choice['text']}" for choice in choices)}
Output: """

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
        You are a sophisticated language model assistant designed to solve riddles by understanding their context, linguistic nuances, and implied meanings.
        Your current task is to analyze the provided riddle and select the most appropriate answer by carefully examining the riddle's descriptive clues and logical structure.
        Ensure your reasoning captures the metaphorical and lateral thinking required to solve the riddle correctly.
    </system>
    <user>
        Let's think step by step.
    </user>
"""
    for i, module in enumerate(modules):
        pml_template += f"""
  <module name="module_{i + 1}">
     {format_riddle(module)}
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
    pml_content = generate_pml_schema("riddle", data)

    # Save PML schema to file
    print(f"Saving PML schema to {output_file}...")
    save_pml(output_file, pml_content)
    print("Done.")

if __name__ == "__main__":
    main()
