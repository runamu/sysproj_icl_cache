import json
import os

# Input file and output file paths
input_file = "dataset/csqa/csqa_train.jsonl"
output_file = "csqa.xml"

def format_csqa(csqa_data, with_answer=True):
    # Extract question and choices from the dictionary
    question = csqa_data['question']
    choices = csqa_data['choices']
    answer_key = csqa_data['answerKey']

    # Find the text of the choice corresponding to the answer key
    answer_index = choices['label'].index(answer_key)
    answer_text = choices['text'][answer_index]

    # Create the formatted output using an f-string
    if with_answer:
        output = f"""Question: {question}
Choices: {' '.join([f'({label}) {text}' for label, text in zip(choices['label'], choices['text'])])}
Output: {answer_text}"""
    else:
        output = f"""Question: {question}
Choices: {' '.join([f'({label}) {text}' for label, text in zip(choices['label'], choices['text'])])}
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
        You will learn how to answer questions by studying a few example interactions. Analyze the provided examples carefully, identify the underlying patterns, reasoning approach, and response structure. Then apply these insights to solve a new, unseen question.
        For the upcoming interaction, I will provide you with several example question-answer pairs to demonstrate the desired response style and reasoning method. After studying these examples, you will be asked to answer a new question using the same approach.
    </system>
    <user>
        Answer the last question.
    </user>
"""
    for i, module in enumerate(modules):
        pml_template += f"""
  <module name="module_{i + 1}">
     {format_csqa(module)}
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
    pml_content = generate_pml_schema("csqa", data)

    # Save PML schema to file
    print(f"Saving PML schema to {output_file}...")
    save_pml(output_file, pml_content)
    print("Done.")

if __name__ == "__main__":
    main()
