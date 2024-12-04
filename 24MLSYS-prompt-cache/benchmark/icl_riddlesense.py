import re

from .benchmark_base import Benchmark, Entry
from .utils import XMLSchemaBuilder
from datasets import load_dataset
import os
import json
import random

def read_jsonl(file_path):
    """
    Read a JSONL file and return a list of dictionaries.

    Args:
        file_path (str): Path to the JSONL file

    Returns:
        list: A list of dictionaries, each representing a JSON object from the file
    """
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            # Parse each line as a JSON object
            data.append(json.loads(line.strip()))

    return data

def escape_tags(input_str):
    # pattern = r'<(?P<content>.*?)>'

    # # The lambda function ensures only the first letter is capitalized
    # def repl(match):
    #     return '(' + match.group("content").capitalize() + ')'
    #
    # return re.sub(pattern, repl, input_str)
    return input_str.replace('<', '(').replace('>', ')')

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

class ICLRiddleSense(Benchmark):
    def __init__(self):
        super().__init__("icl_riddlesense")
        # self.next_query_idx = 0

    def init(self, limit_entries=None):
        """
        Download (one time) and load the dataset to run;
        Preprocess the dataset to be organized in the `Entry` format.
        """
        targets_filename = "../dataset/riddlesense/rs_test_targets.jsonl"
        examples_filename = "../dataset/riddlesense/rs_test_examples.jsonl"


        data_targets = read_jsonl(targets_filename)
        data_examples = read_jsonl(examples_filename)

        # with open(targets_filename, 'r', encoding='utf-8') as f:
        #     data_targets = f.readlines()
        # with open(examples_filename, 'r', encoding='utf-8') as f:
        #     data_examples = f.readlines()

        self.dataset = {
            "targets": data_targets,
            "examples": data_examples
        }

        # import ipdb; ipdb.set_trace()

        icl_cache_module_template = """<module name="{name}">
{example}
</module>"""

        icl_cache_modules = ""
        ex_module_names = []
        for example in self.dataset["examples"]:
            icl_cache_modules += icl_cache_module_template.format(name=f'ex_{example["id"]}', example=format_riddle(example))
            ex_module_names.append(f'ex_{example["id"]}')

        count = 0
        for target in self.dataset["targets"]:
            if limit_entries is not None and count >= limit_entries:
                break

            # import ipdb; ipdb.set_trace()
            schema_name = f"schema_tg_{target['id']}"
            # TODO: use `self.dataset_prompt`
            schema_content = f"""<schema name="{schema_name}">
	<system>You will learn how to answer questions by studying a few example interactions. Analyze the provided examples carefully, identify the underlying patterns, reasoning approach, and response structure. Then apply these insights to solve a new, unseen question.</system>
	<user>For the upcoming interaction, I will provide you with several example question-answer pairs to demonstrate the desired response style and reasoning method. After studying these examples, you will be asked to answer a new question using the same approach.
		{icl_cache_modules}
	</user>
	<assistant>I understand. I will carefully examine the example interactions, extract the key principles and patterns of response, and then apply this learned approach to the new question. Please provide the example question-answer pairs.</assistant>
</schema>"""

            #
            #
            # builder = XMLSchemaBuilder(schema_name)
            # context = item["context"]
            # # print(len(context))
            # # title = item["title"]
            # question = item["input"]
            # answer = item["answers"]
            # builder.set_system_description(_system_description)
            # builder.set_user_description(_user_description)
            # builder.add_document_module("context",
            #                             self.dataset_prompt["context"].format(context=escape_tags(context)))
            # builder.set_assistant_description(_assistant_description)

            schema_file_name = f"{schema_name}.xml"
            with open(os.path.join(self.schema_path, schema_file_name), "w") as f:
                f.write(schema_content)

            # randomly select 2 examples
            selected_examples = random.sample(ex_module_names, 3)

            prompt = f"""<prompt schema='{schema_name}'>
<{selected_examples[0]}/>
<{selected_examples[1]}/>
<{selected_examples[2]}/>
<user>
{format_riddle(target, with_answer=False)}
</user>
</prompt>"""
            self.entries.append(Entry(schema_file_name, prompt, target["answerKey"]))

            count += 1

            # import ipdb; ipdb.set_trace()
            # print("schema_name", schema_name)

if __name__ == '__main__':
    sq = ICLRiddleSense()
    sq.init()
    print(sq.get_entry_count())
    print(sq.get_query((100, 101)))
