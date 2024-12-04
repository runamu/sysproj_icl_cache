import random
import re

import numpy as np
import torch.cuda
import fire
import random
from promptcache.model import Llama2, Falcon, Mpt, CodeLlama
import json
from promptcache import Prompt, CompactSpaces, read_file, CacheEngine, \
    GenerationEngine, GenerationParameters, llama2_template


def escape_tags(input_str):
    pattern = r'<(?P<content>.*?)>'

    def repl(match):
        return '(' + match.group("content").capitalize() + ')'

    return re.sub(pattern, repl, input_str)


def main(enable_cache=False, cuda_device=0):
    enable_cpu_inference = False
    disable_prompt_cache = not enable_cache

    torch.cuda.set_device(cuda_device)

    lm_for_cache = CodeLlama("codellama/CodeLlama-7b-Instruct-hf",
                             load_in_8bit=True,
                             device_map=f"cuda:{cuda_device}")
                            #  device_map=None)

    lm = lm_for_cache

    if enable_cpu_inference:
        lm = CodeLlama("codellama/CodeLlama-7b-Instruct-hf",
                       load_in_8bit=True,
                       device_map=None)

    # lm = Falcon("tiiuae/falcon-7b-instruct",
    #             load_in_8bit=True if not disable_cuda else False,
    #             device_map="auto" if not disable_cuda else None)

    # lm = Mpt("mosaicml/mpt-7b-chat-8k",
    #          load_in_8bit=True if not disable_cuda else False,
    #          device_map="auto" if not disable_cuda else None)

    cache_engine = CacheEngine(5000, lm_for_cache, target_device='cpu' if enable_cpu_inference else None)
    gen_engine = GenerationEngine(lm)

    preproc = [
        # CompactSpaces(),
        lm.get_formatter()
    ]

    # cache_engine.add_schema(read_file("/home/stilex/24MLSYS-prompt-cache/math.xml", preproc), max_tokens=800)
    cache_engine.add_schema(read_file("../math.xml", preproc), max_tokens=800)

    parameter = GenerationParameters(
        temperature=1.0,
        repetition_penalty=1.0,
        top_p=0.95,
        top_k=-1,
        max_new_tokens=512,
        stop_token_ids=lm.stop_token_ids,
        stop_str=lm.stop_str
    )
    # with open('/home/stilex/dst/LLaMA-Factory/data/gsm8k.jsonl', 'r') as file:
    with open('../dataset/gsm8k/gsm8k.jsonl', 'r') as file:
        # Read all lines from the file
        lines = file.readlines()

        # Start from the 101st entry (index 100)
        for i in range(100, len(lines)):
            # Parse each line as a JSON object
            question_data = json.loads(lines[i].strip())  # Strip any extra whitespace/newlines
            question = question_data.get('question', 'No question field found')  # Adjust based on your data structure
            print(f"Question {i + 1}: {question}")
            random_numbers = [random.randint(1, 10) for _ in range(3)]


            prompt_text = f"""
            <prompt schema='math'>
                <module_{random_numbers[0]}/>
                <module_{random_numbers[1]}/>
                <module_{random_numbers[2]}/>
                <user>
                    {question}
                </user>
            </prompt>
            """
            prompt = Prompt(prompt_text, preproc)
            print("prompt:", prompt)
            token_ids, position_ids, cache_time, cache = cache_engine.process(prompt, no_cache=disable_prompt_cache,
                                                                            return_full_position_ids=lm.use_full_position_ids)
            # print("cache:", cache)
            output_stream = gen_engine.generate(token_ids, position_ids, parameter, cache, stream_interval=2,
                                                use_full_position_ids=lm.use_full_position_ids)

            print(f"Assistant: ", end="", flush=True)

            resp = ""
            pre = 0
            for outputs in output_stream:
                output_text = outputs.new_text.strip().split(" ")
                now = len(output_text) - 1
                if now > pre:
                    tt = " ".join(output_text[pre:now])
                    resp += tt + " "
                    print(tt, end=" ", flush=True)
                    pre = now
            tt = " ".join(output_text[pre:])
            print(tt, flush=True)
            resp += tt

            print("\n")
            prompt_text += f"<assistant>{resp}</assistant>"


def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":
    seed_everything(42)
    fire.Fire(main)
