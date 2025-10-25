import re
import os
import json
import random
import textwrap

def format_test_example(q, tests, code: str=None):
    prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
    if code:
        code = code.replace("\r", "").replace("\t", "    ")
        prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
    return prompt
def mbpp_read_data(data_path: str):
    examples = [json.loads(x) for x in open(data_path)]
    examples_str = []
    for i in range(0, len(examples)):
        ex = examples[i]

        if 'prompt' not in ex:
            raise KeyError(f"Missing 'prompt' field in example {i}")
        if 'test_list' not in ex or not isinstance(ex['test_list'], list) or len(ex['test_list']) == 0:
            raise ValueError(f"'test_list' must be a non-empty list in example {i}")

        q = ex['prompt']
        test = ex['test_list']

        prompt = format_test_example(q, test, code=None)

        fn_candidates = re.findall(r"([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", test[0])
        if not fn_candidates:
            raise ValueError(f"Can't find entry point in test[0]: {i} {test[0]}")
        entry_point = fn_candidates[-1]
        yield {
            'task_id': ex.get('task_id', str(i)),
            'prompt': prompt,
            'entry_point': entry_point,
            'examples': examples_str,   
        }
def mbpp_extract_answer(text, entry_point):
    pattern = re.compile(
        rf"(def\s+{entry_point}\s*\(.*?\):\n.*?)(?=^```)",
        re.DOTALL | re.MULTILINE
    )
    match = pattern.search(text)
    if match:
        return match.group(1).rstrip()
    
    return textwrap.indent(text, " " * 4)


def mbpp_doc_to_text(doc):
    prompt = 'You are an expert Python programmer. Please write a python function to solve the following problem:\n' + doc['prompt']
    context = [{"role": "user", "content": prompt}]
    return context, doc, f">>> Code:\n```python\n"

