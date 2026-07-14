import re
import os
import json
import textwrap
from .registry import register_dataset, DatasetHandler
from .eval_correctness_mbpp.evaluation import evaluate_functional_correctness
import tempfile
import json as json_module

def format_test_example(q, tests, code=None):
    prompt = ">>> Problem:\n{}\n>>> Test Cases:\n{}\n".format(q.strip(), "\n".join(tests))
    if code:
        code = code.replace("\r", "").replace("\t", "    ")
        prompt += "\n>>> Code:\n```python\n{}\n```".format(code)
    return prompt

def mbpp_read_data(data_path):
    examples = [json.loads(x) for x in open(data_path)]
    for i in range(len(examples)):
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
            'examples': [],
        }

@register_dataset('mbpp')
class MBPPHandler(DatasetHandler):
    def load_dataset(self, config):
        dataset_cfg = config['dataset_config']
        dataset_path = os.path.join(config['data_root'], dataset_cfg['load_dataset_args']['path'])
        return list(mbpp_read_data(dataset_path))

    def doc_to_text(self, doc):
        prompt = 'You are an expert Python programmer. Please write a python function to solve the following problem:\n' + doc['prompt']
        context = [{"role": "user", "content": prompt}]
        return context, doc, f">>> Code:\n```python\n"

    def extract_answer(self, generated_text, doc):
        pattern = re.compile(
            rf"(def\s+{doc['entry_point']}\s*\(.*?\):\n.*?)(?=^```)",
            re.DOTALL | re.MULTILINE
        )
        match = pattern.search(generated_text)
        if match:
            return match.group(1).rstrip()
        return textwrap.indent(generated_text, " " * 4)

    def preprocess_generated_text(self, generated_text, doc):
        return f"```python\n" + generated_text + "\n```"

    def evaluate(self, outputs, config):
        dataset_cfg = config['dataset_config']
        problem_file = dataset_cfg.get('problem_file', "./data/mbpp/mbpp_sanitized.jsonl")
        problem_file_et = dataset_cfg.get('problem_file_et', "./data/mbpp/MBPP_ET.jsonl")
        
        k_values = [1]
        num_samples = config['generation_args'].get('num_samples', 1)
        if num_samples > 1:
            k_values.append(num_samples)
        
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".jsonl") as temp_f:
            for item in outputs:
                temp_f.write(json_module.dumps(item) + "\n")
                temp_file_path = temp_f.name
        
        metrics = evaluate_functional_correctness(temp_file_path, problem_file=problem_file, is_mbpp=True, k=k_values)
        metrics_et = evaluate_functional_correctness(temp_file_path, problem_file=problem_file_et, is_mbpp=True, k=k_values)
        
        os.unlink(temp_file_path)
        
        return {**metrics, **{f"{k}_et": v for k, v in metrics_et.items()}}