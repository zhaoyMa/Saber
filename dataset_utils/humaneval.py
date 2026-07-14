import re
import os
from .registry import register_dataset, DatasetHandler
from .eval_humaneval.all_evaluate import evaluate_solution, evaluate_solution_et

@register_dataset('humaneval')
class HumanEvalHandler(DatasetHandler):
    def load_dataset(self, config):
        from datasets import load_dataset
        dataset_cfg = config['dataset_config']
        dataset_path = os.path.join(config['data_root'], dataset_cfg['load_dataset_args']['path'])
        split = dataset_cfg.get('split', 'test')
        return load_dataset(dataset_path, split=split)

    def doc_to_text(self, doc):
        prompt = f"Write a solution to the following problem and make sure that it passes the tests:\n```python\n{doc['prompt']}\n```"
        context = [{"role": "user", "content": prompt}]
        trailing_prompt = f"Here is the completed function:\n```python\n{doc['prompt']}\n"
        return context, doc, trailing_prompt

    def extract_answer(self, generated_text, doc):
        text_with_prefix = f"```python\n{doc['prompt']}\n" + generated_text
        entry_point = doc['entry_point']
        pattern = re.compile(rf"def\s+{entry_point}.*?:\n(.*?)\n```", re.DOTALL)
        match = pattern.search(text_with_prefix)
        if match:
            return match.group(1).rstrip()
        fallback = re.split(r"```", generated_text, maxsplit=1)
        if fallback:
            return fallback[0].rstrip()
        return generated_text

    def evaluate(self, outputs, config):
        dataset_cfg = config['dataset_config']
        problem_file = dataset_cfg.get('problem_file', "./data/humaneval/HumanEval.jsonl")
        problem_file_et = dataset_cfg.get('problem_file_et', './data/humaneval/HumanEval_ET.jsonl')
        
        k_values = [1]
        num_samples = config['generation_args'].get('num_samples', 1)
        if num_samples > 1:
            k_values.append(num_samples)
        
        metrics = evaluate_solution(outputs, problem_file=problem_file, save_path=None, k=k_values)
        metrics_et = evaluate_solution_et(outputs, problem_file=problem_file_et, k=k_values)
        
        return {**metrics, **{f"{k}_et": v for k, v in metrics_et.items()}}