import re
import os
from .registry import register_dataset, DatasetHandler

@register_dataset('livecodebench')
class LiveCodeBenchHandler(DatasetHandler):
    def load_dataset(self, config):
        from datasets import load_dataset
        dataset_cfg = config['dataset_config']
        dataset_path = os.path.join(config['data_root'], dataset_cfg['load_dataset_args']['path'])
        return load_dataset("json", data_files=dataset_path, split="train")

    def doc_to_text(self, doc):
        prompt = f"You will be given a question (problem specification) and will generate a correct Python program that matches the specification and passes all tests. You will NOT return anything except for the program.\n\n"
        prompt += f"Question:\n{doc['question_content']}\n"
        if doc["starter_code"]:
            prompt += "You will use the following starter code to write the solution to the problem and enclose your code within delimiters.\n"
            prompt += f"```python\n{doc['starter_code']}\n```\n\n"
        else:
            prompt += "Read the inputs from stdin solve the problem and write the answer to stdout (do not directly test on the sample inputs). Enclose your code within delimiters as follows. Ensure that when the python program runs, it reads the inputs, runs the algorithm and writes output to STDOUT.\n"
            prompt += "```python\n# YOUR CODE HERE\n```\n\n"
        context = [{"role": "user", "content": prompt}]
        return context, doc, ""

    def extract_answer(self, generated_text, doc):
        if not isinstance(generated_text, str):
            generated_text = str(generated_text)
        m = re.search(r'```python\s*(.*?)\s*```', generated_text, flags=re.DOTALL)
        if m:
            return m.group(1).strip()
        return generated_text.strip()

    def evaluate(self, outputs, config):
        return None