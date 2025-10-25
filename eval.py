import os
import json
import argparse
import tqdm
from datasets import load_dataset
import torch
import yaml
import jsonlines
from transformers import  AutoTokenizer
from modeling_llada import LLaDAModelLM
from modeling_llada_fast import LLaDAModelLM as LLaDAModelLMFast
from decoding import decoding_default, decoding_wino, generate_with_saber, generate_with_dual_cache,generate_with_entropy,generate_with_margin,generate_with_remdm
import dataset_utils
import tempfile
from dataset_utils.eval_correctness_mbpp.evaluation import evaluate_functional_correctness
from dataset_utils.eval_humaneval.all_evaluate import evaluate_solution, evaluate_solution_et
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    parser = argparse.ArgumentParser(description="Unified Config-driven Evaluation Script for Language Models")
    parser.add_argument("--config", type=str, required=True, help="Path to the dataset config YAML file (e.g., configs/gsm8k.yaml)")
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    print(f"==> Loaded config for: {config['dataset_name']}")

    model_path = config['model_path']
    print(f"==> Loading model: {model_path}")
    method_name = config['method']
    if method_name == 'fast':
        model = LLaDAModelLMFast.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda().eval()  
    else:
        model = LLaDAModelLM.from_pretrained(model_path, torch_dtype=torch.bfloat16).cuda().eval()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    run_single_task_evaluation(config, model, tokenizer)


def get_generation_function(method_name):
    if method_name == 'default': return decoding_default
    elif method_name == 'wino': return decoding_wino
    elif method_name == 'saber': return generate_with_saber
    elif method_name == 'fast': return generate_with_dual_cache 
    elif method_name == 'entropy': return generate_with_entropy
    elif method_name == 'margin': return generate_with_margin
    elif method_name == 'remdm': return generate_with_remdm
    else: raise ValueError(f"Unknown method: {method_name}")

def run_single_task_evaluation(config, model, tokenizer):
    dataset_name = config['dataset_name']
    print(f"==> Running Single-Task Evaluation for {dataset_name}")
    dataset_cfg = config['dataset_config']
    doc_to_text_fn = getattr(dataset_utils, dataset_cfg['doc_to_text_fn'])
    extract_answer_fn = getattr(dataset_utils, dataset_cfg.get('extract_answer_fn')) if dataset_cfg.get('extract_answer_fn') else None
    
    gen_cfg = config['generation_args']
    method_name = config['method']
    method_params = config.get('method_args', {}).get(method_name, {})
    generation_fn = get_generation_function(method_name)
    print(f"==> Loading dataset...")
    loader_type = dataset_cfg.get('data_loader', 'huggingface')
    dataset_path = os.path.join(config['data_root'], dataset_cfg['load_dataset_args']['path'])
    
    if loader_type == 'huggingface':
        dataset_name_hf = dataset_cfg['load_dataset_args'].get('name')
        dataset = load_dataset(dataset_path, dataset_name_hf, trust_remote_code=True)[dataset_cfg['split']]
    elif dataset_name == 'mbpp':
        loader_fn = getattr(dataset_utils, dataset_cfg['loader_fn'])
        dataset = list(loader_fn(dataset_path))
    elif dataset_name == 'livecodebench':
        dataset = load_dataset("json",data_files = dataset_path, split="train")
    print(f"==> Loaded {len(dataset)} samples from {dataset_name} dataset.")
    
    # ---Warm-up---
    print("==> Performing warm-up run with one sample...")
    warmup_doc = dataset[0]
    trailing_prompt = "" 
    context, _, trailing_prompt = doc_to_text_fn(warmup_doc) 
    prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False) + trailing_prompt
    print(f"==> Prompt: {prompt}")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()
    if method_name == 'remdm':
        gen_output, steps = generate_with_remdm(model, input_ids, gen_length=256, init_unmask_ratio=0.875, unmask_k=1, loop_steps=32, temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336, tokenizer=tokenizer, block_length=128)
    else:
        gen_output, steps = generation_fn(model, input_ids,**gen_cfg, **method_params)
    print("==> Warm-up complete.")

    # --- Warm-up complete ---
    total_len = len(dataset)
    raw_outputs, total_steps = [], 0
    trailing_prompt = "" 
    for i in tqdm.tqdm(range(total_len), desc=f"Evaluating {dataset_name} with method '{method_name}'"):
        doc = dataset[i]
        context, gt_doc, trailing_prompt = doc_to_text_fn(doc)
        prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False) + trailing_prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        if method_name == 'remdm':
            gen_output, steps = generate_with_remdm(model, input_ids, gen_length=256, init_unmask_ratio=0.875, unmask_k=1, loop_steps=32, temperature=0., cfg_scale=0., remasking='low_confidence', mask_id=126336, tokenizer=tokenizer, block_length=128)
        else:
            gen_output, steps = generation_fn(model, input_ids,**gen_cfg, **method_params)
        gen_str = tokenizer.batch_decode(gen_output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]
        
        total_steps += steps
        
        result_item = {'completion':'', 'full_response': gen_str, 'steps': steps,'index': i}

        if dataset_name == 'mbpp':
            gen_str = f"```python\n" + gen_str
            gen_code = extract_answer_fn(gen_str, doc['entry_point'])
            result_item['completion'] = gen_code
        else:
            result_item['completion'] = extract_answer_fn(gen_str, doc)
        if 'task_id' in gt_doc: result_item['task_id'] = gt_doc['task_id']
        #if 'prompt' in gt_doc: result_item['prompt'] = gt_doc['prompt']
        if 'question_id' in gt_doc: result_item["question_id"] = gt_doc["question_id"]
            
        raw_outputs.append(result_item)
    
    final_metrics = {}
    final_metrics_et = None
    if evaluate_functional_correctness is None:
        print("Warning: 'human-eval' library not found. Skipping functional correctness evaluation.")
    else:
        print("\n==> Generations complete. Calling official evaluation script...")
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix=".jsonl") as temp_f:
            for item in raw_outputs:
                temp_f.write(json.dumps(item) + "\n")
                temp_file_path = temp_f.name
        if dataset_name == 'humaneval':
            problem_file = config['dataset_config'].get('problem_file', "./data/humaneval/HumanEval.jsonl")
            problem_file_et = config['dataset_config'].get('problem_file_et', './data/humaneval/HumanEval_ET.jsonl')
            final_metrics = evaluate_solution(raw_outputs,problem_file=problem_file)
            print(f"ET Evaluation...{problem_file_et}")
            final_metrics_et = evaluate_solution_et(raw_outputs,problem_file=problem_file_et)
        elif dataset_name == 'mbpp':
            problem_file = config['dataset_config'].get('problem_file', "./data/mbpp/mbpp_sanitized.jsonl")
            problem_file_et = config['dataset_config'].get('problem_file_et', "./data/mbpp/MBPP_ET.jsonl")
            final_metrics = evaluate_functional_correctness(temp_file_path,problem_file=problem_file,is_mbpp=True)
            final_metrics_et = evaluate_functional_correctness(temp_file_path,problem_file=problem_file_et,is_mbpp=True)
        elif dataset_name == 'livecodebench':
            problem_file = config['dataset_config'].get('problem_file', 'default')
        os.unlink(temp_file_path)
    external_metrics = final_metrics
    output_path = f"./results/{config['dataset_name']}_{method_name}.jsonl"
    print(f"Results saved in .jsonl format to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with jsonlines.Writer(open(output_path, "w", encoding="utf-8")) as writer:
        writer.write_all(raw_outputs)
    
    if dataset_name != 'livecodebench':
        print("\n--- Evaluation Summary ---")
        print(f"Dataset: {config['dataset_name']}")
        print(f"Method: {method_name}")
        if external_metrics:
            accuracy = external_metrics['pass@1']
        else:
            print("Accuracy: N/A (no external metrics available)")
        print(f"Accuracy: {accuracy:.4f}" if isinstance(accuracy, float) else f"Accuracy: {accuracy}")

        if final_metrics_et is not None:
            print("\n--- ET Evaluation Summary ---")
            print(f"Accuracy_et: {final_metrics_et['pass@1']:.4f}")
        avg_steps = total_steps / total_len if total_len > 0 else 0
        print(f"Average Steps: {avg_steps:.2f}")


if __name__ == "__main__":
    main()