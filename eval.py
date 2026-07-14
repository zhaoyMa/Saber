import os
import json
import argparse
import tqdm
import torch
import yaml
import jsonlines
from transformers import AutoTokenizer
from modeling_llada import LLaDAModelLM
from modeling_llada_fast import LLaDAModelLM as LLaDAModelLMFast
from decoding import decoding_default, decoding_wino, generate_with_saber, generate_with_dual_cache, generate_with_entropy, generate_with_margin, generate_with_remdm
import dataset_utils
import datetime
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def main():
    parser = argparse.ArgumentParser(description="Unified Config-driven Evaluation Script for Language Models")
    parser.add_argument("--config", type=str, required=True, help="Path to the dataset config YAML file")
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

    handler_class = dataset_utils.get_dataset_handler(config['dataset_name'])
    handler = handler_class()

    run_single_task_evaluation(config, model, tokenizer, handler)

def get_generation_function(method_name):
    if method_name == 'default':
        return decoding_default
    elif method_name == 'wino':
        return decoding_wino
    elif method_name == 'saber':
        return generate_with_saber
    elif method_name == 'fast':
        return generate_with_dual_cache
    elif method_name == 'entropy':
        return generate_with_entropy
    elif method_name == 'margin':
        return generate_with_margin
    elif method_name == 'remdm':
        return generate_with_remdm
    else:
        raise ValueError(f"Unknown method: {method_name}")

def run_single_task_evaluation(config, model, tokenizer, handler):
    dataset_name = config['dataset_name']
    print(f"==> Running Single-Task Evaluation for {dataset_name}")

    gen_cfg = config['generation_args']
    num_samples = gen_cfg.get('num_samples', 1)
    method_name = config['method']
    method_params = config.get('method_args', {}).get(method_name, {})
    generation_fn = get_generation_function(method_name)

    print(f"==> Loading dataset...")
    dataset = handler.load_dataset(config)
    print(f"==> Loaded {len(dataset)} samples from {dataset_name} dataset.")

    print("==> Performing warm-up run with one sample...")
    warmup_doc = dataset[0]
    context, _, trailing_prompt = handler.doc_to_text(warmup_doc)
    prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False) + trailing_prompt
    print(f"==> Prompt: {prompt}")
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

    if method_name == 'remdm':
        remdm_params = method_params.copy()
        remdm_params.setdefault('gen_length', 256)
        remdm_params.setdefault('init_unmask_ratio', 0.875)
        remdm_params.setdefault('unmask_k', 1)
        remdm_params.setdefault('loop_steps', 32)
        remdm_params.setdefault('temperature', 0.)
        remdm_params.setdefault('cfg_scale', 0.)
        remdm_params.setdefault('remasking', 'low_confidence')
        remdm_params.setdefault('mask_id', 126336)
        remdm_params.setdefault('tokenizer', tokenizer)
        remdm_params.setdefault('block_length', 128)
        gen_output, steps = generate_with_remdm(model, input_ids, **remdm_params)
    else:
        gen_output, steps = generation_fn(model, input_ids, **gen_cfg, **method_params)
    print("==> Warm-up complete.")

    total_len = len(dataset)
    raw_outputs, total_steps = [], 0
    for i in tqdm.tqdm(range(total_len), desc=f"Evaluating {dataset_name} with method '{method_name}'"):
        doc = dataset[i]
        context, gt_doc, trailing_prompt = handler.doc_to_text(doc)
        prompt = tokenizer.apply_chat_template(context, add_generation_prompt=True, tokenize=False) + trailing_prompt
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.cuda()

        for sample_idx in range(num_samples):
            if method_name == 'remdm':
                gen_output, steps = generate_with_remdm(model, input_ids, **remdm_params)
            else:
                gen_output, steps = generation_fn(model, input_ids, **gen_cfg, **method_params)
            gen_str = tokenizer.batch_decode(gen_output[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

            total_steps += steps

            processed_gen_str = handler.preprocess_generated_text(gen_str, doc)
            completion = handler.extract_answer(processed_gen_str, doc)

            result_item = {
                'completion': completion,
                'full_response': gen_str,
                'steps': steps,
                'index': i,
                'sample_idx': sample_idx
            }

            if 'task_id' in gt_doc:
                result_item['task_id'] = gt_doc['task_id']
            if 'question_id' in gt_doc:
                result_item["question_id"] = gt_doc["question_id"]

            raw_outputs.append(result_item)

    final_metrics = handler.evaluate(raw_outputs, config)

    output_path = f"./results/{config['dataset_name']}_{method_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    print(f"Results saved in .jsonl format to {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    with jsonlines.Writer(open(output_path, "w", encoding="utf-8")) as writer:
        writer.write_all(raw_outputs)

    if final_metrics is not None:
        print("\n--- Evaluation Summary ---")
        print(f"Dataset: {config['dataset_name']}")
        print(f"Method: {method_name}")
        for k, v in final_metrics.items():
            print(f"{k}: {v:.4f}")

    avg_steps = total_steps / total_len if total_len > 0 else 0
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Number of samples per problem: {num_samples}")

if __name__ == "__main__":
    main()