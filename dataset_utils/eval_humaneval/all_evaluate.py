import re
import json
import copy
import argparse
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))

from .utils import build_test_method, find_method_name, code_split, prompt_split_humaneval
from .execute.execution import evaluate_with_test_code, evaluate_with_test_code_T
from .evaluation import pass_at_K, AvgPassRatio
from datasets import load_dataset, load_from_disk

def evaluate_solution(generation_list, problem_file, dataset_type='humaneval', lang='python', timeout=10):

    if dataset_type == 'humaneval':
        with open(problem_file, 'r') as f:
            dataset = [json.loads(line) for line in f]
    prompt_dict = {item['task_id']: item['prompt'] for item in dataset}
    test_dict = {item['task_id']: item['test'] for item in dataset}

    handled_solutions = generation_list

    
    print(f"Loaded {len(handled_solutions)} solutions")
    
    for solution in handled_solutions:
        id = solution['task_id']
        solution["generation"] = prompt_dict[id] + solution["completion"]  
        solution["prompt"] = ""
        solution["test"] = test_dict[id]
        solution["entry_point"] = find_method_name(solution["generation"]) if find_method_name(solution["generation"]) else "candidate"
        solution["completion"] = solution["generation"]
    
    exec_result = evaluate_with_test_code(handled_solutions, timeout=timeout)

    pass_at_1_result = pass_at_K(exec_result, k=[1])
    print(f"Pass@1 with extended test cases: {pass_at_1_result}")
    return pass_at_1_result

def evaluate_solution_et(generation_list, problem_file, dataset_type='humaneval', lang='python', timeout=10):

    if dataset_type == 'humaneval':
        with open(problem_file, 'r') as f:
            dataset = [json.loads(line) for line in f]
    prompt_dict = {item['task_id']: item['prompt'] for item in dataset}
    handled_solutions = generation_list
    
    print(f"Loaded {len(handled_solutions)} solutions")

    for solution in handled_solutions:
        id = solution['task_id']
        solution["generation"] = prompt_dict[id] + solution["completion"]  
        solution["prompt"] = ""
        solution["entry_point"] = find_method_name(solution["generation"]) if find_method_name(solution["generation"]) else "candidate"
        solution["completion"] = solution["generation"]

    with open(problem_file, 'r') as f:
        test_cases = [json.loads(line) for line in f]
    
    test_cases_dict = {}
    for case in test_cases:
        test = build_test_method(case['test_case_list'], "", case['entry_point'])
        test_cases_dict[case['task_id']] = test
    
    for solution in handled_solutions:
        solution['test'] = test_cases_dict[solution['task_id']]
    
    exec_result_T = evaluate_with_test_code(handled_solutions, timeout=timeout)
    
    pass_at_1_result = pass_at_K(exec_result_T, k=[1])
    
    print(f"Pass@1 with extended test cases: {pass_at_1_result}")
    return pass_at_1_result

