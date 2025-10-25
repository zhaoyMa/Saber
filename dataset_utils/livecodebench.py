import re
def lcb_extract_answer(code_str, doc):
    if not isinstance(code_str, str):
        code_str = str(code_str)

    m = re.search(r'```python\s*(.*?)\s*```', code_str, flags=re.DOTALL)
    if m:
        return m.group(1).strip()
    return code_str.strip()

def lcb_doc_to_text(doc):
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
    