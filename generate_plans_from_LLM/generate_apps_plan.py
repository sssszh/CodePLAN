
import openai
import os
import json
import glob
import torch
import time
from tqdm import tqdm

# Your openai API key
openai.api_key = ''

def main(args):
    original_problems = glob.glob(args.test_path + '\*')
    problems = sorted(original_problems)
    if args.start > len(problems) or args.start < 0:
        print(f"start index {args.start} > number of problems {len(problems)}")
        return
    start = args.start
    if args.end is None or args.end > len(problems):
        end = len(problems)
    else:
        end = args.end
    problems = problems[start:end]

    for index, problem in tqdm(enumerate(problems), ncols=0, total=len(problems)):

        prob_path = os.path.join(problem)
        print(f"problem path = {prob_path}")

        problem_id = int(problem.split('\\')[-1])
        if os.path.exists(os.path.join(args.save_path, f"{problem_id}.txt")):
            continue
        code_path = os.path.join(prob_path, "solutions.json")
        if not os.path.exists(code_path):
            continue
        with open(code_path, 'r', encoding='utf-8') as f:
            code_list = json.load(f)
        min_code = min(code_list, key=lambda x: len(x))
        with open("code_prompt.txt", 'r', encoding='utf-8') as f:
            prompt_plan = f.read()
        input_text = "code:\n" + min_code + "Write a step-by-step solution plan following the above code:\n"
        with torch.no_grad():
            try:
                completion = openai.ChatCompletion.create(
                    model="gpt-3.5-turbo",
                    temperature=0.2,
                    messages=[
                            {"role": "system", "content": "You are an assistant that helps programmers understand the code step by step."},
                            {"role": "user", "content": prompt_plan +input_text},
                        ]
                )
                save_path = os.path.join(args.save_path, f"{problem_id}.txt")
                save_prompt = completion.choices[0].message["content"]
                with open(save_path, 'w', encoding='utf-8') as fff:
                    fff.write(save_prompt)
                time.sleep(20)
            except:
                time.sleep(20)
                continue

if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description="use gpt-3.5-turbo to generate plan.")

    parser.add_argument("--test_path", default="", type=str, help='Path to test samples') 
    parser.add_argument("--save_path", default="", type=str, help='Path to save plans') 
    parser.add_argument("-s","--start", default=0, type=int, help='start index of test samples')
    parser.add_argument("-e","--end", default=5000, type=int, help='end index of test samples')

    args = parser.parse_args()
    
    main(args)