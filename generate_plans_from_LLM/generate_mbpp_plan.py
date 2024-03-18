
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
    with open(args.test_path, 'r', encoding='utf-8') as f:
        prompt_list = f.readlines()

    with open("code_prompt.txt", 'r', encoding='utf-8') as ff:
        prompt_plan = ff.read()

    if args.start > len(prompt_list) or args.start < 0:
        print(f"start index {args.start} > number of problems {len(prompt_list)}")
        return
    start = args.start
    if args.end is None or args.end > len(prompt_list):
        end = len(prompt_list)
    else:
        end = args.end
    prompt_list = prompt_list[start:end]

    for index, prompt in tqdm(enumerate(prompt_list), ncols=0, total=len(prompt_list)):

        json_prompt = json.loads(prompt)
        json_prompt_ = json_prompt['code']
        problem_id = json_prompt["task_id"]
        if os.path.exists(os.path.join(args.save_path, f"{problem_id}.txt")):
            continue
        input_text = "code:\n"
        input_text += json_prompt_
        input_text += "\nWrite steps according to the code:\n"  
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
                i = json_prompt["task_id"]
                save_path = os.path.join(args.save_path, f"{i}.txt")
                save_prompt = completion.choices[0].message["content"]
                with open(save_path, 'w', encoding='utf-8') as fff:
                    fff.write(save_prompt)
                time.sleep(20)
            except:
                time.sleep(20)
                continue

if __name__ == "__main__":
    
    import argparse

    parser = argparse.ArgumentParser(description="se gpt-3.5-turbo to generate plan")

    parser.add_argument("--test_path", default="", type=str, help='Path to test samples') 
    parser.add_argument("--save_path", default="", type=str, help='Path to save plan') 
    parser.add_argument("-s","--start", default=600, type=int, help='start index of test samples')
    parser.add_argument("-e","--end", default=2000, type=int, help='end index of test samples')

    args = parser.parse_args()
    
    main(args)