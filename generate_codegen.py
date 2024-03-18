
import json
import os
import pprint
import torch
import pdb 
import glob 
from tqdm import tqdm
import pickle as pkl 
import numpy as np 
from collections import Counter 
from transformers import AutoTokenizer, AutoModelForCausalLM

os.environ["CUDA_VISIBLE_DEVICES"] = '1'

def generate_prompt(args, test_case_path, pro_path, solutions_path, tokenizer, 
                    starter_path=None):
    
    if args.is_plan:
        _input = "[GEN_PLAN]"
    else:
        _input = "[GEN_CODE]"
    _input += "\nQUESTION:\n"
    with open(pro_path, "r") as f:
        data = f.readlines()
        data = "".join(data)
    _input += data
    
    if starter_path != None:
        with open(starter_path, "r") as f:
            data = f.readlines()
            data = "".join(data)
            data = "\n" + data 
        _input += data
    
    if os.path.exists(test_case_path):
        with open(test_case_path, "r") as f:
            data = json.load(f)
        if not data.get("fn_name"):
            _input += "\nUse Standard Input format"
        else:
            _input += "\nUse Call-Based format"
    elif starter_path is not None and os.path.exists(starter_path):
        _input += "\nUse Call-Based format"
    else:
        _input += "\nUse Standard Input format"
        
    if args.is_plan:
        _input += "\nLet's think step by step:\n"
    
    return _input

def main(args):

    argsdict = vars(args)
    print(pprint.pformat(argsdict))

    original_problems = glob.glob(args.test_path + '/*')
    problems = sorted(original_problems) 

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path, exist_ok=True)
    print("Saving results to {}".format(args.output_path))

    if args.start > len(problems) or args.start < 0:
        print(f"start index {args.start} > number of problems {len(problems)}")
        return
    start = args.start
    if args.end is None or args.end > len(problems):
        end = len(problems)
    else:
        end = args.end
    problems = problems[start:end]
    
    # Set up model
    tokenizer = AutoTokenizer.from_pretrained('Salesforce/codegen-350M-mono')
    print("Loading model from {}...".format(args.model_path))
    model = AutoModelForCausalLM.from_pretrained(args.model_path, pad_token_id=tokenizer.eos_token_id, clone_pl_head = args.plan_head)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sols = []
    index_list = []
    # for i in range(1692, 2000):
    #     index_list.append(i)

    for index, problem in tqdm(enumerate(problems), ncols=0, total=len(problems)):
        
        prob_path = os.path.join(problem)
        print(f"problem path = {prob_path}")
        
        problem_id = int(problem.split('/')[-1])
        
        test_case_path = os.path.join(prob_path, "input_output.json")
        
        pro_path = os.path.join(prob_path, "question.txt")
        starter_path = os.path.join(prob_path, "starter_code.py")

        solutions_path = os.path.join(prob_path, "solutions.json")

        if not os.path.exists(starter_path):
            starter_path = None
        if os.path.exists(os.path.join(args.output_path, f"{problem_id}.json")):
            continue
        try:
            input_text = generate_prompt(args, test_case_path, pro_path, solutions_path, 
                                          tokenizer, starter_path)
        except:
            continue

        with torch.no_grad():

            try:
            
                input_ids = tokenizer.encode(input_text, verbose=False)
                
                
                if len(input_ids) > 1024:
                    input_ids = input_ids[:1024] + tokenizer.encode("\nANSWER:\n",verbose=False)
                    
                else:
                    input_ids = input_ids + tokenizer.encode("\nANSWER:\n",verbose=False)

                input_ids = torch.LongTensor(input_ids).unsqueeze(0).cuda()
            except:
                with open("information_tensor_new.txt", 'a') as fff:
                    fff.write(pro_path+'\n')
            num_loops = int(args.num_seqs / args.num_seqs_per_iter)
            output_programs = []
            try: 
                for i in tqdm(range(num_loops), ncols=0, total=num_loops, leave=False):
                    output_ids = model.generate(
                            input_ids, 
                            do_sample=True, 
                            temperature=args.temperature, 
                            max_length=2048, 
                            num_return_sequences=args.num_seqs_per_iter,
                            top_p=0.95)

                    for output_id in output_ids: 
                        output_str = tokenizer.decode(output_id, skip_special_tokens=True)
                        if len(output_str):
                            output_program = output_str.split("ANSWER:\n")[1].replace("<|endoftext|>", "")
                            output_programs.append(output_program)
                        else:
                            output_programs.append("None code!")
            except Exception as e:
                with open("information_generate_error.txt", 'a') as ff:
                    ff.write(pro_path+'\n')
            if len(output_programs) == 0:
                output_programs = ["No code!"]*10
            save_codes = {
                f"{problem_id}": {'codes': output_programs}
            }
            save_loc = os.path.join(args.output_path, f"{problem_id}.json")
            with open(save_loc, 'w') as f:
                json.dump(save_codes, f)


if __name__ == "__main__":
    
    from configs.generate_codegen_configs import * 
    
    main(args)

        