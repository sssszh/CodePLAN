
import torch
import glob
import logging
import random
import fnmatch

from multiprocessing import Manager
# from multiprocessing.shared_memory import ShareableList
import sys
import Datasets_codegen.util as dsutil
import numpy as np
import gc
import os
import io

import transformers

from .reindent import run as run_reindent
from tqdm import tqdm 

import json

class APPSBaseDataset(torch.utils.data.Dataset):
    def __init__(self, dataroot, problem_dirs, mode, max_tokens, sample_mode):
        self.dataroot = dataroot 
        self.problem_dirs = problem_dirs 

        self.mode = mode
        self.sample_mode = sample_mode # Either "uniform_sol" or "uniform_prob"
        self.max_tokens = max_tokens

        self.samples = []           # Should be set in initialize()
        self.initialize()
        print("===================================================================================")
        print("load tokenizer:",mode)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained("Salesforce/codegen-350M-mono")

    def load_plan_samples(self, sols_plan, answer_type, starter_code, question_str): 
        samples = []
        
        for idx, plan_str in enumerate(sols_plan):   
            
            sample = (question_str, starter_code, plan_str, answer_type)
            samples.append(sample)
            
        return samples

    def initialize(self):
        """
        Assume self.dataroot is set to folderName/data
        """

        all_samples = []
        plan_samples = []
        skipped_problems = []

        all_samples_dict = {} # Mapping from question_fname to list of samples

        print(f"Loading {len(self.problem_dirs)} problems from {self.dataroot}.")
        for problem_name in tqdm(self.problem_dirs):
            question_fname = os.path.join(self.dataroot, problem_name, "question.txt")
            sols_fname = os.path.join(self.dataroot, problem_name, "solutions.json")
            plans_fname = os.path.join(self.dataroot, problem_name, 'plans.json')
            starter_code = os.path.join(self.dataroot, problem_name, "starter_code.py")

            # print(question_fname)

            if os.path.exists(starter_code):
                answer_type = "\nUse Call-Based format\n"
            else:
                answer_type = "\nUse Standard Input format\n"

            if (not os.path.isfile(question_fname)) or (not os.path.isfile(sols_fname)):
                skipped_problems.append(problem_name)
                continue
            if not os.path.exists(plans_fname):
                plans_fname = None
            if (os.path.isfile(starter_code)):
                with open(starter_code, 'r') as f:
                    starter_code = f.read()
            else:
                starter_code = ""

            # Read the question description
            with open(question_fname, 'r') as f:
                question_str = f.read()

            # Read all the solutions
            with open(sols_fname, 'r') as f:
                sols_str_list = json.load(f)
                for sol_str in sols_str_list:
                    sol_str = reindent_code(sol_str)
                    sample = (question_str, starter_code, sol_str, answer_type)
                    
                    all_samples.append(sample)
                    if question_str in all_samples_dict:
                        all_samples_dict[question_str].append(sample)
                    else:
                        all_samples_dict[question_str] = [sample]

            sols_str_list = json.load(open(sols_fname, 'r'))
            if  plans_fname:
                sols_plan_list = json.load(open(plans_fname, 'r'))
                plan_sample = self.load_plan_samples(sols_plan_list, answer_type, starter_code, question_str)
                plan_samples += plan_sample
        
        print(f"Loaded {len(all_samples)} saamples from {self.dataroot}.")
        print(f"Skipped {len(skipped_problems)} problems from {self.dataroot}.")
        self.samples = all_samples
        self.samples_dict = all_samples_dict
        self.plan_samples = plan_samples


    def __len__(self):
        return len(self.samples)


    def pack_samples(self, idx, sample_type=None):
        curr_num_tokens = 0
        curr_samples = [] 

        if sample_type == 'plan':
            sample_pool = self.plan_samples
        else:
            sample_pool = self.samples

        if self.sample_mode == 'uniform_sol':
            curr_q, curr_s, curr_a, curr_q_prefix = sample_pool[idx]
        elif self.sample_mode == 'uniform_prob':
            curr_q = random.choice(list(self.samples_dict.keys()))
            curr_q, curr_s, curr_a, curr_q_prefix = random.choice(self.samples_dict[curr_q])
        else:
            raise NotImplementedError()

        while curr_num_tokens < self.max_tokens:

            # Never remove. Fixes stalling bug.
            curr_q = curr_q[:150000]
            curr_s = curr_s[:150000]
            curr_a = curr_a[:150000]

            curr_num_tokens += len(self.tokenizer.tokenize(curr_q))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_s))
            curr_num_tokens += len(self.tokenizer.tokenize(curr_a))

            curr_samples.append((curr_q, curr_s, curr_a, curr_q_prefix))

            if self.sample_mode == 'uniform_sol':
                curr_q, curr_s, curr_a, curr_q_prefix = random.choice(self.samples)
            elif self.sample_mode == 'uniform_prob':
                curr_q = random.choice(list(self.samples_dict.keys()))
                curr_q, curr_s, curr_a, curr_q_prefix = random.choice(self.samples_dict[curr_q])
            else:
                raise NotImplementedError()

        return curr_samples

    def __getitem__(self, idx):
        
        raw_samples = self.pack_samples(idx)

        plan_sample_idx = random.randint(0, len(self.plan_samples)-1)
        plan_samples = self.pack_samples(plan_sample_idx, "plan")

        retval = sample_gpt_task(
                raw_samples,
                max_tokens=self.max_tokens, 
                tokenizer=self.tokenizer, 
            )

        plan_inputs = sample_gpt_plan_task(
                plan_samples,
                max_tokens=self.max_tokens, 
                tokenizer=self.tokenizer, 
            )

        for k, v in plan_inputs.items():
            retval['pl_{}'.format(k)] = v
    
        gc.collect()
        return retval

def sample_gpt_task(raw_samples, max_tokens, tokenizer):
    """
    Create the true sample used for the GPT task
    """

    input_ids = []
    label_ids = []
    
    for q_str, s_str, a_str, answer_type in raw_samples:
        
        # Loss is not calculated on this
        q_str =  "[GEN_CODE]"+"\nQUESTION:\n" + q_str + "\n" + s_str + "\n" + answer_type + "\nANSWER:\n"

        question_token_ids = tokenizer.encode(q_str, verbose=False)
        answer_token_ids   = tokenizer.encode(a_str, verbose=False)
        answer_token_ids.append(tokenizer.eos_token_id)

        input_ids.extend(question_token_ids)
        input_ids.extend(answer_token_ids)
        
        label_ids.extend([-100] * len(question_token_ids))
        label_ids.extend(answer_token_ids)
    
    # Sanity check
    assert len(input_ids) == len(label_ids)

    if len(input_ids) < max_tokens:
        print(len(input_ids))
        import pdb; pdb.set_trace()

    # Cut off the excess
    input_ids = input_ids[:max_tokens]
    label_ids = label_ids[:max_tokens]

    return {
        "input_ids" : torch.LongTensor(input_ids),
        "labels" :  torch.LongTensor(label_ids)
    }

def sample_gpt_plan_task(raw_samples, max_tokens, tokenizer):
    """
    Create the true sample used for the GPT task
    """

    input_ids = []
    label_ids = []
    
    for q_str, s_str, p_str, answer_type in raw_samples:
        
        # Loss is not calculated on this
        q_str =  "[GEN_PLAN]"+"\nQUESTION:\n" + q_str + "\n" + s_str + "\n" + answer_type + "\nLet's think step by step:\n"

        question_token_ids = tokenizer.encode(q_str, verbose=False)
        answer_token_ids   = tokenizer.encode(p_str, verbose=False)
        answer_token_ids.append(tokenizer.eos_token_id)

        input_ids.extend(question_token_ids)
        input_ids.extend(answer_token_ids)
        
        label_ids.extend([-100] * len(question_token_ids))
        label_ids.extend(answer_token_ids)
    
    # Sanity check
    assert len(input_ids) == len(label_ids)

    if len(input_ids) < max_tokens:
        print(len(input_ids))
        import pdb; pdb.set_trace()

    # Cut off the excess
    input_ids = input_ids[:max_tokens]
    label_ids = label_ids[:max_tokens]

    return {
        "input_ids" : torch.LongTensor(input_ids),
        "plan_id": [1],
        "labels" :  torch.LongTensor(label_ids)
    }


def reindent_code(codestr):
    """
    Given code string, reindent it in the same way that the
    Github dataset was indented
    """
    codestr = io.StringIO(codestr)
    ret = io.StringIO()

    run_reindent(
        codestr, 
        ret, 
        config = {
            "dry-run": False,
            "help": False,
            "to": 4,
            "from": -1,
            "tabs": True,
            "encoding": "utf-8",
            "is-tabs": False,
            "tabsize": 4,
            "all-tabs": False
        }
    )

    return ret.getvalue()
