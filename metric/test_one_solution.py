import json
import numpy as np
import os
import os.path 
import pprint
import glob 
from tqdm import tqdm
import pdb
import traceback 
import pickle as pkl 
# import dill as pkl
from typing import List
import multiprocessing
from testing_util import run_test
TIMEOUT = 4

def check_correctness(prob_path, generation, timeout, debug , example_tests):
    """Check correctness of code generation with a global timeout.
    The global timeout is to catch some extreme/rare cases not handled by the timeouts
    inside `run_test`"""
    def _temp_run(prob_path, generation, debug,example_tests, result_result,result_error):
        # result.append(run_test(prob_path=prob_path, test=generation, debug=debug, example_tests = example_tests ))
        tmp = run_test(prob_path=prob_path, test=generation, debug=debug, example_tests = example_tests )
        result_result.extend(tmp[0])
        result_error.extend(tmp[1])
        # result_code = tmp[3]
    manager = multiprocessing.Manager()
    result_result = manager.list()
    result_error = manager.list()
    # result_code = manager.list()
    p = multiprocessing.Process(target=_temp_run, args=(prob_path, generation, debug, example_tests,result_result,result_error))
    p.start()
    p.join(timeout=timeout + 1)
    if p.is_alive():
        p.kill()
    if not result_result:
        # Reamark: ideally we would consider that all tests failed but we can't access number of tests here easily
        # so we use 21=the average number of tests for a smaple in the test split instead 
        result_result = [-1]
        result_error = [None]
        if debug:
            print(f"global timeout")
        return result_result,result_error
    return result_result,result_error



def eval_and_save_problems(args):

    problems = sorted(glob.glob(args.test_path + '/*'))  
    test_indices = [] 
    for problem_idx, problem in enumerate(problems): 
        problem_id = int(problem.split('/')[-1])
        code_file_path = args.code_path + '/{}.json'.format(problem_id)
        if os.path.exists(code_file_path):
            test_indices.append(problem_idx) 
    
    real_index = test_indices[args.index] 
    problem = problems[real_index]   
    
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    
    print('Testing sample {}'.format(problem))
    
    if args.example_tests:
        print("Using example tests") 
    
    codes_loc = args.code_path + '/{}.json'.format(real_index)   
    if not os.path.isfile(codes_loc):      
        exit() 
    with open(codes_loc, "r") as file: 
        gen_codes = json.load(file)[str(real_index)]['codes']

    test_file = os.path.join(problem, "input_output.json")   
    if not os.path.isfile(test_file):      
        exit()
    tests = json.load(open(test_file, 'r'))
    nb_tests = len(tests['inputs'])
    if args.max_tests!=-1 and nb_tests > args.max_tests: 
        exit() 
    
    if os.path.isfile(args.output_path + '/{}.pkl'.format(real_index)):
        print("remove file {}".format(args.output_path + '/{}.pkl'.format(real_index)))
        os.remove(args.output_path + '/{}.pkl'.format(real_index))

    print("Saving to {}".format(args.output_path + '/{}.pkl'.format(real_index)))

    all_results, all_errors, all_sols = [], [], []

    for o_idx, o in tqdm(enumerate(gen_codes), total=len(gen_codes), ncols=0, leave=False):   
        if args.debug:
            print("\ncandidate idx: ",o_idx,"========================================================================================\n")
        curr_results = []
        curr_errors = []
        try:
            curr_results, curr_errors = check_correctness(prob_path=problem,generation=o,timeout=TIMEOUT, debug=args.debug, 
                                          example_tests=args.example_tests)

            # curr_errors = [(e, traceback.format_tb(e.__traceback__)) if e is not None else e for e in curr_errors]  # current error为None表示成功运行但是
            fixed = []
            for e in curr_results:
                if isinstance(e, np.ndarray):
                    if len(e)==0:
                        e = -2
                        fixed.append(e)
                        continue
                    e = e.item(0)
                if isinstance(e, np.bool_):
                    e = bool(e)
                fixed.append(e)
            curr_results = fixed

        except Exception as e:
            print(f"test framework exception = {repr(e)}{e}\n")
            break
        finally:
            assert isinstance(curr_results, list)
            all_results.append(curr_results)
            # all_errors.append(curr_errors)

    '''
    How to read results:
    [-2] = compile error, 
    [-1] = runtime error 
    [False] = failed test case 
    [True] = passed test case
    '''

    save_results = {real_index : {'results': all_results}} 
    pkl.dump(save_results,  open(args.output_path + '/{}.pkl'.format(real_index), "wb"))                    

def main(args):    
    argsdict = vars(args)    
    eval_and_save_problems(args)

if __name__ == "__main__":
    from unit_test_configs import * 
    main(args)
