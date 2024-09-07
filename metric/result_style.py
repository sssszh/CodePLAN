
import os
import pickle as pkl
import numpy as np

def sort_by_number(filename):
    number = filename.split('.')[0]
    return int(number)

def deal_result(root_path):
    filename_list = os.listdir(root_path)
    filename_list = sorted(filename_list, key=sort_by_number)
    save_result = {}
    for filename in filename_list:
        try:
            with open(os.path.join(root_path, filename), 'rb') as f:
                data = pkl.load(f)
            file_id = int(filename.split('.')[0])
            result = data[file_id]["results"]
            # result = [[x[0] for x in result_]]
            save_result[file_id] = result
        except:
            # print(f"{file_id}")
            continue
    return save_result
