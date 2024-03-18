# CodePLAN

### Installation
Please follow the `requirements.txt` file to install the relevant dependencies or run:
```bash
pip install -r requirements.txt
```
Since our method modifies the transformers of huggingface, please make sure to install the same transformers as ours (we use transformers version 4.26.1):
```
cd transformers
pip install -e .
```
### Datasets
We used two datasets to fine-tune our model:
* **APPS**: You can download it [here](https://github.com/hendrycks/apps). 
* **MBPP**: This dataset can be found [here](https://github.com/google-research/google-research/tree/master/mbpp). 
  

### Generating Plans from LLM
We use openai's gpt-3.5-turbo to generate the solution plans data for training.You can get it by running `generate_plans_from_LLM\generate_apps_plan.py`:
```
python generate_apps_plan.py --test_path data/apps/train --save_path data/apps/train --start 0 --end 5000
```

### Finetuning with solution plans
Using CodeT5 as an example, you can run `train_codet5.py` to finetune the CodeT5 with solution plans:
```
python train_codet5.py --model codet5-large-ntp-py --save_dir models -- train_path data/appps/train --tuning_mode plan --clone_pl_head True --epochs 10
```

### Generating codes with finetuned model
You can run `generate_codet5.py` to generate codes:
```
python generate_codet5.py --test_path data/apps/test -- output_path outputs/codes --model_path model --plan_head True --is_plan False --temperature
```

### Generating solution plans with finetuned model
You can run `generate_codet5_plan.py` to generate solution plans:
```
python generate_codet5_plan.py --test_path data/apps/test --output_path outputs/plans --model_path model --plan_head True --is_plan True --temperature 0.6
```
 
### Generating codes with solution plan
You can run 'generate_code_with_plan.py` to generate code with generated solution plan as prompt:
```
python generate_code_with_plan.py --test_path data/apps/test --output_path outputs/codes --model_path model --plan_head True --is_plan False
```

### Evaluate generated codes
You can run `test_one_solution.sh` to evaluate generated codes:
```
bash test_one_solution.sh
```
