
import argparse

parser = argparse.ArgumentParser(description="Training CodeT5 model for code generation")
parser.add_argument('--model', default='codet5-large-ntp-py', type=str, help='type of transformers model as model backbone')
parser.add_argument('--model_path', default=None, type=str, help='path to model backbone pretrained weights') 
parser.add_argument('--save_dir', default='', type=str, help='path to save trained model checkpoints') 

# Dataloading
parser.add_argument('--train-path', default='', type=str, help='path to training data')
parser.add_argument('--sample-mode', default='uniform_sol', help='sampling output programs following a uniform distribution by program population')

# Model
parser.add_argument('--tuning_mode', default='plan', type=str, help='tuning mode for training LMs')
parser.add_argument('--clone_pl_head', default=True, action='store_true', help='Optional: clone a seperate linear layer for plan generation')


# Training
parser.add_argument('--epochs', default=20, type=int, help='total number of training epochs')
parser.add_argument('--lr', default=2e-5, type=float, help='training learning rate')
parser.add_argument('--batch-size-per-replica', default=2, type=int, help='batch size per GPU')
parser.add_argument('--grad-acc-steps', default=16, type=int, help='number of training steps before each gradient update')
parser.add_argument('--deepspeed', default = None, type=str, help='path to deepspeed configuration file; set None if not using deepspeed')
parser.add_argument('--fp16', default=True, action='store_true', help='set 16-bit training to reduce memory usage')
parser.add_argument('--local_rank', default=-1, type=int)
parser.add_argument('--db', default=False, action='store_true', help='set to turn on debug mode i.e. using dummy small data split and only 1 data worker')

# Logging
parser.add_argument('--log-freq', default=10, type=int, help='save training log after this number of training steps')
parser.add_argument('--save-freq', default=3580, type=int, help='save model checkpoints after this number of training steps')
parser.add_argument('--save_total_limit', default=1, type=int, help='total of number checkpoints to keep; only keep the latest ones') 

args = parser.parse_args()


