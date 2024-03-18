
import argparse

parser = argparse.ArgumentParser(description="Testing generated programs with unit testing")

parser.add_argument("-t","--test_path", default="", type=str, help="Path to test samples")
parser.add_argument("--output_path", default="",type=str, help="Path to output test results")
parser.add_argument("--code_path", default="",type=str, help='Path to generated programs') 

parser.add_argument("-i", "--index", default=0, type=int, help='specific sample index to be tested against unit tests')
parser.add_argument("-d", "--debug", default =False, help='test in debugging mode with printout messages')
parser.add_argument('--max_tests', type=int, default=-1, help='Filter for test samples by maximum number of unit tests') 
parser.add_argument('--example_tests', type=int, default=1, help='0: run hidden unit tests; 1: run example unit tests')

args = parser.parse_args()

