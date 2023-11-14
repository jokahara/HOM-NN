
import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from hommani.datasets import load_data, save_pickled_dataset

def add_to_test(dspath, files):
    print(dspath)
    print(files)
    return

if __name__ == '__main__':
    if len(sys.argv) > 1:
        add_to_test(sys.argv[1], sys.argv[2:])