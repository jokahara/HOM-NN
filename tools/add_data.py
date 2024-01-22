
import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from hommani.datasets import DataContainer

def add_to_test(pickled_dataset_path, files):
    
    pickled_data = DataContainer.load_data(pickled_dataset_path)
    datasets = [pickled_data]
    for f in files:
        if not os.path.isfile(f):
            print('Error! File '+f+' not found')
        d = DataContainer.load_data(f, train_test_split=0, 
                                    energy_shifter=pickled_data.energy_shifter, 
                                    autosave=False)
        datasets.append(d)

    merged_data = DataContainer.merge(datasets)
    merged_data.save(pickled_dataset_path)

    print('Added '+str(len(merged_data.test)-len(pickled_data.test))+' datapoints to test set.')
    return

if __name__ == '__main__':
    add_to = None
    new_data = []
    if len(sys.argv) > 1:
        for arg in sys.argv[1:]:
            if arg == '-to':
                add_to = True
                continue
            if add_to == True:
                add_to = arg
                continue
            new_data.append(arg)
    
    if add_to[-7:] == '.nn.pkl' and os.path.isfile(add_to):
        add_to_test(add_to, new_data)