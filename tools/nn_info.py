import os, sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))

from hommani.datasets import DataContainer

def print_info(paths):
    pickled_dataset_path = paths[0]
    f = []
    tr = [] 
    k = [] 
    te = [] 
    e = []

    for pickled_dataset_path in paths:
        f.append(pickled_dataset_path.split('/')[-1].split('.')[0])
        dc = DataContainer.load_data(pickled_dataset_path, autosave=False)
        
        print(dc.train)
        if dc.train == None:
            tr.append(0)
        else:
            tr.append(len(dc.train))
        
        if dc.kfold == None:
            k.append(1)
        else:
            k.append(len(dc.kfold))

        if dc.test == None:
            te.append(0)
        else:
            te.append(len(dc.test))

        if dc.energy_shifter == None:
            e.append(None)
        else:
            e.append(dc.energy_shifter.self_energies)
    
    print('')
    print(f'file           |    train | k |     test | energy shifter')
    print('----------------------------------------------------------')
    
    se = dc.energy_shifter.self_energies
    for i in range(len(paths)):
        match = 'x'
        if e[i] == None:
            match = '-'
        elif False in (se == e[i]):
            match = 'DID NOT MATCH'
        print(f'{f[i]:15}| {tr[i]:8} |{k[i]:2} | {te[i]:8} | {match}')

    print('----------------------------------------------------------')
    print('Self energies: ', list(se.numpy()))
    print('')

    return

if __name__ == '__main__':
    if len(sys.argv) > 1:
        print_info(sys.argv[1:])