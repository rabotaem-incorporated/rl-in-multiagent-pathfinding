import os
import sys
import numpy as np

sys.path.append(os.getcwd() + '/scrimp_reference/od_mstar3')
from od_mstar import find_path
import re
import pickle

def run_one(args):
    mp, axy, txy = args
    try:
        res = find_path(mp.tolist(), axy.tolist(), txy.tolist())
        return True, len(res), None, None, None
    except Exception:
        return False, 0, None, None, None
        

if __name__ == '__main__':
    for filename in os.listdir('./tests/test_set'):
        if filename[1] == 'x' and filename.endswith('30.pth'):
            print(filename)
            data = pickle.load(open(f'./tests/test_set/{filename}', 'rb'))
            a, b, c, d, e = zip(*map(run_one, data))
            pickle.dump((a, b, c, d, e), open(f'./tests/results_odrm_star/{filename}', 'wb'))
