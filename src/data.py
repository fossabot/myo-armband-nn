import os
import numpy as np

from common import *
from pathlib import Path

def get_data_set(location):
    filenames = []
    for root, dirs, files in os.walk(location):
        for filename in files:
            if filename.endswith(('csv')):
                print(filename)
                filenames.append(location + filename)
    merged = []
    for fname in filenames:
        fileLoad = np.loadtxt(open(Path(fname), "rb"), delimiter=",")
        merged.append(fileLoad)
    merged = np.concatenate(merged)
    x = merged[:,:64]
    a_class = merged[:,64]
    print(a_class.size)
    y = np.zeros((a_class.size, get_num_classes()), dtype=np.int)
    index = 0
    for i in np.nditer(a_class):
        y[index, int(i)] = 1
        index+=1
    return x, y
