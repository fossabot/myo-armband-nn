import os
import numpy as np

_NUM_CLASSES = 6

def load_scvs(location="./../data/"):
    filenames = []
    for root, dirs, files in os.walk(location):
        for filename in files:
            if filename.endswith(('csv')):
                print(filename)
                filenames.append(location + filename)
    merged = []
    for fname in filenames:
        fileLoad = np.loadtxt(open(fname, "rb"), delimiter=",")
        merged.append(fileLoad)
    merged = np.concatenate(merged)
    x = merged[:,:64]
    a_class = merged[:,64]
    print(a_class.size)
    y = np.zeros((a_class.size, _NUM_CLASSES), dtype=np.int)
    index = 0
    for i in np.nditer(a_class):
        y[index, int(i)] = 1
        index+=1
    return x, y

def get_data_set(name="bymotion"):
    x = None
    y = None

    if name is "train":
        npzfile = np.load("./data/train_set.npz")
        x = npzfile['x']
        y = npzfile['y']
    elif name is "test":
        npzfile = np.load("./data_set/test_set.npz")
        x = npzfile['x']
        y = npzfile['y']
    elif name is "bymotion":
        x, y = load_scvs()
    return x, y
