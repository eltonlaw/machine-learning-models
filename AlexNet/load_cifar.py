""" Extract and read CIFAR-10 dataset

Dataset downloaded from: https://www.cs.toronto.edu/~kriz/cifar.html

"""
import tarfile
import pickle
import numpy as np


def extract(tarpath):
    if not tarfile.is_tarfile(tarpath):
        raise Exception("'{}' is not a tarfile".format(tarpath))
    with tarfile.open(name=tarpath) as f:
        members = f.getmembers()
        files = []
        for mem in members:
            if mem.isfile():
                files.append(mem)
                continue
            if mem.isdir():
                pass
        f.extractall(path="./", members=files)
        data_path = [f.name for f in files if "_batch" in f.name]
        data_path.sort()
    return data_path


def load_batch(path):
    with open(path, 'rb') as f:
        data = pickle.load(f, encoding='bytes')
    images = data[list(data.keys())[2]]
    labels = data[list(data.keys())[1]]
    return images, labels


def load_meta(path="./cifar-10-batches-py/batches.meta"):
    with open(path, 'rb') as f:
        meta = pickle.load(f, encoding='bytes')
    labels_bytes = meta[list(meta.keys())[1]]
    labels = [l.decode("utf-8") for l in labels_bytes]
    return labels


def one_hot_encode(vector, n_unique):
    """ One hot encode a 1D vector

    PARAMETERS
    ----------
    vector: 1D vector. shape: (?, )
    n_unique: Total number of unique values in array

    RETURNS
    -------
    Each label is now a vector with 0's everywhere except a 1 at i=label
    Shape:(?, n_label)
    """
    one_hot_matrix = np.zeros((np.shape(vector)[0], n_unique))
    for i, y in enumerate(vector):
        one_hot_matrix[i, y] = 1
    return one_hot_matrix
