""" Extract and read CIFAR-10 dataset

Dataset downloaded from: https://www.cs.toronto.edu/~kriz/cifar.html

"""
import tarfile
import pickle


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
