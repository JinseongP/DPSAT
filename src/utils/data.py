import torch
from torch.utils.data import TensorDataset, DataLoader

def get_subloader(given_loader, n_limit):
    if n_limit is None:
        return given_loader

    sub_loader = []
    num = 0
    for item in given_loader:
        sub_loader.append(item)
        if isinstance(item, tuple) or isinstance(item, list):
            batch_size = len(item[0])
        else:
            batch_size = len(item)
        num += batch_size
        if num >= n_limit:
            break
    return sub_loader

class FixedLoader:
    def __init__(self, path, **kwargs):
        self.path = path
        self.kwargs = kwargs
        self.iteration = -1
        self.loader = self.load(0)
        
    def __iter__(self):
        self.iteration += 1
        self.loader = self.load(self.iteration)
        return self.loader.__iter__()

    def __len__(self):
        return self.loader.__len__()

    def __next__(self):
        return self.loader.__next__()
    
    def reset(self):
        self.iteration = -1
        self.loader = self.load(0)
    
    def load(self, iteration):
        data = torch.load(self.path+"/%d.pt"%iteration)
        images, labels = data
        images = images.float()/255
        labels = labels.long()
        dataset = TensorDataset(images, labels)
        return DataLoader(dataset, **self.kwargs)
