import numpy as np
import torch
from torch.utils.data import Dataset
class ActionDataset(Dataset):
    def __init__(self,path):
        d=np.load(path); self.actions=torch.from_numpy(d['actions']).float(); self.conditions=torch.from_numpy(d['conditions']).float() if 'conditions' in d else torch.zeros(len(self.actions),256)
    def __len__(self): return len(self.actions)
    def __getitem__(self,i): return {'actions':self.actions[i],'condition':self.conditions[i]}
