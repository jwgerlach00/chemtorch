from typing import Iterable
from torch.utils.data import Datset


class SmilesDataset(Dataset):
    def __init__(self, smiles:Iterable, targets:Iterable, vocab_table:str, pad_len:int):
        self.smiles = smiles
        self.targets = targets
        self.vocab_table = vocab_table
        self.pad_len = pad_len

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        input = self.smiles[idx]
        target = self.targets[idx]

        return input, target, self.vocab_table, self.pad_len