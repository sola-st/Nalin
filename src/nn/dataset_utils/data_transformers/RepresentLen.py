"""

Created on 29-June-2020
@author Jibesh Patra


"""
from typing import Dict
import torch


class RepresentLen:
    def __init__(self):
        self.max_len = 1000
        self.num_bins = 100
        self.bin_spacing = self.max_len // self.num_bins

    def __call__(self, sample: Dict) -> Dict:
        one_hot = [0] * self.num_bins
        if sample['len'] >= self.max_len:
            idx = -1
        else:
            idx = int(sample['len']) // self.bin_spacing
        one_hot[idx] = 1
        sample['len'] = torch.tensor(one_hot).float()
        return sample
