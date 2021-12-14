"""

Created on 24-September-2020
@author Jibesh Patra


"""

from typing import Dict
import torch


class RepresentShape:
    def __init__(self):
        self.max_shape = 100000
        self.num_bins = 100
        self.bin_spacing = self.max_shape // self.num_bins

    def __call__(self, sample: Dict) -> Dict:
        one_hot = [0] * (self.num_bins + 1)
        # Shape is either a tuple or -1 . -1 means shape could not be obtained. Eg. lists
        if sample['shape'] == -1:
            idx = -1
        else:
            m = 1
            try:
                for s in sample['shape']:
                    m *= s
                idx = m // self.bin_spacing
                if idx > 99:
                    idx = 99
            except Exception as e:
                idx = -1
        one_hot[idx] = 1
        sample['shape'] = torch.tensor(one_hot).float()
        return sample
