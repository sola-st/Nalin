"""

Created on 11-May-2020
@author Jibesh Patra


"""
from typing import Dict
import unicodedata
import string
import torch

all_letters = string.ascii_letters + \
              " .,;'0123456789,;.!?:'\"/\\|_@#$%^→&*~`+-=<>()[]{} "
nbs_chars = len(all_letters)


class ValueToCharSequence:
    def __init__(self, len_of_value: int):
        self.char_to_idx = {c: i for i, c in enumerate(all_letters)}
        self.nbs_chars = nbs_chars
        self.len_of_value = int(len_of_value)

    # Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
    @staticmethod
    def unicodeToAscii(s):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn'
            and c in all_letters
        )

    def str_to_one_hot(self, val_as_str: str) -> torch.Tensor:
        token_seq_as_tensor = torch.zeros(
            self.len_of_value, 1, nbs_chars)
        for id, char in enumerate(val_as_str):
            if char in self.char_to_idx:
                index = self.char_to_idx[char]
                token_seq_as_tensor[id][0][index] = 1
        token_seq_as_tensor = torch.squeeze(token_seq_as_tensor)
        return token_seq_as_tensor

    def __call__(self, sample: Dict) -> Dict:
        s = sample['value']
        s = ValueToCharSequence.unicodeToAscii(s)
        # idx_list = self.str_to_char_indices(s)
        # while len(idx_list) < self.len_of_value:
        #     idx_list.append(self.char_to_idx[" "])
        # Now convert to one-hot vectors

        sample['value_as_one_hot'] = self.str_to_one_hot(val_as_str=s)
        return sample
