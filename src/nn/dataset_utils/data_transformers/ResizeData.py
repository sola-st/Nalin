"""

Created on 11-May-2020
@author Jibesh Patra


"""
from typing import Dict

PAD_CHAR = ' '


class ResizeData:
    def __init__(self, len_of_value: int) -> None:
        self.len_of_value = int(len_of_value)

    def __call__(self, sample: Dict) -> Dict:
        # ----- Resize value ----
        if not isinstance(sample['value'], str):
            sample['value'] = str(sample['value'])
        # Value of the variable
        if len(sample['value']) > self.len_of_value:
            sample['value'] = sample['value'][:self.len_of_value // 2] + sample['value'][-self.len_of_value // 2:]

        len_of_sample_value = len(sample['value'])

        # mid_char = ''
        # if len(sample['value']) > len_of_sample_value // 2:
        #     mid_char: str = sample['value'][len_of_sample_value // 2]
        #
        # mid_char_next = ''
        # if len_of_sample_value > len_of_sample_value // 2 + 1:
        #     mid_char_next: str = sample['value'][len_of_sample_value // 2 + 1]
        #
        # # If the middle character is an integer
        # if mid_char.isdigit() or (mid_char_next != '' and mid_char_next.isdigit()):
        #     PAD_CHAR = '0'

        while len(sample['value']) < self.len_of_value:
            sample['value'] += PAD_CHAR
        sample['value'] = sample['value'][:self.len_of_value]
        return sample
