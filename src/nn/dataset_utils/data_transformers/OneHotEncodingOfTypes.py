"""

Created on 11-May-2020
@author Jibesh Patra


"""

from typing import Dict, List
import torch
import json


class OneHotEncodingOfType:
    def __init__(self, max_types_to_select: int, types_in_dataset_file_path: str) -> None:
        """
        Create the one-hot encoding for the types in the dataset.

        :param max_types_to_select: The maximum number of types to consider in the dataset.
                                    It does not mean, we discard the remaining types. We
                                    create one-hot encoding of types for the most popular
                                    types and mark the remaining as 'other' type.
        :param types_in_dataset_file_path: A path to a file that contains types in the
                                    dataset and the corresponding frequency.
        """
        types_to_idx = {}

        with open(types_in_dataset_file_path) as f:
            types_and_frequencies = json.load(f)
        types_and_frequencies_sorted = dict(sorted(types_and_frequencies.items(), key=lambda x: x[1], reverse=True))
        selected_types = [tp for tp, freq in list(types_and_frequencies_sorted.items())[:max_types_to_select]]

        one_hot_vector_size = max_types_to_select
        if len(selected_types) < max_types_to_select:
            one_hot_vector_size = len(selected_types)

        for tp in selected_types:
            types_to_idx[tp] = len(types_to_idx)
        self.types_to_idx = types_to_idx
        # One extra for encoding types not present in the dataset
        self.one_hot_init = [0] * (one_hot_vector_size + 1)

    def __call__(self, sample: Dict) -> Dict:
        one_hot_encoded_type = list(self.one_hot_init)
        typ = sample['type']

        if typ not in self.types_to_idx:
            idx = -1
        else:
            idx = self.types_to_idx[typ]

        one_hot_encoded_type[idx] = 1
        sample['type'] = torch.tensor(one_hot_encoded_type).float()
        return sample
