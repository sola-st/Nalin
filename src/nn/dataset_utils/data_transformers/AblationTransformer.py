"""

Created on 13-June-2020
@author Jibesh Patra


"""
from typing import Dict, List
import torch


class AblationTransformer:
    def __init__(self, features_to_ablate=None) -> None:
        if features_to_ablate is None:
            # Example: ['type'] OR ['type', 'value']
            features_to_ablate = []
        self.features_to_ablate = features_to_ablate

    def __call__(self, sample: Dict) -> Dict:
        if len(self.features_to_ablate):
            for feature in self.features_to_ablate:
                if feature in sample:
                    if isinstance(sample[feature], torch.Tensor):
                        size = sample[feature].size()
                    else:
                        size = len(sample[feature])
                    sample[feature] = torch.zeros(size)
                else:
                    print(f'WARNING: The feature "{feature}" to ablate does not exist in sample')
        return sample
