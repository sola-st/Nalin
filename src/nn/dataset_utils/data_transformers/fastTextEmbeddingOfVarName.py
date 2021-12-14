"""

Created on 23-June-2020
@author Jibesh Patra


"""

import fasttext
from typing import Dict


class fastTextEmbeddingOfVarName:
    def __init__(self, embedding_path: str):
        self.embedding = fasttext.load_model(path=embedding_path)
        self.embedding_dim = self.embedding.get_dimension()

    def __call__(self, sample:Dict)->Dict:
        sample['var'] = self.embedding[sample['var']]
        return sample
