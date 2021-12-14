"""

Created on 04-May-2020
@author Jibesh Patra


"""
from torch.utils.data import Dataset
from torchvision.transforms import Compose
import pandas as pd


class DynamicAnalysisDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame, transform: Compose) -> None:
        self.data = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        it = dict(self.data.iloc[idx])
        # it = read_json_file(self.files[idx_label])
        if self.transform:
            return self.transform(it)
        else:
            return it


if __name__ == '__main__':
    # from pathlib import Path
    #
    # in_dir = 'results/dynamic_analysis_outputs'
    # l_of_files = list(Path(in_dir).glob('**/*.json'))
    # d = DynamicAnalysisDataset(files=l_of_files)
    # print(d[1])
    pass
