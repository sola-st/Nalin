"""

Created on 02-June-2020
@author Jibesh Patra


"""
from multiprocessing.spawn import freeze_support

from torchvision.transforms import Compose
from DynamicAnalysisDataset import DynamicAnalysisDataset
from torch.utils.data import random_split
import random
from pathlib import Path
from typing import Tuple, List
from fileutils import create_dir_list_if_not_present
import pandas as pd
import torch
import json
import os
from dataset_utils.pre_process_dataset import read_dataset_given_files

SEED = 42
torch.manual_seed(SEED)
random.seed(SEED)


def get_training_val_dataset(positive_examples_dataset_file_path: str,
                             negative_examples_dataset_file_path: str,
                             all_transformations: List,
                             nb_examples: int = -1) -> Tuple[DynamicAnalysisDataset, DynamicAnalysisDataset]:
    if not Path(positive_examples_dataset_file_path).is_file() or not Path(
            negative_examples_dataset_file_path).is_file():
        print(
            f"Either or both of '{positive_examples_dataset_file_path}', '{negative_examples_dataset_file_path}' does not exist, can't train")
        import sys
        sys.exit(1)

    positive_example_dataset = pd.read_pickle(
        filepath_or_buffer=positive_examples_dataset_file_path, compression='gzip')
    if nb_examples > 0:
        positive_example_dataset = positive_example_dataset[:nb_examples // 2]

    negative_example_dataset = pd.read_pickle(filepath_or_buffer=negative_examples_dataset_file_path,
                                              compression='gzip')
    if nb_examples > 0:
        negative_example_dataset = negative_example_dataset[:nb_examples // 2]

    positive_example_dataset.reset_index(drop=True, inplace=True)
    negative_example_dataset.reset_index(drop=True, inplace=True)

    # Concat and alternate the positive and negative examples
    dataset = pd.concat([positive_example_dataset, negative_example_dataset]).sort_index(kind='merge')
    dataset.reset_index(drop=True, inplace=True)
    
    # dataset = dataset.sample(frac=1, random_state=SEED)  # random shuffle
    # dataset.reset_index(drop=True, inplace=True)

    print('Few values from the dataset -->')
    # Special case for priniting in a jupyter notebook environment
    try:
        cfg = get_ipython().config
        from IPython import display
        display.display(dataset)
    except NameError:
        print(dataset[:10])

    num_postive_examples = len(dataset.loc[dataset['p_buggy'] == 1.])
    num_negative_examples = len(dataset.loc[dataset['p_buggy'] == 0.])
    print(
        f"\nUsing {len(dataset)} examples for training and validation which contains {num_postive_examples} positive examples & {num_negative_examples} negative examples\n")

    transform = Compose(all_transformations)
    assignment_dataset = DynamicAnalysisDataset(
        dataset=dataset, transform=transform)

    len_of_dataset = len(assignment_dataset)
    tr_size = int(len_of_dataset * 0.8)  # 80% training data
    vl_size = len_of_dataset - tr_size
    train, validation = random_split(assignment_dataset, [tr_size, vl_size])
    return train, validation


def get_test_dataset(test_examples_dir: str, results_dir: str, all_transformations: List,
                     dataset_out_file: str = 'results/test_dataset.pkl',
                     size_of_context: int = 10) -> DynamicAnalysisDataset:
    create_dir_list_if_not_present([results_dir])

    if Path(dataset_out_file).is_file():
        print(f"Reading '{dataset_out_file}'")
        test_dataset = pd.read_pickle(dataset_out_file, compression='gzip')
    else:
        extracted_data_files = list(Path(test_examples_dir).glob('**/*.json'))
        test_dataset = read_dataset_given_files(extracted_data_files=extracted_data_files)
        test_dataset.dropna(inplace=True)

        print(f"Saving dataset as '{dataset_out_file}'")
        test_dataset.to_pickle(path=dataset_out_file, compression='gzip')

    # test_dataset = discard_and_merge_types(dataset=test_dataset)
    if 'idf_list_before' in test_dataset:
        test_dataset['idf_list_before'] = test_dataset['idf_list_before'].apply(lambda x: x[:size_of_context])
    if 'idf_list_after' in test_dataset:
        test_dataset['idf_list_after'] = test_dataset['idf_list_after'].apply(lambda x: x[:size_of_context])

    print(f"\nThe dataset contains {len(test_dataset)} examples for testing")
    print('-' * 70)
    print('A sample of the dataset  -->')
    try:
        cfg = get_ipython().config
        from IPython import display
        display.display(test_dataset)
    except NameError:
        print(test_dataset.sample(n=5))
    print('-' * 70)

    transform = Compose(all_transformations)
    dataset = DynamicAnalysisDataset(
        dataset=test_dataset, transform=transform)
    return dataset
