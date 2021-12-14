"""

Created on 04-May-2020
@author Jibesh Patra

Given a dataset, enrich it with -ve examples.

"""
from pathlib import Path
from multiprocessing import Pool, cpu_count, get_context
from tqdm import tqdm
import fileutils as fs
from typing import List, DefaultDict, Tuple, Dict, Set
import numpy as np
import random
import os
import pandas as pd
import time
import json

SEED = 100
random.seed(SEED)
# data_set_pos: pd.DataFrame = pd.DataFrame([])
# types_freq_in_dataset = {}
# value_freq_in_dataset = {}


def read_dataset(in_dir: str) -> List:
    extracted_data_files = list(Path(in_dir).glob('**/*.json'))
    d = []
    with Pool(cpu_count()) as p:
        with tqdm(total=len(extracted_data_files)) as pbar:
            pbar.set_description_str(desc="Reading files", refresh=False)
            for i, each_vars in enumerate(p.imap_unordered(fs.read_json_file, extracted_data_files)):
                pbar.update()
                d.append(each_vars)
            p.close()
            p.join()
    return d


def get_statistics_given_dimension(dataset: pd.DataFrame):
    """
    The dataset can be either a single assignment or
    :param dataset:
    :return:
    """
    pass




def create_a_negative_example(assignment_and_columns) -> Dict:
    """
    Given an extracted variable, create a negative example from it
    :param assignment:
    :return:
    """
    assignment, columns_to_keep = assignment_and_columns
    negative_example = dict(assignment)  # create a copy
    var_name = negative_example['var']  # This remains in the negative example

    types_freq_var_has_been = dict(
        data_set_pos.loc[data_set_pos['var'] == var_name]['type'].value_counts())
    types_the_var_has_never_been = set(
        types_freq_in_dataset.keys()) - set(types_freq_var_has_been.keys())
    selected_type = None
    """
    Strategy  -->
    A particular variable 'Z' has been types: {'ndarray', 'str', 'list', 'matrix', 'MutableDenseMatrix'}
    It has never been types: {'int', 'float'}

    Now, to create a negative example, we would select from the types 'Z' has never been. But, it is
    possible that 'Z' only occurs very few times as 'str', may be just once. As a result, we add 'str'
    to types it has never been. 

    In short, create a negative example based on the infrequent types a variable has been assigned to 
    in the dataset. The infrequent types selection is based on a configurable parameter 'K' that represents
    the infrequent percentage.
    """
    K = 0.03  # 3%
    infrequent_types = set()
    total_freq = np.sum(list(types_freq_var_has_been.values()))
    for tp, freq in types_freq_var_has_been.items():
        if round((freq / total_freq), 2) < K:
            infrequent_types.add(tp)
    # Also put very unlikely types that occurs less than K% of the cases
    types_the_var_has_never_been.update(infrequent_types)

    types_to_select_from = []
    frequencies_of_selectable_types = []
    for tp in types_the_var_has_never_been:
        types_to_select_from.append(tp)
        frequencies_of_selectable_types.append(types_freq_in_dataset[tp])
    if len(types_to_select_from):
        selected_type = random.choices(
            population=types_to_select_from, weights=frequencies_of_selectable_types)[0]
    else:
        raise Exception("No type found to select")
    created_negative_data = dict(
        data_set_pos.loc[data_set_pos['type'] == selected_type].sample(1)[columns_to_keep].iloc[0])

    assert selected_type is not None, "Type selected during creation of negative example, can't be None"
    assert len(created_negative_data) != 0, "The created data must not be empty"

    negative_example['value'] = created_negative_data['value']
    negative_example['type'] = created_negative_data['type']
    negative_example['len'] = created_negative_data['len']
    negative_example['shape'] = created_negative_data['shape']
    negative_example['p_buggy'] = 1.0

    return negative_example



def save_and_create_negative_examples_given_specific_type(dataset: pd.DataFrame, given_type: str) -> None:
    """
    Given a cleaned dataset and a specific type:
        - Save positive examples for the type
        - Save negative examples for the type
    :param dataset:
    :param given_type: The type could be 'str', 'int', 'list' etc.
    :return:
    """
    p_only = dataset.loc[dataset['type'] == given_type]
    p_only = p_only.reset_index(drop=True)
    p_only.to_pickle('results/positive_examples_' +
                     given_type + '.pkl', compression='gzip')
    n_only = create_negative_examples_by_randomizing_values(p_only, ['value', 'len', 'shape', 'type'])
    n_only.to_pickle('results/negative_examples_' +
                     given_type + '.pkl', compression='gzip')


def create_negative_examples_by_interchanging_type_values(positive_examples_dataset: pd.DataFrame,
                                                          types: List) -> pd.DataFrame:
    """
    Given a positive_examples_dataset and say, we have two types 'int' and 'float'. Create negative examples
    by interchanging values of type 'int' with 'float' and vice versa instead of random shuffling

    :param types:
    :param positive_examples_dataset:
    :return:
    """
    assert len(
        types) == 2, "The types should be exactly two eg. ['int', 'float'] "
    print('Creating negative examples by interchanging integer and float values. \nMake Sure to filter postive '
          'examples for only these types \nelse the number of -ve examples will be unbalanced')
    positive_examples_dataset = positive_examples_dataset.reset_index(
        drop=True)

    first_type, second_type = types[0], types[1]  # eg. 'int', 'float'

    # Select only the data belonging to the first and second types
    only_first_type_dataset = positive_examples_dataset[positive_examples_dataset['type'] == first_type].reset_index(
        drop=True)
    num_of_first_types = len(only_first_type_dataset)

    only_second_type_dataset = positive_examples_dataset[positive_examples_dataset['type'] == second_type].reset_index(
        drop=True)
    num_of_second_types = len(only_second_type_dataset)

    values_first_type = list(only_first_type_dataset['value'])
    random.shuffle(values_first_type)

    values_second_type = list(only_second_type_dataset['value'])
    random.shuffle(values_second_type)

    # Create a balanced dataset
    while len(values_second_type) < num_of_first_types:
        values_second_type.append(random.choice(values_second_type))

    while len(values_first_type) < num_of_second_types:
        values_first_type.append(random.choice(values_first_type))

    values_first_type = values_first_type[:num_of_second_types]
    values_second_type = values_second_type[:num_of_first_types]

    only_first_type_dataset['value'] = values_second_type
    only_first_type_dataset['type'] = second_type

    only_second_type_dataset['value'] = values_first_type
    only_second_type_dataset['type'] = first_type

    new_dataset = pd.concat(
        (only_first_type_dataset, only_second_type_dataset), ignore_index=True)
    new_dataset['p_buggy'] = 1.0
    return new_dataset


def create_negative_examples_by_randomizing_values(positive_examples_dataset: pd.DataFrame,
                                                   columns_to_keep: List) -> pd.DataFrame:
    """
    This works if there exists only 'ONE' type in the entire dataset. 
    Select a 'particular' type and create negative examples for only this type.

    Eg. If we select the type as 'float'
        a = 2.42
        b = -23.0
        When creating a negative example, we swap only among float values

        so, a becomes -23.0
    """
    negative_examples_dataset = positive_examples_dataset.reset_index(
        drop=True)
    shuffled_data = negative_examples_dataset[columns_to_keep].sample(frac=1, random_state=SEED).reset_index(drop=True)
    negative_examples_dataset[columns_to_keep] = shuffled_data
    return negative_examples_dataset


def create_negative_examples_from_data(positive_examples_dataset: pd.DataFrame, columns_to_keep: List,
                                       way: str = 'weighted_random') -> pd.DataFrame:
    if way == 'random':
        print("Creating negative examples by randomizing the values")
        return create_negative_examples_by_randomizing_values(positive_examples_dataset, columns_to_keep)
    else:  # default
        global data_set_pos
        global types_freq_in_dataset
        global value_freq_in_dataset
        types_freq_in_dataset = dict(
            positive_examples_dataset['type'].value_counts())
        value_freq_in_dataset = dict(
            positive_examples_dataset['value'].value_counts())
        data_set_pos = positive_examples_dataset

        # TODO: pre-compute more things Eg. types_freq_var_has_been, types_freq_var_has_never_been etc.
        data_samples = ((row, columns_to_keep) for _, row in data_set_pos.iterrows())
        negative_examples = []
        with get_context("fork").Pool(cpu_count()) as p:
            with tqdm(total=len(data_set_pos)) as pbar:
                pbar.set_description_str(
                    desc="Creating negative examples ", refresh=False)
                for _, neg_example in enumerate(
                        p.imap(create_a_negative_example, data_samples,chunksize=2)):
                    negative_examples.append(neg_example)
                    pbar.update()
                p.close()
                p.join()
        # for data_sample in tqdm(data_samples, desc="Creating negative"):
        #     negative_examples.append(create_a_negative_example(data_sample))
        return pd.DataFrame(negative_examples)



if __name__ == '__main__':
    import sys
    sys.path.extend(['src/nn'])
    print("Reading file")
    df = pd.read_pickle(
        filepath_or_buffer='results/positive_examples.pkl',
        compression='gzip')
    df.reset_index(drop=True, inplace=True)
    print(f"Positive examples size is {len(df)}")
    ndf = create_negative_examples_from_data(positive_examples_dataset=df, columns_to_keep = ['value', 'len', 'shape', 'type'])
    ndf.to_pickle('results/negative_examples.pkl','gzip')
