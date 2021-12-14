"""

Created on 01-June-2020
@author Jibesh Patra

* Call the creation of negative examples and also create_neg labels (buggy/non-buggy) for data.

* Additionally, clean the dataset by removing noise & select some parts of dataset based on
  different heuristics e.g.,
    - 'select only the top-n most common types'
    - 'select only list type data' etc.
"""

import pickle
from collections.abc import Iterable
from multiprocessing import Pool, cpu_count, get_context
from pathlib import Path
from typing import List, Dict, Any, Tuple
import codecs
import numpy as np
import pandas as pd
from fileutils import read_json_file
from tqdm import tqdm
import json
from sklearn.model_selection import KFold
import shutil
import os

from dataset_utils import create_negative_examples as create_neg
from dataset_utils.types_processing import process_types

MAX_VALUE_SIZE = 200
min_var_name_len = 4


def get_size_and_resize(data: Dict) -> Tuple[int, int, Any]:
    """
    Given 'val' read from a pickled file return the size and also resize if
    it exceeds given size.

    :param data:
    :return:
    """
    value = data['value']

    # if not isinstance(value, str):
    #     return 0, 0, None
    # else:
    #     return len(value), -1, value[:MAX_VALUE_SIZE]

    # len() does not support every datatype.
    length = -1
    size = -1
    trimmed_val = None

    if hasattr(value, 'shape'):
        size = value.shape
    elif hasattr(value, 'size'):
        size = value.size

    if hasattr(value, '__len__'):
        length = len(value)

    if isinstance(value, Iterable):
        # Examples of iterables ->  str, dict, tuple, set, numpy array, pandas dataframe, range
        # Iterables we can't trim directly -> dict, set
        if isinstance(value, dict):
            if len(value) > MAX_VALUE_SIZE:
                trimmed_val = {k: v for k, v in value.items()[:MAX_VALUE_SIZE]}
            else:
                trimmed_val = value
        elif isinstance(value, set):
            if len(value) > MAX_VALUE_SIZE:
                trimmed_val = set(list(value)[:MAX_VALUE_SIZE])
            else:
                trimmed_val = value
        elif isinstance(value, range):
            # special case for range
            value = value[:MAX_VALUE_SIZE]
            trimmed_val = list(value)
        else:
            try:
                # Rest of the iterables can be (hopefully) sliced directly
                if isinstance(size, tuple) and len(size) > 1 and isinstance(value, np.ndarray):
                    # trim some elements from every dimension
                    l = [slice(MAX_VALUE_SIZE)] * value.ndim
                    trimmed_val = value[tuple(l)]
                else:
                    trimmed_val = value[:MAX_VALUE_SIZE]
            except Exception as e:
                # print(f"Can't trim values of {type(value)}", e)
                trimmed_val = value
    elif isinstance(value, float) or isinstance(value, int) or value is None:
        trimmed_val = value
    else:
        pass

    # trimmed_val = str(trimmed_val)

    # assert trimmed_val != 'UNKNWON', 'The trimmed value should not be unknown'

    return length, size, trimmed_val


def read_json_file_trim_value(file_path: str) -> Dict:
    data = read_json_file(file_path)
    # Number of characters in
    if 'value' in data and len(data['value']) > MAX_VALUE_SIZE:
        data['value'] = data['value'][:MAX_VALUE_SIZE // 2] + \
                        data['value'][-MAX_VALUE_SIZE // 2:]
    return data


def read_pickle_trim_value(file_path: str) -> Tuple[Dict, str]:
    data = {}
    # t = time.localtime()
    # current_time = time.strftime("%H:%M:%S", t)
    # tqdm.write(f'Loading {file_path}, --> {current_time}')
    try:
        with codecs.open(file_path, mode='rb') as f:
            data = pickle.load(f)
            # ---- Filtering only certain types of values. Eg. list with only numbers etc.
            # if not is_value_matching_criterion(data['value']):
            #     return {}
            # -----------------------------------------------------------------------------
            if not data: return {}, file_path
            length, shape, trimmed_value = get_size_and_resize(data)
            if trimmed_value is None:
                return {}, file_path
            if data['len'] < 0:
                data['len'] = length
            data['shape'] = shape
            data['value'] = str(trimmed_value)
            return data, file_path
    except Exception as e:
        data = {}
        # print('Error reading pickle ', e)
    return {}, file_path


def read_dataset_given_files(extracted_data_files: List, file_extension: str = '.pickle') -> pd.DataFrame:
    d = []
    if file_extension == '.pickle':
        read_file_trim_value = read_pickle_trim_value
    else:  # json
        read_file_trim_value = read_json_file_trim_value

    if cpu_count() > 4:
        parallel_processes = cpu_count()
    else:
        parallel_processes = 1
    print(f'Number of files are {len(extracted_data_files)}')
    still_to_extract = set(extracted_data_files)
    if parallel_processes == 1:
        for file_path in tqdm(extracted_data_files, desc='Reading dataset from files'):
            each_vars = read_file_trim_value(file_path)
            d.append(each_vars)
    else:
        with get_context("fork").Pool(parallel_processes) as p:
            with tqdm(total=len(extracted_data_files)) as pbar:
                pbar.set_description_str(
                    desc="Reading dataset from files", refresh=False)
                for i, out in enumerate(
                        p.imap_unordered(read_file_trim_value, extracted_data_files)):
                    pbar.update()
                    each_vars, file_path = out
                    # ---- Workaround for cases where the extraction from pickle files get stuck ----
                    # still_to_extract.discard(file_path)
                    # if len(still_to_extract) < 10:
                    #     extracted_dataset = pd.DataFrame(d)
                    #     extracted_dataset.dropna(inplace=True)
                    #     extracted_dataset['type'] = extracted_dataset['type'].apply(process_types)
                    #     return extracted_dataset
                    if len(each_vars):
                        d.append(each_vars)
                p.close()
                p.join()
    extracted_dataset = pd.DataFrame(d)
    extracted_dataset.dropna(inplace=True)
    print(f"Processing 'type' field of the dataset")
    if len(extracted_dataset) >= 1:
        extracted_dataset['type'] = extracted_dataset['type'].apply(process_types)
    return extracted_dataset


# ---------------------------------- Types Processing --------------------------------------------

def select_only_particular_types(dataset: pd.DataFrame, types: List) -> pd.DataFrame:
    if len(types) < 1:
        return dataset
    types_to_retain = set(types)
    dataset = dataset.loc[dataset['type'].isin(types_to_retain)]
    if len(dataset):
        for tp in types:
            print(f"\tThe percent of {tp} is {round(len(dataset[dataset['type'] == tp]) / len(dataset) * 100, 2)}%")
    return dataset


def select_only_top_n_common_types(dataset: pd.DataFrame, n: int = 10) -> pd.DataFrame:
    """
    First find the most popular 'n' types. Remove any uncommon types from the
    dataset

    :param dataset: The complete dataset
    :param n: The number of top types to select
    :return: Return the dataframe once the top 'n' types has been removed
    """
    len_before_filtering = len(dataset)
    print(f'*** Selecting only the most common "{n}" types from the dataset. Current length is {len_before_filtering}')
    top_types = dataset['type'].value_counts()[:n].to_dict()
    dataset = dataset[dataset['type'].apply(lambda x: x in top_types)]
    len_after_filtering = len(dataset)
    print(
        f'Removed {len_before_filtering - len_after_filtering} elements, the current length of the dataset is {len_after_filtering}\n')
    return dataset


def discard_and_merge_types_select_most_common_n(dataset: pd.DataFrame, most_common_n: int = 20) -> pd.DataFrame:
    """
    It is possible that the negative example creation suffers due to a
    long tail distribution of types.

    - Can we merge some of the types?
        - Using name based matching? Eg.
            * 'frozenset' and 'set', although distinct types, actually have the
              same values and variable name
            * 'dict' and 'defaultdict'
            - Disadvantage of name based matching:
            * 'dictkeys' 'dictvalues' are similar to 'list' but using our naive
            name matching we will change it to 'dict'
        - Using embedding?
    - If the types can't be merged, delete them based on the frequency of occurrence.
      If a certain type is very infrequent, remove it.
    - Also tone down the most frequent type

    :param most_common_n:
    :param dataset: The positive example dataset
    :return: The dataset with the processing of the types done
    """
    # Find the frequency of the types
    type_frequency = dataset['type'].value_counts()

    # Select the most frequent K types
    top_K = 15
    most_frequent_types = dict(type_frequency[:top_K])

    # Merge some of the types as the frequent types using name matching
    # Eg. Search for 'dict' in 'defaultdict' and if it exists, replace 'defaultdict' as 'dict'
    dataset['orig_type'] = dataset['type']  # Keep a copy as original type
    for freq_tp in most_frequent_types:
        dataset['type'] = dataset['type'].apply(
            lambda tp: freq_tp if freq_tp != tp and freq_tp.lower() in tp.lower() else tp)

    # Not every type can be merged using
    # Also remove very infrequent types/take only the topK types
    K = 0.05
    # After merging find the type frequency again and select only top-K
    type_frequency = dict(dataset['type'].value_counts()[:most_common_n])
    types_to_take = set(type_frequency.keys())

    dataset = dataset.loc[dataset['type'].isin(types_to_take)]
    return dataset


# ---------------------------------- Values Processing --------------------------------------------

def does_the_list_contains_only_numbers(lst):
    try:
        _ = np.sum(np.array(lst))
        if hasattr(lst, '__len__') and len(lst) < 1:
            return False  # If empty list
        return True
    except Exception as e:
        return False


def is_value_matching_criterion(value: Any) -> bool:
    """
    Say we have a value and we want to filter based only certain
    types of values. eg.

    Select data that is iterable and contain only numeric type

    :param value:
    :return:
    """

    try:
        if isinstance(value, np.ndarray):
            return True
        elif isinstance(value, pd.DataFrame):
            one_row = value.iloc[0].to_list()
            return does_the_list_contains_only_numbers(one_row)
        elif isinstance(value, list):
            return does_the_list_contains_only_numbers(value)
    except Exception as e:
        return False


def is_valid_size(size) -> bool:
    if isinstance(size, int):
        return True
    else:
        try:
            a = len(size)
            return True
        except Exception as e:
            return False


def remove_noise_from_dataset(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Manually going through a sample of the dataset we encountered some instances
    where, either the value has not been extracted properly or the size is not correct eg. it's
    a method where we expect either an iterable or a int.

    Remove such instances.
    :param dataset:
    :return:
    """
    dataset = dataset[dataset['var'].apply(lambda x: len(x) < 20)]
    dataset = dataset[dataset['shape'].apply(is_valid_size)]
    return dataset


# ---------------------------------- Varname Processing --------------------------------------------

def is_var_name_with_greater_than_len_n(var_name: str) -> bool:
    """
    Given a variable name, return if this is acceptable according to the
    filtering heuristics.
    Here, we try to discard variable names like X, y, a, b etc.
    :param var_name:
    :return:
    """
    unacceptable_names = {}
    if len(var_name) < min_var_name_len:
        return False
    elif var_name in unacceptable_names:
        return False
    return True


def select_the_most_common_n_variable_names(dataset: pd.DataFrame, most_common_n: int = 200) -> pd.DataFrame:
    print(f"*** Selecting only the most common {most_common_n} variable names ***")
    most_common_names = dataset['var'].value_counts()[:most_common_n].index.to_list()

    selected_dataset = dataset.loc[dataset['var'].isin(most_common_names)]
    return selected_dataset


def get_balanced_dataset_given_two_type(dataset: pd.DataFrame, two_types: List) -> pd.DataFrame:
    """
    Given a dataset and two types select the create_neg a balanced dataset. The
    balanced dataset is created by finding the smaller of two types and
    truncating the larger one.

    :param dataset:
    :param two_types: List of two types as string Eg. ['int', 'float']
    :return:
    """
    assert len(two_types) == 2, 'There must be exactly two types'

    first_type, second_type = two_types[0], two_types[1]
    only_first_type_dataset = dataset[dataset['type'] == first_type]
    only_second_type_dataset = dataset[dataset['type'] == second_type]

    smaller_dataset_len = len(only_first_type_dataset)
    if len(only_second_type_dataset) < len(only_first_type_dataset):
        smaller_dataset_len = len(only_second_type_dataset)

    only_first_type_dataset = only_first_type_dataset[:smaller_dataset_len]
    only_second_type_dataset = only_second_type_dataset[:smaller_dataset_len]

    only_first_type_dataset.reset_index(drop=True, inplace=True)
    only_second_type_dataset.reset_index(drop=True, inplace=True)
    dataset = pd.concat((only_first_type_dataset, only_second_type_dataset), ignore_index=True)
    return dataset


def select_variable_names_based_on_heuristics(dataset: pd.DataFrame) -> pd.DataFrame:
    """
    Given certain heuristics for variable names, only select them.
    E.g., we might want to select only variable names that are not like d_1_2 etc.

    :param dataset:
    :return:
    """

    n_chars = 3

    def var_name_not_joined_by_small_words(var_name):
        # Many times the variable names are like 'ab_1c' which does not convey much semantic information.
        # Filter such type of variable names.
        # This is not a perfect heuristic because we will retain variable names such as 'abc_1c'.
        split_name = var_name.split('_')
        all_less_than_n_chars = True
        for nm in split_name:
            if len(nm) >= n_chars:
                all_less_than_n_chars = False
                break
        return not all_less_than_n_chars

    dataset = dataset.loc[dataset['var'].apply(var_name_not_joined_by_small_words)]

    return dataset


def remove_the_most_common_n_variable_names(dataset: pd.DataFrame, most_common_n: int = 5) -> pd.DataFrame:
    """
    Few variable names are very popular and can be shared by many types and values that
    add to the noise in the dataset

    :param dataset:
    :param most_common_n:
    :return:
    """
    var_name_frequency = dict(dataset['var'].value_counts()[:most_common_n])
    var_names_to_exclude = set(var_name_frequency.keys())

    dataset = dataset.loc[dataset['var'].apply(lambda x: x not in var_names_to_exclude)]
    return dataset


def remove_generic_names(dataset: pd.DataFrame, num_times: int = 4) -> pd.DataFrame:
    """
    Go through the dataset and find the variable names that has been more than 'num_times'
    and discard those samples from the dataset.
    :param dataset:
    :param num_times:
    :return:
    """
    variable_names_to_types = dataset[['var', 'type']].groupby(['var']).agg(set).to_dict()['type']
    variable_names_been_few_types = {var_name for var_name, types_it_has_been in
                                     variable_names_to_types.items() if len(types_it_has_been) < num_times}

    dataset = dataset[dataset['var'].apply(lambda var_name: var_name in variable_names_been_few_types)]
    return dataset


def write_types_and_frequencies(positive_example_out_file_path: str, list_of_types_in_dataset_out_file: str) -> None:
    """
    Write the name of the type and the frequency of the type in the dataset.
    :param positive_example_out_file_path:  The file that contains all assignments.
    :param list_of_types_in_dataset_out_file: The file where the output will be written
    :return:
    """
    print(f"Reading '{positive_example_out_file_path}'")
    positive_examples_dataset = pd.read_pickle(positive_example_out_file_path, compression='gzip')
    positive_examples_dataset.dropna(inplace=True)

    all_types = dict(positive_examples_dataset['type'].value_counts())
    all_types = {k: int(v) for k, v in all_types.items()}
    print(f'Writing to {list_of_types_in_dataset_out_file} the types and corresponding frequencies')
    with open(list_of_types_in_dataset_out_file, 'w') as f:
        json.dump(all_types, f)


def create_k_fold_test_datasets(positive_example_dataset, negative_example_dataset, K, pos_out_file_path,
                                neg_out_file_path, test_out_file_path):
    k_fold = KFold(n_splits=K, shuffle=True, random_state=42)

    pos_indices = positive_example_dataset.index
    neg_indices = negative_example_dataset.index

    for fold, (tr_indices, tst_indices) in enumerate(k_fold.split(pos_indices)):
        pos_examples_training = positive_example_dataset.iloc[tr_indices]
        neg_examples_training = negative_example_dataset.iloc[tr_indices]

        print(f"\nSaving the positive examples dataset as '{pos_out_file_path.replace('.pkl', f'_p{fold}.pkl')}'")
        pos_examples_training.to_pickle(pos_out_file_path.replace('.pkl', f'_p{fold}.pkl'), 'gzip')
        print(f"\nSaving the negative examples dataset as '{neg_out_file_path.replace('.pkl', f'_p{fold}.pkl')}'")
        neg_examples_training.to_pickle(neg_out_file_path.replace('.pkl', f'_p{fold}.pkl'), 'gzip')

        pos_examples_test = positive_example_dataset.iloc[tst_indices]
        neg_examples_test = negative_example_dataset.iloc[tst_indices]

        test_dataset = pd.concat((pos_examples_test, neg_examples_test), ignore_index=True)
        test_dataset.to_pickle(test_out_file_path.replace('.pkl', f'_p{fold}.pkl'), 'gzip')


def process(positive_examples_dir: str,
            positive_example_out_file_path: str,
            negative_example_out_file_path: str,
            heuristics_for_generating_negative_examples: str,
            test_example_out_file_path: str = None) -> None:
    """
    This function goes through the extracted files (.json/.pickle) created during the dynamic analysis of the
    Python files and creates a single file containing all assignments, called positive examples. Next, it uses
    the positive examples to generate negative examples.

    :param heuristics_for_generating_negative_examples: We can have two heuristics 'random' or 'weighted random'
    :param positive_examples_dir: The directory that contains all the .json/.pickle files created during
                                    dynamic analysis.
    :param positive_example_out_file_path: The output single file created by collecting all assignments present
                                    in positive_examples_dir. (If this filepath already exists, then going through
                                    the files of positive_examples_dir is skipped).
    :param negative_example_out_file_path: The file path where all negative example assignments are present. (If
                                        this file is not present then it is generated else existing one is used)
    :param test_example_out_file_path: The output file path where the test dataset gets written to.
    :return: Does not return anything rather writes everything to file.
    """

    # If the single file (positive examples) containing all assignments does not exist, create it. This step may take
    # some time to complete.
    if not Path(positive_example_out_file_path).is_file():
        print(f"Trimming 'value' field to {MAX_VALUE_SIZE} characters")
        file_extension = ('.json', '.pickle')[1]
        extracted_data_files = list(Path(positive_examples_dir).rglob(f'*{file_extension}'))

        positive_examples_dataset = read_dataset_given_files(extracted_data_files=extracted_data_files,
                                                             file_extension=file_extension)
        positive_examples_dataset.dropna(inplace=True)

        # Create the labels
        positive_examples_dataset['p_buggy'] = 0.0
        print(f"Saving dataset as '{positive_example_out_file_path}'")
        positive_examples_dataset.to_pickle(path=positive_example_out_file_path, compression='gzip')

    # If using jupyter-notebook environment, then do not pre-process and simply return
    try:
        cfg = get_ipython().config
        return
    except NameError:
        from tqdm import tqdm

    # If the user chooses 'n' here then the pre-processing of positive examples is done and then negative
    # examples are created.
    if Path(positive_example_out_file_path).is_file() and Path(negative_example_out_file_path).is_file():
        ans = input(
            "Both  'positive' and 'negative' dataset exists. \n Should the negative examples be recreated 'y'/'n' \n (The default option is to press 'n' and continue) --> ") or 'n'
        if ans != 'y':
            print('Not recreating')
            return

    # We keep a backup of the non pre-processed version of the positive examples because
    # we might want to not use some of the heuristics used during pre-processing.
    backup_dataset = 'results/backups_of_datasets/positive_examples_unprocessed.pkl'
    print(f'Copying {backup_dataset} from backup')
    shutil.copy(backup_dataset, 'results/positive_examples.pkl')
    print(f"Reading '{positive_example_out_file_path}'")
    positive_examples_dataset = pd.read_pickle(positive_example_out_file_path, compression='gzip')
    positive_examples_dataset.dropna(inplace=True)

    init_len_pos_examples = len(positive_examples_dataset)

    # ====================================== HEURISTICS BASED PRE-PROCESSING =====================================
    print(
        f"\n{'-' * 10} Pre-processing the dataset. Current size of the dataset is {init_len_pos_examples} {'-' * 10}")
    # Remove some noise created during the extraction
    print(f"{'*' * 5} Removing noise")
    positive_examples_dataset = remove_noise_from_dataset(dataset=positive_examples_dataset)
    print(f'\t New Size --> {len(positive_examples_dataset)}')
    # Remove small variable names
    print(f"{'*' * 5} Removing all variable names whose length is less than {min_var_name_len}")
    positive_examples_dataset = positive_examples_dataset[
        positive_examples_dataset['var'].apply(is_var_name_with_greater_than_len_n)]
    print(f'\t New Size --> {len(positive_examples_dataset)}')

    # select only variable names that has not been constructed only using small words eg. ab_cd
    print(f"{'*' * 5} Removing variable names based on heuristics")
    positive_examples_dataset = select_variable_names_based_on_heuristics(dataset=positive_examples_dataset)
    print(f'\t New Size --> {len(positive_examples_dataset)}')

    # merge similar types such as dict and defaultdict and select only the most frequent 'n' types
    print(f"{'*' * 5} Removing uncommon types and merging similar types like 'dict' & 'defaultdict'")
    positive_examples_dataset = discard_and_merge_types_select_most_common_n(dataset=positive_examples_dataset,
                                                                             most_common_n=10)
    print(f'\t New Size --> {len(positive_examples_dataset)}')

    after_processed_len_pos_examples = len(positive_examples_dataset)

    columns_to_keep = ['value', 'len', 'shape', 'type']

    cleaned_pos_save_path = positive_example_out_file_path.replace('.pkl', '_cleaned.pkl')
    print(f"Saving the cleaned positive examples as {cleaned_pos_save_path}")
    positive_examples_dataset.to_pickle(path=cleaned_pos_save_path,
                                        compression='gzip')

    # ====================================== NEGATIVE EXAMPLES =====================================
    print(f"{'-' * 15} Negative Examples {'-' * 15}")

    negative_examples_dataset = create_neg.create_negative_examples_from_data(
        positive_examples_dataset=positive_examples_dataset, columns_to_keep=columns_to_keep,
        way=heuristics_for_generating_negative_examples)
    # Add label
    negative_examples_dataset['p_buggy'] = 1.0

    # ====================================== TEST DATASET ===========================================
    if test_example_out_file_path and init_len_pos_examples != after_processed_len_pos_examples:
        print(f"{'-' * 15} Test Dataset {'-' * 15}")
        # Randomly shuffle both the datasets
        positive_examples_dataset = positive_examples_dataset.sample(frac=1, random_state=42)
        negative_examples_dataset = negative_examples_dataset.sample(frac=1, random_state=42)

        n = 5000  # 'n' each from positive and negative examples
        print(
            f'Randomly selecting {n * 2} examples for test dataset creation and writing to {test_example_out_file_path}')

        test_dataset_pos = positive_examples_dataset[:n]
        test_dataset_neg = negative_examples_dataset[:n]

        test_dataset = pd.concat((test_dataset_pos, test_dataset_neg), ignore_index=True)

        positive_examples_dataset = positive_examples_dataset[n:]
        negative_examples_dataset = negative_examples_dataset[n:]

        test_dataset.to_pickle(test_example_out_file_path, 'gzip')
    # ---------------------------------------------------------------------------
    print(
        f"\nSaving the positive examples dataset of size {len(positive_examples_dataset)} as '{positive_example_out_file_path}'")
    positive_examples_dataset.to_pickle(path=positive_example_out_file_path, compression='gzip')

    # Create the labels
    print(
        f"Saving the negative examples dataset of size {len(negative_examples_dataset)} as '{negative_example_out_file_path}'")
    negative_examples_dataset.to_pickle(path=negative_example_out_file_path, compression='gzip')

    # --------------- Save a sample of the dataset for inspection ----------------
    # pos_sample_fp = positive_example_out_file_path.replace('.pkl', '.csv')
    # neg_sample_fp = negative_example_out_file_path.replace('.pkl', '.csv')
    # K = 100
    # positive_examples_dataset.sample(n=K).reset_index(drop=True, inplace=True)
    # positive_examples_dataset.to_csv(pos_sample_fp)
    # negative_examples_dataset.sample(n=K).reset_index(drop=True, inplace=True)
    # negative_examples_dataset.to_csv(neg_sample_fp)
    # print(f"Saved a sample of {K} examples as '{pos_sample_fp}' & '{neg_sample_fp}' respectively for inspection")