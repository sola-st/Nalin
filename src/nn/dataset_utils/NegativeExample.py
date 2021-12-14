"""

Created on 16-December-2020
@author Jibesh Patra

Given an index assignment, create a negative example for provided dimensions.

"""
from multiprocessing import cpu_count, Pool
from typing import List, Tuple

import numpy as np
from tqdm import tqdm
import pandas as pd
import random


class NegativeExample:
    def __init__(self, dataset: pd.DataFrame, dimensions: List) -> None:
        self.dataset = dataset
        self.dimensions = dimensions

        self.name_val_to_probability = {}  # optimization?
        self.total = len(dataset)  # optimization?
        self.all_idx_labels = list(dataset.index)  # optimization?

    def get_prob(self, indices: Tuple) -> List:
        idx_label, assignment_idx_label = indices
        per_classification_probabilities = []
        for dimension in self.dimensions:
            probability = dimension(assignment_idx_label, idx_label)
            per_classification_probabilities.append(probability)
        return per_classification_probabilities

    @staticmethod
    def scale_to_distribution(distribution):
        # WARNING: Sum can be rounded to 0 if it is very small or very large
        # This will throw error for random_index selection
        # https://stackoverflow.com/questions/27784528/numpy-division-with-runtimewarning-invalid-value-encountered-in-double-scalars
        sm = sum(distribution)
        return [p / sm for p in distribution]

    def __call__(self, assignment_idx_label: int) -> Tuple[int, int]:
        """

        Given an index label in the dataset, return a index label of the negative example
        using the dimensions.

        :param assignment_idx_label: The index label for which a negative example needs to
                                     be created.
        :return: The original index label and a random index label in the dataset
                 whose 'value' is used as the new negative example
        """

        all_values_probabilities = []
        val_to_avoid = self.dataset.loc[assignment_idx_label]['value']

        # var_name = self.dataset.loc[assignment_idx_label]['var']
        # comb_name_val = (var_name, val_to_avoid)

        # if comb_name_val in self.name_val_to_probability:
        #     all_values_probabilities = self.name_val_to_probability[comb_name_val]
        # else:
        for idx_label in self.all_idx_labels:
            # row = self.dataset.loc[idx_label]
            # if val_to_avoid == row['value']:
            #     all_values_probabilities.append(.0)
            #     continue
            per_classification_probabilities = self.get_prob(indices=(idx_label, assignment_idx_label))
            all_values_probabilities.append(np.prod(per_classification_probabilities))
        all_values_probabilities_scaled = NegativeExample.scale_to_distribution(all_values_probabilities)
        # self.name_val_to_probability[comb_name_val] = all_values_probabilities

        sampled_value, random_index, random_index_label = None, None, None

        while sampled_value is None or sampled_value == val_to_avoid:
            try:
                random_index = np.random.choice(np.arange(0, len(all_values_probabilities_scaled)),
                                                p=all_values_probabilities_scaled)
                randomly_selected_row = self.dataset.iloc[random_index]
                sampled_value = randomly_selected_row['value']
                random_index_label = randomly_selected_row.name
            except Exception as e:
                return assignment_idx_label, random.choice(self.all_idx_labels)
        return assignment_idx_label, random_index_label


if __name__ == '__main__':
    import sys

    sys.path.extend(['src/nn'])
    from dataset_utils.dimensions_over_data.EvenOdd import EvenOddDim
    from dataset_utils.dimensions_over_data.PosNegZeroOther import PosNegZeroOtherDim
    from dataset_utils.dimensions_over_data.Length import Length
    from dataset_utils.dimensions_over_data.Type import ValTypeDim
    from dataset_utils.dimensions_over_data.SpecialChar import SpecialChar
    import time

    # positive_examples_dataset_path = 'results/backups_of_datasets/positive_examples_with_static_analysis_small.pkl'  # A sample of 10,000
    positive_examples_dataset_path = 'results/positive_examples_cleaned.pkl'  # The complete dataset

    # --------------------------- MP Dataset ----------------------------------------
    mp_dataset = {
        'var': ['a', 'a', 'a', 'a', 'a', 'a', 'a', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'b', 'c', 'c', 'c', 'c',
                'c', 'c', 'c', 'c', 'c', 'c', 'c', 'c', 'd'],
        'value': [-1, -4, 0, -1, 5, 3, 1, 0, 1, 0, 1, 1, 1, 1, 0, 1, 2, 1, 4, 5, 2, 1, 3, 0, 1, 3, 4, 2, 1, 0]
    }
    np.random.seed(1)
    max_num_of_chars_in_val = 100

    positive_examples_dataset = pd.read_pickle(positive_examples_dataset_path, 'gzip')

    # positive_examples_dataset = pd.read_pickle(positive_examples_dataset_path, 'gzip').sample(
    #     10000)  # Random sample of 100
    # positive_examples_dataset = pd.DataFrame(mp_dataset)

    # def trim_vals(val):
    #     if len(val) >= max_num_of_chars_in_val:
    #         return val[:max_num_of_chars_in_val // 2] + val[-max_num_of_chars_in_val // 2:]
    #     else:
    #         return val
    #
    #
    # print(f" Resizing the value field and keeping at most {max_num_of_chars_in_val} characters in  value")
    # positive_examples_dataset['value'] = positive_examples_dataset['value'].apply(trim_vals)

    print(f'{time.asctime()}: The "positive_examples_dataset" contains {len(positive_examples_dataset)} items')
    print(positive_examples_dataset)

    len_dimension = Length(positive_examples_dataset)
    value_type_dimension = ValTypeDim(positive_examples_dataset)
    special_char_dimension = SpecialChar(positive_examples_dataset)
    pos_neg_dimension = PosNegZeroOtherDim(positive_examples_dataset)

    dims = [len_dimension, value_type_dimension, special_char_dimension, pos_neg_dimension]

    combined_dims_name = '&'.join([dm.dim_name for dm in dims])
    out_json_file_path = f'results/negative_labels_{len(positive_examples_dataset)}_{combined_dims_name}.json'

    create_negative_example = NegativeExample(dataset=positive_examples_dataset,
                                              dimensions=dims)

    neg_example_labels = []
    positive_dataset_index_labels = list(positive_examples_dataset.index)
    pos_idx_to_neg_idx = {}
    # ----- Non multi-processing version for debugging -----------------------------
    # for ix, pos_idx_label in enumerate(
    #         tqdm(positive_dataset_index_labels, total=len(positive_examples_dataset), desc='SEQUENTIALLY Creating '
    #                                                                                        'negative examples ')):
    #     source_idx_label, selected_idx_label = create_negative_example(pos_idx_label)
    #     neg_example_labels.append((source_idx_label, selected_idx_label))

    # ----- Multi-processing version -----------------------------------------------
    with Pool(cpu_count()) as p:
        with tqdm(total=len(positive_examples_dataset)) as pbar:
            pbar.set_description_str(desc='Creating negative examples ', refresh=False)
            for ix, ret_labels in enumerate(
                    p.imap_unordered(create_negative_example, positive_dataset_index_labels, chunksize=250)):
                source_idx_label, selected_idx_label = ret_labels
                pos_idx_to_neg_idx[str(source_idx_label)] = str(selected_idx_label)
                neg_example_labels.append((source_idx_label, selected_idx_label))
                pbar.update()
            p.close()
            p.join()
    import json

    with open(out_json_file_path, 'w') as out:
        print(f'Writing to {out_json_file_path}')
        json.dump(pos_idx_to_neg_idx, out)
    # -------------------------------------------------------------------------
    # print('\nThe selected negative examples are: ')
    # selected_values = []
    # for source_idx_label, selected_idx_label in neg_example_labels:
    #     source_row = positive_examples_dataset.loc[source_idx_label]
    #     selected_row = positive_examples_dataset.loc[selected_idx_label]
    #     selected_values.append(selected_row['value'])
    #     # print(f"Source: {source_row['var']},{source_row['value']} --- Selected: {selected_row['var']},
    #     # {selected_row['value']} ")
    #     # print('='*80)
    # # print(selected_values)
    #
    # print(f'{time.asctime()}:, {len(selected_values)} selected values')
    # -------------------------------------------------------------------------
