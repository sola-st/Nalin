"""

Created on 21-April-2020
@author Jibesh Patra

"""
import argparse
import os
import sys
from multiprocessing import cpu_count
from pathlib import Path
import torch
from torch.utils.data import DataLoader


from dataset_utils.data_transformers.AblationTransformer import AblationTransformer

from dataset_utils.data_transformers.fastTextEmbeddingOfVarName import fastTextEmbeddingOfVarName
from dataset_utils.data_transformers.OneHotEncodingOfTypes import OneHotEncodingOfType
from dataset_utils.data_transformers.RepresentLen import RepresentLen
from dataset_utils.data_transformers.RepresentShape import RepresentShape
from dataset_utils.data_transformers.ResizeData import ResizeData
from dataset_utils.data_transformers.ValueToCharSequence import ValueToCharSequence

from dataset_utils.pre_process_dataset import process, write_types_and_frequencies
from read_dataset import get_training_val_dataset, get_test_dataset

from models.VarValueClassifierRNN import VarValueClassifierRNN

from command_line_args import get_parsed_args

if __name__ == '__main__':
    args = get_parsed_args(argparse=argparse)
    positive_examples_dir = 'results/dynamic_analysis_outputs'
    list_of_types_in_dataset_out_file = 'results/list_of_types_in_dataset.json'
    token_embedding_path = 'benchmark/python_embeddings.bin'
    results_dir = 'results'

    train, test = args.train, args.test

    if not train and not test:
        print('Either "training" or "testing" is required')
        sys.exit(1)

    batch_size = args.batch_size
    num_epochs = args.num_epochs
    max_num_of_chars_in_value = 100  # Number of characters in the value part of the assignment

    # You may specify your name
    if args.name:
        model_name_suffix = args.name
    else:
        model_name_suffix = 'nalin'

    model_name = f'RNNClassifier_{model_name_suffix}'


    pos_dataset_file_path = args.pos_dataset
    neg_dataset_file_path = args.neg_dataset
    test_dataset_file_path = args.test_dataset

    """
    There are three heuristics for generating negative examples:
        1. use_dimension: refers to computing various properties on the positive examples and then using them to
        generate the negative examples. (Code adapted from the initial code by MP)
        2. random: only useful for cases when the data contains single type (eg.string). The approach is simply randomizes the
        values. The idea is to check if certain idenfiers such as URL are only assigned values having certain properties
        3. weighted_random: This is the default strategy. Refer to the code where it is implemented for further details.
    """
    heuristics_for_generating_negative_examples = ['random','weighted_random'][1]

    # Types and the corresponding frequency in the dataset
    """
    Pre-process dataset. This is an one time task ==>
        - Remove empty/malformed extracted data
        - Create negative examples
        - Create labels for the extracted data (label -> probability of buggy)
    """
    if not test:
        process(positive_examples_dir=positive_examples_dir,
                positive_example_out_file_path=pos_dataset_file_path,
                negative_example_out_file_path=neg_dataset_file_path,
                test_example_out_file_path=test_dataset_file_path,
                heuristics_for_generating_negative_examples=heuristics_for_generating_negative_examples)
        write_types_and_frequencies(positive_example_out_file_path=pos_dataset_file_path,
                                    list_of_types_in_dataset_out_file=list_of_types_in_dataset_out_file)
    # Embeddings have been learned from ALL python files in the benchmark (~1M files). We could
    # successfully extract assignments from some of these python files.
    if not os.path.exists(token_embedding_path):
        print(f'Could not read from {token_embedding_path}. \nNeed an embedding path to continue')
        sys.exit(1)
    test_examples_dir = 'results/test_examples'
    saved_model_path = None
    if args.test and args.saved_model:
        saved_model_path = args.saved_model
    elif args.test and not args.saved_model:
        print("A saved model path is needed")
        sys.exit(1)
    embedding_dim = 0
    features_to_ablate = args.ablation


    # Workaround for debugging on a laptop. Change with the cpu_count of your machine if required for debugging data loading
    # else leave it alone
    if cpu_count() > 20:
        num_workers_for_data_loading = cpu_count()
    else:
        num_workers_for_data_loading = 0
        batch_size = 25
    config = {"num_workers": num_workers_for_data_loading, "pin_memory": True}

    device = torch.device(
        'cuda:0' if torch.cuda.is_available() else 'cpu')

    # Initialize model and model specific dataset data_transformers
    print(f"\n{'-' * 20} Using model '{model_name}' {'-' * 20}")


    resize_data = ResizeData(len_of_value=max_num_of_chars_in_value)
    value_to_one_hot = ValueToCharSequence(
        len_of_value=max_num_of_chars_in_value)

    one_hot_encoding_of_type = OneHotEncodingOfType(max_types_to_select=10,
                                                    types_in_dataset_file_path=list_of_types_in_dataset_out_file)  # We select only top 10 types
    size_of_type_encoding = len(one_hot_encoding_of_type.one_hot_init)

    var_name_fastText_embd = fastTextEmbeddingOfVarName(embedding_path=token_embedding_path)
    embedding_dim = var_name_fastText_embd.embedding_dim

    len_repr = RepresentLen()
    shape_repr = RepresentShape()

    data_transformations = [resize_data,  # must be always the first transformation
                            var_name_fastText_embd,
                            value_to_one_hot,
                            one_hot_encoding_of_type,
                            len_repr,
                            shape_repr]

    model = VarValueClassifierRNN(embedding_dim=embedding_dim,
                                  num_of_characters_in_alphabet=value_to_one_hot.nbs_chars,
                                  model_name=model_name,
                                  size_of_value=resize_data.len_of_value)

    assert model is not None, "Initialize a model to run training/testing"
    model.to(device)

    if len(features_to_ablate):
        ablation_transformer = AblationTransformer(features_to_ablate=features_to_ablate)
        print(f"## Not using features --> {features_to_ablate} ##")
        data_transformations.append(ablation_transformer)

    if train:
        print(f"{'-' * 15} Reading dataset for training {'-' * 15}")
        print(f"-- Resizing the values to {max_num_of_chars_in_value} characters during training")
        # Read the dataset
        training_dataset, validation_dataset = get_training_val_dataset(
            positive_examples_dataset_file_path=pos_dataset_file_path,
            negative_examples_dataset_file_path=neg_dataset_file_path,
            all_transformations=data_transformations,
            nb_examples=-1)
        train_data = DataLoader(
            dataset=training_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **config)
        validation_data = DataLoader(
            dataset=validation_dataset, batch_size=batch_size, shuffle=True, drop_last=True, **config)

        model.run_epochs(training_data=train_data,
                         validation_data=validation_data, num_epochs=num_epochs, results_dir=results_dir)

    if test:
        print(f"{'-' * 15} Reading dataset for testing {'-' * 15}")
        test_dataset = get_test_dataset(
            test_examples_dir=test_examples_dir,
            results_dir=results_dir,
            all_transformations=data_transformations,
            dataset_out_file=test_dataset_file_path)
        batched_test_dataset = DataLoader(
            dataset=test_dataset, batch_size=batch_size, shuffle=False, drop_last=False, **config)
        model.load_model(path_to_saved_model=saved_model_path)
        predictions = model.run_testing(data=batched_test_dataset)

        test_data_with_predictions = test_dataset.data
        test_data_with_predictions['predicted_p_buggy'] = predictions

        predicted_outfile_path = os.path.join(results_dir,
                                              f'prediction_results/{Path(test_dataset_file_path).stem}_predictions.pkl')
        print(f"Writing to '{predicted_outfile_path}'")
        test_data_with_predictions.sort_values('predicted_p_buggy', ascending=False, inplace=True)
        test_data_with_predictions.reset_index(drop=True, inplace=True)
        # test_data_with_predictions.to_csv(predicted_outfile_path)
        test_data_with_predictions.to_pickle(path=predicted_outfile_path, compression='gzip')

