"""

Created on 22-January-2021
@author Jibesh Patra


"""


def get_parsed_args(argparse):
    parser = argparse.ArgumentParser(
        prog='python src/nn/run_classification.py',
        description="training and testing for model",
        epilog="You must provide at least one of the train, test flags")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        metavar="N",
        help="input batch size for training and testing (default: 128)")
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=15,
        metavar="N",
        help="number of epochs to train (default: 10)")
    parser.add_argument(
        '--train',
        action='store_true',
        help='Provide this flag if you want to Train the model')
    parser.add_argument(
        '--pos-dataset',
        type=str,
        default='results/positive_examples.pkl',
        help='Provide the path for positive dataset')
    parser.add_argument(
        '--neg-dataset',
        type=str,
        default='results/negative_examples.pkl',
        help='Provide the path for negative dataset')
    parser.add_argument(
        '--test',
        action='store_true',
        help='Provide this flag if you want to Test the model')
    parser.add_argument(
        '--test-dataset',
        type=str,
        default='results/test_examples.pkl',
        help='Provide the path to the test dataset')
    parser.add_argument(
        '--saved-model',
        type=str,
        help='Provide this flag if you want to Test the model')
    parser.add_argument(
        '--name',
        type=str,
        help='This is added to the saved model name for future reference')
    parser.add_argument(
        '--ablation',
        nargs='*',
        default=[],
        help='Provide the features to ablate in as space separated values eg. --ablation var type \n # Possible values -->'
             "'value_as_one_hot', 'var', 'type', 'len', 'shape'"
    )
    args = parser.parse_args()
    return args
