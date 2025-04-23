import os
os.makedirs("logs", exist_ok=True)
import argparse
from script.test.tester import Tester


def parse_args():
    parser = argparse.ArgumentParser()

    general_group = parser.add_argument_group("General Parameters")  #parameters that do not affect train/test dataset.
    pre_split_group = parser.add_argument_group("Pre-Split Parameters")  #parameters that affect both the train/test dataset.
    split_group = parser.add_argument_group("Split Parameters") #parameters that affect both the train/test dataset.
    post_split_group = parser.add_argument_group("Post-Split Parameters") #parameters that affect both the train/test dataset.
    test_group = parser.add_argument_group("Test Parameters") #parameters that affect only the test dataset.

    general_group.add_argument('--models',
                               type=str,
                               required=False,
                               nargs="+",
                               help='Name(s) of the model(s) to run.')

    general_group.add_argument('--display_logs',
                               type=str,
                               required=False,
                               help="Whether to log in console the stderr, if false the stderr (progress bar) "
                                    "will be redirected to a file inside the logs dir.")

    general_group.add_argument('--autoload',
                               type=str,
                               required=False,
                               help='Whether to automatically load a model checkpoint')

    general_group.add_argument('--path_to_checkpoint',
                               type=str,
                               required=False,
                               nargs="+",
                               help='Specifies the exact path to a checkpoint file to load')

    general_group.add_argument('--n_samples',
                               type=str,
                               required=False,
                               nargs="+",
                               help='Number of passwords to generate.')

    general_group.add_argument('--test_config',
                               type=str,
                               required=False,
                               nargs="+",
                               help='PATH to a scenario config file(s). Note that the config file must match a specific format')

    general_group.add_argument('--test_reference',
                               type=str,
                               required=False,
                               nargs="+",
                               help='Instead of generating new passwords, the model will use the passwords located at the specified PATH. '
                                    'Option 1 (faster method): If a path to a scenario config file (.yaml) is provided, '
                                    'the model will automatically derive the passwords from the respective file, '
                                    'based on the settings in the config. Any missing settings will be taken from the '
                                    'command-line arguments. e.g.: using `--test_reference config/test/x.yaml` and '
                                    '`--test_config config/test/y.yaml`, the program will evaluate the model on test X '
                                    'using passwords generated during test Y. '
                                    'Option 2 (more flexible): If a path to a directory containing guesses is provided,'
                                    ' the model will use the passwords from the first file found in that directory.'
                                    )

    general_group.add_argument('--use_existing_samples',
                               type=str,
                               required=False,
                               nargs="+",
                               help='Subsamples `--n_samples` passwords from the --use_existing_samples previously generated ones. '
                                    'NOTE: All settings (except for `n_samples`) must exactly match those used to generate the original samples.')

    general_group.add_argument('--data_to_embed',
                               type=str,
                               required=False,
                               nargs="+",
                               help="Path to the data to be embedded. The embedding process will be handled solely by PassFlow's encoder.")

    pre_split_group.add_argument('--max_length',
                                 type=str,
                                 required=False,
                                 nargs="+",
                                 help='Maximum length of passwords. ')

    pre_split_group.add_argument('--char_bag',
                                 type=str,
                                 required=False,
                                 nargs="+",
                                 help='Specifies the set of characters to consider.')

    split_group.add_argument('--train_split_percentage',
                             type=int,
                             required=False,
                             nargs="+",
                             help='Percentage of data to be used for training. For example, if set to 80, '
                                  'the dataset will be split into 80% training and 20% testing.')

    split_group.add_argument('--train_datasets',
                             type=str,
                             required=False,
                             nargs="+",
                             help='Name(s) of the training dataset(s). ')

    split_group.add_argument('--test_datasets',
                             type=str,
                             required=False,
                             nargs="+",
                             help='Name of the testing dataset. Specify only if you want a'
                                  'cross-dataset scenario.')

    post_split_group.add_argument('--train_chunk_percentage',
                                  type=int,
                                  required=False,
                                  nargs="+",
                                  help='After splitting the dataset into test and train, this parameter allows '
                                       'selecting a subset of the training data. '
                                       'If a value between 1 and 100 is provided, it will be treated as a percentage '
                                       '(e.g., 75% means 75% of the original training dataset). '
                                       'If a larger number, such as 1000000, is specified, it will select exactly '
                                       'that many passwords from the training dataset to be used for training.')

    test_group.add_argument('--test_frequency',
                            type=str,
                            required=False,
                            nargs="+",
                            help='Selects the top or bottom percentage of passwords from the testing dataset, based on '
                                 'their frequency. A positive value selects the most frequent passwords. e.g.: --test_frequency 10. '
                                 'While a negative value selects the least frequent ones. e.g.: --test_frequency -10')
    return parser.parse_args()


def main():
    print("Starting...")

    args = parse_args()

    general_params = {
        "autoload": args.autoload,
        "display_logs": args.display_logs,
        "models": args.models,
        "n_samples": args.n_samples,
        "test_config": args.test_config,
        "test_reference": args.test_reference,
        "use_existing_samples": args.use_existing_samples,
        "path_to_checkpoint": args.path_to_checkpoint,
        "data_to_embed": args.data_to_embed,
    }

    pre_split_params = {
        "train_datasets": args.train_datasets,
        "max_length": args.max_length,
        "char_bag": args.char_bag,
    }

    split_params = {
        "train_split_percentage": args.train_split_percentage,
    }

    post_split_params = {
        "train_chunk_percentage": args.train_chunk_percentage,
    }

    test_params = {
        "test_frequency": args.test_frequency,
        "test_datasets": args.test_datasets,
    }

    args = {
        "general_params": general_params,
        "pre_split_params": pre_split_params,
        "split_params": split_params,
        "post_split_params": post_split_params,
        "test_params": test_params,
    }

    tester = Tester(args)
    tester.run_test()


if __name__ == '__main__':
    main()