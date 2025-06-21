import argparse

from script.test.tester import Tester
from script.config.config import args_to_dict, build_args_settings

def parse_args():
    parser = argparse.ArgumentParser()

    # Parameters that do not affect train/test dataset.
    general = parser.add_argument_group("General Parameters")

    # Parameters that affect both the train/test dataset.
    pre_split = parser.add_argument_group("Pre-Split Parameters")
    split = parser.add_argument_group("Split Parameters")

    # Parameters that affect only the train dataset.
    post_split = parser.add_argument_group("Post-Split Parameters")

    # Parameters that affect only the test dataset.
    test = parser.add_argument_group("Test Parameters")

    # General
    general.add_argument('--models', nargs='+', type=str, help='Model(s) to run. Must match keys in model_settings.yaml.')
    general.add_argument('--display_logs', type=int, choices=[0, 1], help='0 = suppress stderr to file, 1 = show in console.')
    general.add_argument('--autoload', type=int, choices=[0, 1],help='Whether to automatically load a model checkpoint.')
    general.add_argument('--overwrite', type=int, choices=[0, 1], help='0 = skip test if already done, 1 = overwrite previous results')
    general.add_argument('--path_to_checkpoint', type=str, help='Path to model checkpoint file.')
    general.add_argument('--n_samples', nargs='+', type=int, help='Number(s) of passwords to generate.')
    general.add_argument('--test_config', type=str, help='Path to YAML scenario configuration file.')
    general.add_argument('--guesses_file', nargs='+', type=str, help='Use existing password guesses from file(s).')
    general.add_argument('--sub_samples_from_file', nargs='+', type=str, help='Subsample passwords from given file(s).')
    general.add_argument('--data_to_embed', nargs='+', type=str, help="Path(s) to data for PassFlow encoder embedding.")
    general.add_argument('--save_guesses', type=int, choices=[0, 1], default=1, help="1 = save guesses, 0 = don't save guesses")
    general.add_argument('--save_matches', type=int, choices=[0, 1], default=1, help="1 = save matched passwords, 0 = don't save matched passwords"
    )

    # Pre-split
    pre_split.add_argument('--max_length', nargs='+', type=int, help='Maximum password length(s).')
    pre_split.add_argument('--char_bag', nargs='+', type=str, help='Character set(s) to consider.')
    pre_split.add_argument('--train_datasets', nargs='+', type=str, help='Name(s) of training dataset(s).')

    # Split
    split.add_argument('--train_split_percentage', nargs='+', type=int, help='Percentage split for training data.')

    # Post-split
    post_split.add_argument('--train_chunk_percentage', nargs='+', type=int, help='Subset of training data to use.')

    # Test
    test.add_argument('--test_datasets', nargs='+', type=str, help='Name(s) of testing dataset(s).')
    test.add_argument('--test_frequency', nargs='+', type=int, help='Top/bottom N% of test set by frequency.')

    return parser.parse_args()


def main():
    print("Starting...")
    args = parse_args()
    args = args_to_dict(args)

    args_settings = build_args_settings(args)

    tester = Tester(args_settings)

    if not tester.evaluation_script:
        tester.prepare_environment()
        tester.run_test()
    else:
        tester.prepare_script_settings()
        if tester.evaluation_script != '__null__':
            tester.execute_eval_script()

    csv_rows = tester.written_rows

    if tester.figure_script:
        tester.execute_figure_script(csv_rows=csv_rows)


if __name__ == '__main__':
    main()