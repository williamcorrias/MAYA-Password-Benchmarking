import os
import sys
import argparse

sys.path.append(os.getcwd())

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',
                        type=str,
                        required=True,
                        help='Specify between guesses or matches')

    return parser.parse_args()

def rename_files(base_dir, mode):
    for root, dirs, files in os.walk(base_dir):
        if mode in root:
            for file in files:
                if file.endswith('.gz'):
                    old_file_path = os.path.join(root, file)
                    new_file_path = os.path.join(root, f'{mode}.gz')

                    if not os.path.exists(new_file_path):
                        print(f"{old_file_path} renamed in {new_file_path}")
                        os.rename(old_file_path, new_file_path)


base_directory = 'results'
args = parse_arguments()
mode = args.mode
rename_files(base_directory, mode)