import argparse
from collections import defaultdict
import re
import sys
import os

sys.path.append(os.getcwd())

from script.utils.file_operations import write_to_csv, read_files

regex = {
    'r1': r'^[A-Za-z]+$',
    'r2': r'^[a-z]+$',
    'r3': r'^[A-Z]+$',
    'r4': r'^[0-9]+$',
    'r5': r'^[\W_]+$',
    'r6': r'^(?=.*[A-Za-z])(?=.*\d)[A-Za-z\d]+$',
    'r7': r'^(?=.*[A-Za-z])(?=.*[\W_])[A-Za-z\W_]+$',
    'r8': r'^(?=.*\d)(?=.*[\W_])[\d\W_]+$',
    'r9': r'^(?=.*\d)(?=.*[\W_])(?=.*[A-Za-z])[A-Za-z\d\W_]+$',
    'r10': r'^[a-zA-Z][a-zA-Z0-9\W_]+[0-9]$',
    'r11': r'^[A-Za-z][A-Za-z0-9\W_]+[\W_]$',
    'r12': r'^[0-9][A-Za-z]+$',
    'r13': r'^[0-9][A-Za-z0-9\W_]+[\W_]$',
    'r14': r'^[0-9][A-Za-z0-9\W_]+[0-9]$',
    'r15': r'^[\W_][A-Za-z]+$',
    'r16': r'^[\W_][A-Za-z0-9\W_]+[\W_]$',
    'r17': r'^[\W_][A-Za-z0-9\W_]+[0-9]$',
    'r18': r'^[a-zA-Z0-9\W_]+[!]$',
    'r19': r'^[a-zA-Z0-9\W_]+[1]$',
}

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--matches',
                        type=str,
                        required=True,
                        help='Paths to matches file.')

    parser.add_argument('--test_dataset',
                        type=str,
                        required=True,
                        help='Paths to test datasets.')

    parser.add_argument('--mode',
                        type=str,
                        choices=['length', 'pattern', 'both'],
                        default='both',
                        help='Choose mode: length, pattern, or both.')

    parser.add_argument('--info',
                        type=str,
                        required=True,
                        help='Infos structured as <model_name>-<dataset_name>')

    return parser.parse_args()

def compute_match_per_length(test_passwords, guessed_set):
    diz_matches_total = defaultdict(lambda: [0, 0])

    for password in test_passwords:
        if not password:
            continue
        length = str(len(password))
        diz_matches_total[length][1] += 1
        if password in guessed_set:
            diz_matches_total[length][0] += 1

    matches_per_length = {}
    weights = {}

    for length in diz_matches_total:
        matches_per_length[length] = round((diz_matches_total[length][0] / diz_matches_total[length][1]) * 100, 2)
        weights[length] = diz_matches_total[length][1]

    return matches_per_length, weights

def compute_match_per_pattern(test_passwords, guessed_set):
    diz_matches_total = defaultdict(lambda: [0, 0])

    for password in test_passwords:
        if not password:
            continue

        is_match = password in guessed_set

        for id, pattern in regex.items():
            if re.fullmatch(pattern, password):
                diz_matches_total[id][1] += 1
                if is_match:
                    diz_matches_total[id][0] += 1

    matches_per_pattern = {}
    weights = {}

    for pattern in diz_matches_total:
        matches_per_pattern[pattern] = round((diz_matches_total[pattern][0] / diz_matches_total[pattern][1]) * 100, 2)
        weights[pattern] = diz_matches_total[pattern][1]

    return matches_per_pattern, weights

def compute_both(test_passwords, guessed_set):
    diz_matches_total_pattern = defaultdict(lambda: [0, 0])
    weights_pattern = {}

    diz_matches_total_length = defaultdict(lambda: [0, 0])
    weights_length = {}

    for password in test_passwords:
        if not password:
            continue

        is_match = password in guessed_set

        length = str(len(password))
        diz_matches_total_length[length][1] += 1
        if is_match:
            diz_matches_total_length[length][0] += 1

        for id, pattern in regex.items():
            if re.fullmatch(pattern, password):
                diz_matches_total_pattern[id][1] += 1
                if is_match:
                    diz_matches_total_pattern[id][0] += 1

    matches_per_length = {}
    matches_per_pattern = {}

    for length in diz_matches_total_length:
        matches_per_length[length] = round((diz_matches_total_length[length][0] / diz_matches_total_length[length][1]) * 100, 2)
        weights_length[length] = diz_matches_total_length[length][1]

    for pattern in diz_matches_total_pattern:
        matches_per_pattern[pattern] = round((diz_matches_total_pattern[pattern][0] / diz_matches_total_pattern[pattern][1]) * 100, 2)
        weights_pattern[pattern] = diz_matches_total_pattern[pattern][1]

    return matches_per_length, weights_length, matches_per_pattern, weights_pattern


def prepare_to_csv(stats, info, weights, mode):
    fieldnames = ["model", "dataset", "test-size", "key", "percentage"]

    model, dataset = info.split("-")
    fixed_data = [model, dataset]
    variable_data = []

    for key in stats:
        variable_data.append([weights[key], key, stats[key]])

    write_to_csv(path=f"results/match_per_{mode}.csv",
                 fieldnames=fieldnames,
                 fixed_data=fixed_data,
                 variable_data=variable_data)


def main():
    args = parse_args()

    matches = args.matches
    test_dataset = args.test_dataset
    mode = args.mode
    info = args.info

    test_passwords = set(read_files(test_dataset))
    guessed_set = set(read_files(matches))

    if mode == 'length':
        matches_per_length, weights = compute_match_per_length(test_passwords, guessed_set)
        prepare_to_csv(matches_per_length, info, weights, "length")
    elif mode == 'pattern':
        matches_per_pattern, weights = compute_match_per_pattern(test_passwords, guessed_set)
        prepare_to_csv(matches_per_pattern, info, weights, "pattern")
    else:
        matches_per_length, weights_length, matches_per_pattern, weights_pattern = compute_both(test_passwords, guessed_set)
        prepare_to_csv(matches_per_length, info, weights_length, "length")
        prepare_to_csv(matches_per_pattern, info, matches_per_pattern, "pattern")

if __name__ == '__main__':
    main()