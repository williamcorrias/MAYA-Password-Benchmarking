import os
import sys
import gzip
import argparse
from pathlib import Path
from tqdm import tqdm
from itertools import combinations

sys.path.append(os.getcwd())

from script.utils.file_operations import write_to_csv, read_files


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate and calculate metrics on guesses.")

    parser.add_argument('--guesses', type=str, nargs='+', help='Paths to guesses.')
    parser.add_argument('--test_dataset', type=str, nargs='+', help='Paths to test datasets.')
    parser.add_argument('--generate_combination', type=int, help='Size of combinations to generate for guesses.')
    parser.add_argument('--mode', type=str, choices=['evaluate', 'jaccard_index', 'mergeability_index'],
                        default='evaluate',
                        help='Choose mode: evaluate, jaccard_index, or mergeability_index.')
    parser.add_argument('--info', type=str, help='Optional info to display during evaluation.')

    return parser.parse_args()


def read_chunk(file, chunk_size=2048):
    def read_lines(file_obj):
        chunk = []
        for line in file_obj:
            chunk.append(line.strip())
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    if file.endswith('.gz'):
        with gzip.open(file, 'rt') as f:
            yield from read_lines(f)
    elif file.endswith('.txt'):
        with open(file, 'r') as f:
            yield from read_lines(f)
    else:
        raise ValueError(f"Unsupported file format: {file}")


def get_info(path):
    parts = Path(path).parts
    if "results" in parts:
        idx = parts.index("results")
        if idx + 7 < len(parts):
            return parts[idx + 2], parts[idx + 3], parts[idx + 4], parts[idx + 7]
    return None


def evaluate(test_dataset, guesses):
    test_passwords = set()
    for file in test_dataset:
        test_passwords.update(read_files(file))

    matches = set()
    uniques = set()

    progress_bar = tqdm(guesses)

    for file in guesses:
        for chunk in read_chunk(file):
            guesses_set = set(chunk)
            matches.update(guesses_set & test_passwords)
            uniques.update(guesses_set)

        progress_bar.update(1)

    total_matches = len(matches)
    total_uniques = len(uniques)

    print(f'{total_matches} matches found ({total_matches / len(test_passwords) * 100.0:.4f} of test set).'
          f' out of {total_uniques} unique samples.')


def jaccard_index(guesses, combo):
    idxs = list(range(len(guesses)))
    combos = list(combinations(idxs, combo)) if combo else [tuple(idxs)]

    for combo in combos:
        model, dataset, settings, mode = get_info(guesses[combo[0]])
        intersection = set(read_files(guesses[combo[0]]))
        union = intersection.copy()
        models = [model]

        for dataset_idx in combo[1:]:
            guess_file = guesses[dataset_idx]
            models.append(get_info(guess_file)[0])
            current_intersection = set()

            for chunk in read_chunk(guess_file):
                chunk_set = set(chunk)
                current_intersection.update(chunk_set.intersection(intersection))
                union.update(chunk_set)

            intersection = current_intersection

        if len(union) > 0:
            combo_name = "_".join(sorted(models))
            jaccard_index = len(intersection) / len(union)
            print(f'{combo_name} - Jaccard Index: {jaccard_index}, number of passwords in common: {len(intersection)}, '
                  f'number of unique passwords: {len(union)}')

            fieldnames = ["Dataset", "Test", "Combo", "Settings", "Jaccard", "Intersect", "Union"]
            fixed_data = [dataset, mode, combo_name, settings]
            variable_data = [[jaccard_index, len(intersection), len(union)]]
            write_to_csv(path=f"results/jaccard-{mode}.csv", fieldnames=fieldnames, fixed_data=fixed_data,
                         variable_data=variable_data)
        else:
            print("Empty union: Jaccard Index cannot be computed.")

def compute_mergeability_index(union, matches):
    return (union - max(matches)) / max(matches)


def mergeability_index(guesses, combo):
    idxs = list(range(len(guesses)))
    combos = list(combinations(idxs, combo)) if combo else [tuple(idxs)]

    for combo in combos:
        model, dataset, settings, mode = get_info(guesses[combo[0]])

        union = set(read_files(guesses[combo[0]]))
        models = [model]
        matches = [len(union)]

        for dataset_idx in combo[1:]:
            guess_file = guesses[dataset_idx]
            models.append(get_info(guess_file)[0])

            current_matches = set()

            for chunk in read_chunk(guess_file):
                chunk_set = set(chunk)
                current_matches.update(chunk_set)
                union.update(chunk_set)

            matches.append(len(current_matches))

        if len(union) > 0:
            combo_name = "_".join(sorted(models))
            mergeability_index_val = compute_mergeability_index(len(union), matches)
            print(f'{combo_name} - Mergeability Index: {mergeability_index_val}')

            fieldnames = ["Dataset", "Test", "Combo", "Settings", "Mergeability"]
            fixed_data = [dataset, mode, combo_name, settings]
            variable_data = [[mergeability_index_val]]
            write_to_csv(path=f"results/mergeability-{mode}.csv",
                         fieldnames=fieldnames,
                         fixed_data=fixed_data,
                         variable_data=variable_data)
        else:
            print("Empty union: Mergeability Index cannot be computed.")

def main():
    args = parse_args()

    if args.info:
        print(args.info)

    if args.mode == 'evaluate':
        assert args.guesses and args.test_dataset
        evaluate(args.test_dataset, args.guesses)
    elif args.mode == 'jaccard_index':
        assert args.guesses
        jaccard_index(args.guesses, args.generate_combination)
    elif args.mode == 'mergeability_index':
        assert args.guesses
        mergeability_index(args.guesses, args.generate_combination)


if __name__ == '__main__':
    main()
