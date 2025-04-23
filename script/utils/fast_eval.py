import os
import gzip
import glob

from script.utils.file_operations import load_pickle
from script.utils.file_operations import load_guesses_chunk


def check_if_fast_eval(settings):
    sub_sample_from = settings["use_existing_samples_path"] if "use_existing_samples_path" in settings.keys() else False
    sub_sample_from = check_sub_sample(sub_sample_from)

    skip_generation = settings["guesses_dir"] if "guesses_dir" in settings.keys() else False
    skip_generation = check_skip_generation(skip_generation)

    fast_eval_flag = bool(sub_sample_from or skip_generation)

    return fast_eval_flag, sub_sample_from, skip_generation


def check_sub_sample(sub_sample_from):
    if not sub_sample_from:
        return False

    samples_files = os.path.join(sub_sample_from, "*.gz")
    n_samples_files = len(glob.glob(samples_files))

    if not (os.path.exists(sub_sample_from) and n_samples_files):
        return False

    return sub_sample_from


def check_skip_generation(skip_generation):
    if not skip_generation:
        return False

    samples_files = os.path.join(skip_generation, "*.gz")
    n_samples_files = len(glob.glob(samples_files))

    if not (os.path.exists(skip_generation) and n_samples_files):
        return False

    return skip_generation


def check_mode(dir, mode):
    if not dir:
        return []

    output_array = []
    samples_files = os.path.join(dir, mode)
    n_samples_files = len(glob.glob(samples_files))
    if n_samples_files:
        name = mode.replace("*", "").rstrip(".gz")
        output_array.append(name)

    return output_array


def start_fast_eval(sub_sample_from, skip_generation, n_samples, test_path, output_path, modes=[""]):
    for mode in modes:
        output_path = os.path.join(output_path, "guesses", "sub-sampled" + mode + ".gz")

        if sub_sample_from and not skip_generation:
            try:
                source_path = os.path.join(sub_sample_from, "*" + str(mode) + ".gz")
                source_file = glob.glob(source_path)[0]
            except Exception:
                raise FileNotFoundError(f"No file matches {source_path}.")
            sub_sample(source_file, n_samples, output_path)
            output = fast_eval(test_path, n_samples, output_path)

        elif skip_generation and not sub_sample_from:
            try:
                source_path = os.path.join(skip_generation, "*" + str(mode) + ".gz")
                source_file = glob.glob(source_path)[0]
            except Exception:
                raise FileNotFoundError(f"No file matches {source_path}.")
            output = fast_eval(test_path, n_samples, source_file)

        elif skip_generation and sub_sample_from:
            try:
                source_path = os.path.join(skip_generation, "*" + str(mode) + ".gz")
                source_file = glob.glob(source_path)[0]
            except Exception:
                raise FileNotFoundError(f"No file matches {source_path}.")
            sub_sample(source_file, n_samples, output_path)
            output = fast_eval(test_path, n_samples, output_path)
    return output

def sub_sample(file_in, n_samples, file_out):
    print(f'[I] - Sub-sampling {n_samples} from {file_in}.')
    new_passwords = []
    with gzip.open(file_in, 'rt') as f_in:
        for i, line in enumerate(f_in):
            new_passwords.append(line)
            if i >= (n_samples - 1):
                break
    with gzip.open(file_out, 'wt') as f_out:
        f_out.writelines(new_passwords)
    return new_passwords



def fast_eval(test_file, n_samples, guesses_file):
    print(f'[I] - Starting fast eval mode. :)')
    print(f'[I] - Guesses file: {guesses_file}')
    print(f'[I] - Test file: {test_file}')

    output = []

    total_passwords = 0
    test_passwords = set(load_pickle(test_file).split("\n"))

    matches = set()

    thresholds = sorted([1000000, 2500000, 5000000, 7500000, 10000000, 25000000, 50000000, 75000000, 100000000, 250000000, 500000000])

    for chunk in load_guesses_chunk(guesses_file):
        guesses_set = set(chunk)
        total_passwords += len(guesses_set)

        current_match = guesses_set & test_passwords

        matches.update(current_match)

        if total_passwords >= n_samples:
            break

        if total_passwords >= thresholds[0]:
            total_matches = len(matches)
            test_size = len(test_passwords)
            match_percentage = f'{(total_matches / len(test_passwords)) * 100:.2f}%'
            print(f'[{thresholds[0]}] - {total_matches} matches found ({match_percentage} of test set).')
            output.append([test_size, thresholds[0], total_matches, match_percentage])
            thresholds.pop(0)

    total_matches = len(matches)
    test_size = len(test_passwords)
    match_percentage = f'{(total_matches / len(test_passwords)) * 100:.2f}%'
    print(
        f'[{n_samples}] - {total_matches} matches found ({match_percentage} of test set).')
    output.append([test_size, n_samples, total_matches, match_percentage])
    return output