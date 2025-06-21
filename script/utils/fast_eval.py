import os
import gzip

from script.utils.file_operations import load_pickle
from script.utils.file_operations import load_guesses_chunk


def check_skip_generation(path):
    if not path or not os.path.exists(path):
        return False
    return path


def sub_sample(file_in, thresholds):
    tmp_out = file_in.split(os.sep)

    for n_samples in thresholds:
        print(f'[I] - Sub-sampling {n_samples} from {file_in}.')

        tmp_out[5] = str(n_samples)
        file_out = os.path.join(*tmp_out)
        os.makedirs(os.path.dirname(file_out), exist_ok=True)

        with gzip.open(file_out, 'wt') as f_out:
            with gzip.open(file_in, 'rt') as f_in:
                for i, line in enumerate(f_in):
                    f_out.write(line)
                    if i >= n_samples - 1:
                        break
        print(f'[I] - Done!')


def fast_eval(test_file, thresholds, guesses_file):
    print(f'[I] - Starting fast eval mode. :)')
    print(f'[I] - Guesses file: {guesses_file}')
    print(f'[I] - Test file: {test_file}')

    output = []

    total_passwords = 0
    test_passwords = set(load_pickle(test_file).split("\n"))

    matches = set()

    for chunk in load_guesses_chunk(guesses_file):
        total_passwords += len(chunk)

        guesses_set = set(chunk)

        current_match = guesses_set & test_passwords

        matches.update(current_match)

        if total_passwords >= thresholds[0]:
            total_matches = len(matches)
            test_size = len(test_passwords)
            match_percentage = f'{(total_matches / len(test_passwords)) * 100:.2f}%'
            print(f'[{thresholds[0]}] - {total_matches} matches found ({match_percentage} of test set).')
            output.append([test_size, thresholds[0], total_matches, match_percentage])
            thresholds.pop(0)

        if len(thresholds) == 0:
            break

    return output
