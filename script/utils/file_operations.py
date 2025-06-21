from zipfile import ZipFile
from rarfile import RarFile
import py7zr
import bz2
import os
import pickle
import gdown
import sys
import csv
import gzip


def extract_zip(zip_file, output_path):
    with ZipFile(zip_file, 'r') as zip_ref:
        # Get the list of files in the zip archive
        file_list = zip_ref.namelist()

        # Check if the zip file contains multiple files
        if len(file_list) > 1:
            raise ValueError(
                "Multiple files found in the zip file '{}'. Only single-file zip archives are supported.".format(
                    zip_file))

        # Extract each file from the zip archive
        for file in file_list:
            zip_ref.extract(file)
            # Change the filename to save the unzipped file in the correct directory.
            rename_file(file, output_path)


def extract_rar(rar_file, output_path):
    with RarFile(rar_file, 'r') as rar_ref:

        # Get the list of files in the rar archive
        file_list = rar_ref.namelist()

        # Check if the rar file contains multiple files
        if len(file_list) > 1:
            raise ValueError(
                "Multiple files found in the rar file '{}'. Only single-file rar archives are supported.".format(
                    rar_ref))

        # Extract each file from the rar archive
        for file in file_list:
            rar_ref.extract(file)
            # Change the filename to save the uncompressed file in the correct directory.
            rename_file(file, output_path)


def extract_bz2(bz2_file, output_path):
    with open(bz2_file, 'rb') as f:
        # Decompress the data from the bz2 file
        uncompressed_data = bz2.decompress(f.read())

        # Check if there are multiple files in the bz2 archive
        if uncompressed_data.count(b'\x00\x00\x00') > 1:
            raise ValueError(
                "Multiple files found in the bz2 file '{}'. Only single-file bz2 archives are supported.".format(
                    bz2_file))

        # Write the decompressed data to the output file
        with open(output_path, 'wb') as output_f:
            output_f.write(uncompressed_data)


def extract_7z(sevenzip_file, output_path):
    with py7zr.SevenZipFile(sevenzip_file, 'r') as seven_ref:
        # Get the list of files in the 7zip archive
        file_list = seven_ref.getnames()

        # Check if the 7zip file contains multiple files
        if len(file_list) > 1:
            raise ValueError(
                "Multiple files found in the 7z file '{}'. Only single-file 7z archives are supported.".format(
                    sevenzip_file))

        # Extract each file from the rar archive
        seven_ref.extractall()
        file_name = file_list[0]

        # Change the filename to save the uncompressed file in the correct directory.
        rename_file(file_name, output_path)


def change_extension(path, new_extension):
    # Find the index of the last '.' in the path
    last_dot_index = path.rfind('.')

    # If a '.' is found in the path
    if last_dot_index != -1:
        # Replace the existing extension with the new extension
        new_path = path[:last_dot_index] + "." + str(new_extension)
        return new_path

    # If no '.' is found in the path, simply append the new extension
    return path + "." + str(new_extension)


def get_dataset_name(path):
    # Find the index of the last '_' in the path
    starting_index = path.rfind('_')

    # Find the index of the last '.' in the path
    ending_index = path.rfind('.')

    # If no '.' is found in the path, return the substring starting from the last '_' to the end of the path
    if ending_index == -1:
        return path[starting_index + 1:]

    # Otherwise, return the substring starting from the last '_' up to the last '.' in the path
    return path[starting_index + 1:ending_index]


def remove_last_component(path):
    # Remove the last component (file or directory) from a given path.
    return os.path.dirname(path)


def rename_file(path, output_path):
    # Rename path to output_path
    if not os.path.exists(output_path):
        os.rename(path, output_path)


def delete_file(path):
    # Delete file from a given path
    os.remove(path)


def save_pickle(output_file, lines):
    # Save data to a pickle file.
    with open(output_file, 'wb') as fo:
        pickle.dump(lines, fo)


def load_guesses_chunk(path, chunk_size=2048):
    with gzip.open(path, "rt") as file:
        chunk = []
        for line in file:
            chunk.append(line.strip())
            if len(chunk) >= chunk_size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk


def load_pickle(path):
    with open(path, "rb") as file:
        data = pickle.load(file)
    return data


def download_file(link, path):
    gdown.download(link, output=path)


stdout_file = None
stderr_file = None


def redirect_stdout(to):
    global stdout_file
    if stdout_file:
        stdout_file.close()
    stdout_file = open(to, "a")
    sys.stdout = stdout_file


def redirect_stderr(to):
    global stderr_file
    if stderr_file:
        stderr_file.close()
    stderr_file = open(to, "w")
    sys.stderr = stderr_file


def reset_stdout():
    global stdout_file
    if stdout_file:
        stdout_file.close()
        stdout_file = None
    sys.stdout = sys.__stdout__


def reset_stderr():
    global stderr_file
    if stderr_file:
        stderr_file.close()
        stderr_file = None
    sys.stderr = sys.__stderr__


def save_split(passwords, path):
    with open(path, 'wb') as output_file:
        pickle.dump('\n'.join(passwords) + '\n', output_file)
    print('Pickled dataset saved')


def write_passwords_to_file(file_name, passwords):
    with gzip.open(file_name, 'at') as f:
        for password in passwords:
            result_string = ''.join(map(str, password))
            f.write(result_string + '\n')


def write_to_csv(path, fieldnames, fixed_data, variable_data):
    rows = []
    file_exists = os.path.isfile(path) and os.path.getsize(path) > 0
    with open(path, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        for data in variable_data:
            row = {fieldnames[i]: value for i, value in enumerate(fixed_data + data)}
            writer.writerow(row)
            rows.append(",".join(str(v) for v in row.values()))
    return rows


def read_files(path):
    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            data = f.read().split("\n")

    elif path.endswith('.txt'):
        with open(path, 'r') as f:
            data = f.read().split("\n")

    elif path.endswith('.pickle'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            if isinstance(data, str):
                data = data.split("\n")
    else:
        raise ValueError(f"Unsupported file format: {path}")

    return data
