import argparse
import os
import sys

sys.path.append(os.getcwd())

from script.utils.file_operations import download_file
from script.utils.file_operations import delete_file
from script.utils.file_operations import extract_zip
from script.utils.file_operations import extract_bz2
from script.utils.file_operations import extract_7z
from script.utils.file_operations import extract_rar
from script.utils.file_operations import remove_last_component
from script.utils.file_operations import change_extension
from script.utils.file_operations import get_dataset_name

from script.utils.format_datasets import count_to_plain
from script.utils.format_datasets import format_plain
from script.utils.format_datasets import email_to_plain

"""
Script used to download the following dataset and extract them to the specific directory:

    - Rockyou: https://drive.google.com/file/d/1VTSAAGMiVv6ioYXm7hGCH73xlEBqJAF7/view?usp=sharing
    - myspace: https://drive.google.com/file/d/1gl8F-RasplT3pZ5yqCZ-gSQjazJ-bwzV/view?usp=sharing
    - phpbb: https://drive.google.com/file/d/10CDKYXNbR9aauAPX0DWKX0BJG3PLpuG4/view?usp=sharing
    - LinkedIn: https://drive.google.com/file/d/10KT-my6FTrVwlDNr5wKXs2sWQgpO-sdm/view?usp=sharing
    - Hotmail: https://drive.google.com/file/d/1WYLZEVhMfaVxNAW8pMDwww8Ed7WJdi69/view?usp=sharing
    - Mail.ru: https://drive.google.com/file/d/1sjWNBc2lcjZJfBMjxT0MBQeXEhgbFjKF/view?usp=sharing
    - Yandex: https://drive.google.com/file/d/1WGb_f3tSUP6E9an7udzdr0VrJD9SAAYx/view?usp=sharing
    - Yahoo: https://drive.google.com/file/d/1ksfT7LbkFQVdOvTZauNFNh2Dz58ZYPr-/view?usp=sharing
    - Faith Writer: https://drive.google.com/file/d/1E419LMwpiuKWwANCqyNj2SDwwtGlfofs/view?usp=sharing
    - Hak5: https://drive.google.com/file/d/1ovbvFIOoGdd6GN7wgRoHk22ww4fecAPr/view?usp=sharing
    - Zomato: https://drive.google.com/file/d/1N6y8_qXRrOuspz9PLcHt0ZdnHoeooSlo/view?usp=sharing
    - 000webhost: https://drive.google.com/file/d/1U71v2dVYgBkfN_csoRtL2Dyg7xfWIonp/view?usp=sharing
    - Singles.org: https://drive.google.com/file/d/1KVZMcewPLeq037c43ES-Hqi8S48rUALh/view?usp=sharing
    - Taobao: https://drive.google.com/file/d/1mmz-9YIdH7W4eiSMfkOJ3XJD1sGoJaKf/view?usp=sharing
    - Gmail: https://drive.google.com/file/d/1fFnWziR2qUdw4-WoCia8mYSZsVE1HrIt/view?usp=sharing
    - Mate1.com: https://drive.google.com/file/d/1q-WrNn_0gdecpquGILWVQFyDk_RbZDlW/view?usp=sharing
    - Twitter: https://drive.google.com/file/d/1Q0z4H-yGRT6gPx2ANoeCdbKs1GOHlSj6/view?usp=sharing
    - Ashley Madison: https://drive.google.com/file/d/1n2VEbfvTLXT5PXqlWOFBgOk8rmXIGFYA/view?usp=sharing
    - Libero: https://drive.google.com/file/d/1y3kL-9cMp1T1PVTfZrEruZGuhtJDLK2J/view?usp=drive_link

"""

dict_datasets = {
    "rockyou": {
        "url": "https://drive.google.com/file/d/1VTSAAGMiVv6ioYXm7hGCH73xlEBqJAF7/view?usp=sharing",
        "filename": "rockyou",
        "ext": "7z",
        "format": "count",
        "language": "en",
        "service": "gaming",
    },
    "myspace": {
        "url": "https://drive.google.com/file/d/1gl8F-RasplT3pZ5yqCZ-gSQjazJ-bwzV/view?usp=sharing",
        "ext": "7z",
        "format": "count",
        "filename": "myspace",
        "language": "en",
        "service": "social-net",
    },
    "phpbb": {
        "url": "https://drive.google.com/file/d/10CDKYXNbR9aauAPX0DWKX0BJG3PLpuG4/view?usp=sharing",
        "filename": "phpbb",
        "ext": "7z",
        "format": "count",
        "language": "en",
        "service": "forum",
    },
    "linkedin": {
        "url": "https://drive.google.com/file/d/10KT-my6FTrVwlDNr5wKXs2sWQgpO-sdm/view?usp=sharing",
        "filename": "linkedin",
        "ext": "7z",
        "format": "email",
        "occurrence": "first",
        "language": "en",
        "service": "social-net",
    },
    "hotmail": {
        "url": "https://drive.google.com/file/d/1WYLZEVhMfaVxNAW8pMDwww8Ed7WJdi69/view?usp=sharing",
        "filename": "hotmail",
        "ext": "7z",
        "format": "count",
        "language": "en",
        "service": "mail",
    },
    "mailru": {
        "url": "https://drive.google.com/file/d/1sjWNBc2lcjZJfBMjxT0MBQeXEhgbFjKF/view?usp=sharing",
        "filename": "mailru",
        "ext": "7z",
        "format": "email",
        "occurrence": "first",
        "language": "ru",
        "service": "mail",
    },
    "yandex": {
        "url": "https://drive.google.com/file/d/1WGb_f3tSUP6E9an7udzdr0VrJD9SAAYx/view?usp=sharing",
        "filename": "yandex",
        "ext": "7z",
        "format": "email",
        "occurrence": "first",
        "language": "ru",
        "service": "web-portal",
    },
    "yahoo": {
        "url": "https://drive.google.com/file/d/1ksfT7LbkFQVdOvTZauNFNh2Dz58ZYPr-/view?usp=sharing",
        "filename": "yahoo",
        "ext": "7z",
        "format": "plain",
        "language": "en",
        "service": "web-portal",
    },
    "faithwriters": {
        "url": "https://drive.google.com/file/d/1E419LMwpiuKWwANCqyNj2SDwwtGlfofs/view?usp=sharing",
        "filename": "faithwriters",
        "ext": "7z",
        "format": "plain",
        "language": "en",
        "service": "forum",
    },
    "hak5": {
        "url": "https://drive.google.com/file/d/1ovbvFIOoGdd6GN7wgRoHk22ww4fecAPr/view?usp=sharing",
        "filename": "hak5",
        "ext": "7z",
        "format": "plain",
        "language": "en",
        "service": "forum",
    },
    "000webhost": {
        "url": "https://drive.google.com/file/d/1U71v2dVYgBkfN_csoRtL2Dyg7xfWIonp/view?usp=sharing",
        "filename": "000webhost",
        "ext": "7z",
        "format": "email",
        "occurrence": "last",
        "language": "en",
        "service": "forum",
    },
    "singles": {
        "url": "https://drive.google.com/file/d/1KVZMcewPLeq037c43ES-Hqi8S48rUALh/view?usp=sharing",
        "filename": "singles",
        "ext": "7z",
        "format": "count",
        "language": "en",
        "service": "dating-sites",
    },
    "gmail": {
        "url": "https://drive.google.com/file/d/1fFnWziR2qUdw4-WoCia8mYSZsVE1HrIt/view?usp=sharing",
        "filename": "gmail",
        "ext": "7z",
        "format": "email",
        "occurrence": "first",
        "language": "ru",
        "service": "mail",
    },
    "zomato": {
        "url": "https://drive.google.com/file/d/1N6y8_qXRrOuspz9PLcHt0ZdnHoeooSlo/view?usp=sharing",
        "filename": "zomato",
        "ext": "7z",
        "format": "email",
        "occurrence": "second",
        "language": "en",
        "service": "social-net",
    },
    "taobao": {
        "url": "https://drive.google.com/file/d/1mmz-9YIdH7W4eiSMfkOJ3XJD1sGoJaKf/view?usp=sharing",
        "filename": "taobao",
        "ext": "7z",
        "format": "plain",
        "language": "zh",
        "service": "e-commerce",
    },
    "mate1": {
        "url": "https://drive.google.com/file/d/1q-WrNn_0gdecpquGILWVQFyDk_RbZDlW/view?usp=sharing",
        "filename": "mate1",
        "ext": "7z",
        "format": "email",
        "occurrence": "last",
        "language": "en",
        "service": "dating-sites",
    },
    "twitter": {
        "url": "https://drive.google.com/file/d/1Q0z4H-yGRT6gPx2ANoeCdbKs1GOHlSj6/view?usp=sharing",
        "filename": "twitter",
        "ext": "7z",
        "format": "email",
        "occurrence": "first",
        "language": "ru",
        "service": "social-net",
    },
    "ashleymadison": {
        "url": "https://drive.google.com/file/d/1n2VEbfvTLXT5PXqlWOFBgOk8rmXIGFYA/view?usp=sharing",
        "filename": "ashleymadison",
        "ext": "7z",
        "format": "plain",
        "language": "en",
        "service": "social-net",
    },
    "libero": {
        "url": "https://drive.google.com/file/d/1y3kL-9cMp1T1PVTfZrEruZGuhtJDLK2J/view?usp=sharing",
        "filename": "libero",
        "ext": "7z",
        "format": "plain",
        "language": "it",
        "service": "mail",
    },
}


def construct_output_path(path, filename):
    # Concatenate the path up to the last occurrence of '/' with the filename and ".pickle" extension
    folder = remove_last_component(path)
    output_path = os.path.join(folder, filename + ".pickle")
    return output_path


def format_files(file):
    # Extract the dataset name from the file path
    filename = get_dataset_name(file)
    print(f"[I] - Formatting {filename} ...")

    # Get the format type from the dataset metadata
    format = dict_datasets[filename]["format"]

    # Construct the output path for the formatted file
    output_path = construct_output_path(file, filename)

    # Check if the file should be skipped (if it's already processed)
    skip = (check_if_already_processed(file, filename))

    # Format the file if it shouldn't be skipped
    if not skip:
        if format == "count":
            count_to_plain(file, output_path)
        elif format == "plain":
            format_plain(file, output_path)
        elif format == "email":
            mode = dict_datasets[filename]["occurrence"]
            email_to_plain(file, output_path, mode)
        print("[I] - Formatted!")
        # Delete the original file
        delete_file(file)
    else:
        print("[I] - Skipping: Dataset already formatted.")


def extract_files(file):
    # Extract the dataset name from the file path
    filename = get_dataset_name(file)
    print(f"[I] - Extracting {filename} ...")

    # Get the file extension from the dataset metadata
    ext = dict_datasets[filename]["ext"]

    # Check if the file is already in txt format
    if ext == "txt":
        # If it is already uncompressed txt, just return the path
        print("[I] - Skipping: Dataset already extracted.")
        return file
    else:
        # Change the extension of the output file to txt
        output_path = change_extension(file, "txt")

        # Check if the file should be skipped (if it's already extracted or processed)
        skip = (check_if_already_extracted(file) or
                check_if_already_processed(file, filename))

        # Extract the file if it shouldn't be skipped
        if not skip:
            if ext == "zip":
                extract_zip(file, output_path)
            elif ext == "bz2":
                extract_bz2(file, output_path)
            elif ext == "7z":
                extract_7z(file, output_path)
            elif ext == "rar":
                extract_rar(file, output_path)
            else:
                raise ValueError("Unsupported format: {}".format(ext))

            # Delete the compressed version of the file from the folder
            delete_file(file)

        else:
            print("[I] - Skipping: Dataset already extracted.")

    print("[I] - Extracted!")
    return output_path


def check_if_already_processed(path, filename):
    # Get the parent directory of the given path
    parent_directory = remove_last_component(path)

    # Construct the path to the corresponding pickle file
    pickle_path = os.path.join(parent_directory, filename + str('.pickle'))

    # Check if the pickle file exists
    already_processed = (1 if os.path.isfile(pickle_path) else 0)
    return already_processed


def check_if_already_extracted(path):
    # Construct the path to the corresponding txt file
    txt_path = change_extension(path, '.txt')

    # Check if the txt file exists
    already_extracted = (1 if os.path.isfile(txt_path) else 0)
    return already_extracted


def check_if_already_downloaded(path):
    already_downloaded = (1 if os.path.exists(path) else 0)
    return already_downloaded


def download_files(dataset, folder):
    print(f"[I] - Downloading {dataset} ...")

    # Retrieve information about the dataset
    link = dict_datasets[dataset]["url"]
    filename = dict_datasets[dataset]["filename"]
    language = dict_datasets[dataset]["language"]
    service = dict_datasets[dataset]["service"]
    format = dict_datasets[dataset]["format"]
    ext = dict_datasets[dataset]["ext"]

    # Build the output path
    output_path = os.path.join(folder, language, service,
                               "unformatted_" + str(format) + "_" + str(filename) + "." + str(ext))

    # Check if the file should be skipped (if it's already downloaded, extracted, or processed)
    skip = (check_if_already_downloaded(output_path) or
            check_if_already_extracted(output_path) or
            check_if_already_processed(output_path, filename))

    # Download the file if it shouldn't be skipped
    if not skip:
        download_file(link, output_path)
        print("[I] - Done!")
    else:
        print("[I] - Skipping: Dataset already downloaded.")

    return output_path


def generate_direct_download_link(dataset):
    # Get the URL of the dataset from the dictionary
    link = dict_datasets[dataset]["url"]
    if "drive.google.com" in link:
        # Extract the file ID from the URL
        file_id = link.split('/')[-2]
        # Construct the direct download link using the file ID
        direct_link = "https://drive.google.com/uc?export=download&id=" + str(file_id)
        # Update the URL in the dictionary with the direct download link
        dict_datasets[dataset]["url"] = str(direct_link)


def parse_args():
    parser = argparse.ArgumentParser()

    # Add argument for datasets
    parser.add_argument(
        '--datasets',
        type=str,
        nargs="+",
        choices=list(dict_datasets.keys()),
        default=list(dict_datasets.keys()),
        help="List of datasets to download and extract.",
    )

    # Add argument for output path
    parser.add_argument('--output_path',
                        default='./datasets',
                        help='Path to datasets folder')

    # Parse the command line arguments
    args = parser.parse_args()
    return args


def main(args):
    print("[I] - Starting Process")

    chosen_datasets = args.datasets
    output_path = args.output_path

    for dataset in chosen_datasets:

        if "url" not in dict_datasets[dataset]:
            continue

        # Create Google Drive direct download link
        generate_direct_download_link(dataset)

        # Download and extract files
        downloaded_file = download_files(dataset, output_path)

        extracted_file = extract_files(downloaded_file)

        # Format to one a .pickle file with one psw per line.
        format_files(extracted_file)

    print("[I] - Done")


if __name__ == "__main__":
    args = parse_args()
    main(args)
