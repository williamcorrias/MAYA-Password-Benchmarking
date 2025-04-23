import os
import gzip
import glob
import shutil
import sys

sys.path.append(os.getcwd())

path = "results/"


def compress_txt_file(input_file, output_file):
    try:
        with open(input_file, 'rb') as f_in:
            with gzip.open(output_file, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    except Exception as e:
        print(f"Error during compression: {e}")
        raise


def find_txt_files(starting_path):
    pattern = os.path.join(starting_path, '**', '*.txt')
    txt_files = glob.glob(pattern, recursive=True)
    filtered_txt_files = [file for file in txt_files if not file.endswith('output.txt')]
    return filtered_txt_files


def main():
    txt_files = find_txt_files(path)
    counter = 0
    for file in txt_files:
        counter += 1
        output_name = file.replace("txt", "gz")

        print("Compressing ", file, " into ", output_name)
        compress_txt_file(file, output_name)

        print("Removing ", file)
        if os.path.exists(output_name):
            os.remove(file)

        print("Done! ", counter, "/", len(txt_files))


if __name__ == '__main__':
    main()