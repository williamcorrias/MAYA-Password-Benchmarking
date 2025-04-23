import os
import hashlib


def write_hash_settings(hash, key_list, values, mode):
    hash_dir = os.path.join(os.getcwd(), "hash", hash)

    os.makedirs(hash_dir, exist_ok=True)

    hash_file = os.path.join(hash_dir, str(mode) + "-settings.txt")

    with open(hash_file, "w") as out:
        for key, value in zip(key_list, values):
            out.write("{}={}\n".format(key, value))


def construct_hash(settings, dict_params_to_type, mode):
    final_string = ""

    key_list = sorted(list(settings.keys()))

    if mode == "train":
        key_list = [key for key in key_list if dict_params_to_type[key] not in ["general_params", "test_params"]]
    elif mode == "test":
        key_list = [key for key in key_list if dict_params_to_type[key] not in ["general_params"]]

    values = [str(settings[key]) for key in key_list]

    for key, value in zip(key_list, values):
        final_string = final_string + key + "=" + value + "\n"

    hash = hashlib.md5(final_string.encode()).hexdigest()
    write_hash_settings(hash, key_list, values, mode)

    return hash
