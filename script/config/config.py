import yaml
import os

PATH_TO_MODEL_CONFIG = "./config/model/model_settings.yaml"

CHAR_BAG_CHARS_NUMBERS_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789~!@#$%^&*(),.<>/?'\"{}[]\\|-_=+;: `"
CHAR_BAG_CHARS_NUMBERS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
CHAR_BAG_CHARS_SYMBOLS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ~!@#$%^&*(),.<>/?'\"{}[]\\|-_=+;: `"
CHAR_BAG_CHARS = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ"
CHAR_BAG_NUMBERS = "0123456789"

char_bag_mapping = {
    CHAR_BAG_CHARS_NUMBERS_SYMBOLS: "all",
    CHAR_BAG_CHARS_NUMBERS: "c+n",
    CHAR_BAG_CHARS_SYMBOLS: "c+s",
    CHAR_BAG_CHARS: "c",
    CHAR_BAG_NUMBERS: "n"
}

inverse_char_bag_mapping = {
    "all": CHAR_BAG_CHARS_NUMBERS_SYMBOLS,
    "c+n": CHAR_BAG_CHARS_NUMBERS,
    "c+s": CHAR_BAG_CHARS_SYMBOLS,
    "c": CHAR_BAG_CHARS,
    "n": CHAR_BAG_NUMBERS,
}

def args_to_dict(obj):
    if isinstance(obj, dict):
        return obj
    elif hasattr(obj, "__dict__"):
        return vars(obj)
    raise ValueError("Input must be a dict or argparse.Namespace")


def build_args_settings(dict):

    return {
        "general_params": {
            "models": dict.get("models"),
            "display_logs": dict.get("display_logs"),
            "autoload": dict.get("autoload"),
            "overwrite": dict.get("overwrite"),
            "path_to_checkpoint": dict.get("path_to_checkpoint"),
            "n_samples": dict.get("n_samples"),
            "test_config": dict.get("test_config"),
            "guesses_file": dict.get("guesses_file"),
            "sub_samples_from_file": dict.get("sub_samples_from_file"),
            "data_to_embed": dict.get("data_to_embed"),
            "save_guesses": dict.get("save_guesses"),
            "save_matches": dict.get("save_matches"),
        },
        "pre_split_params": {
            "max_length": dict.get("max_length"),
            "char_bag": dict.get("char_bag"),
            "train_datasets": dict.get("train_datasets"),
        },
        "split_params": {
            "train_split_percentage": dict.get("train_split_percentage"),
        },
        "post_split_params": {
            "train_chunk_percentage": dict.get("train_chunk_percentage"),
        },
        "test_params": {
            "test_datasets": dict.get("test_datasets"),
            "test_frequency": dict.get("test_frequency"),
        },
    }


def read_config(path_to_config):
    if not os.path.exists(path_to_config):
        return {}

    with open(path_to_config, 'r') as file_config:
        settings = yaml.safe_load(file_config)
        return settings


def read_args_settings(args):
    args_settings = {}
    for type in args:
        args_settings[type] = {}
        for key in args[type]:
            val = args[type][key]
            if val is not None:
                args_settings[type][key] = val
    return args_settings


def read_model_args(model_settings, name):
    assert name in model_settings, f"{name} is not a valid model. Please specify a valid model name."
    sett = model_settings[name]

    assert "path_to_class" in sett, f"You must specify a path to the class of {name}, using path_to_class."
    path_to_class = sett["path_to_class"]

    assert "class_name" in sett, f"You must specify the name of the class of {name}, using class_name."
    class_name = sett["class_name"]

    assert "path_to_config" in sett, f"You must specify a path to the config file of {name}, using path_to_config."
    path_to_config = sett["path_to_config"]

    sampling_args = model_settings[name]["sampling_args"] if "sampling_args" in model_settings[name] else {}

    return path_to_class, class_name, path_to_config, sampling_args


def map_param_to_type(settings):
    dict_param_to_type = {}
    for test_name, test_dict in settings.items():
        for params_type in test_dict:
            for option in test_dict[params_type]:
                dict_param_to_type[option] = params_type
    return dict_param_to_type


def update_settings(args_settings, test_settings):
    def update_values(source, target, params_type):
        if params_type not in source or source[params_type] is None:
            return

        for option, val in source[params_type].items():
            if option == "test_config":
                continue

            if isinstance(val, dict):
                for key, value in val.items():
                    value = value if isinstance(value, list) else [value]
                    if key not in target:
                        target[key] = value
            else:
                value = val if isinstance(val, list) else [val]
                if option not in target:
                    target[option] = value

    final_settings = {}

    for test_name, test_dict in test_settings.items():
        final_settings[test_name] = {}

        for param_type in ['general_params', 'pre_split_params', 'split_params',
                           'post_split_params', 'test_params']:
            final_settings[test_name][param_type] = {}

            update_values(args_settings, final_settings[test_name][param_type], param_type)
            update_values(test_dict, final_settings[test_name][param_type], param_type)

    return final_settings


def get_keys_and_values(d):
    result = {}
    for value in d.values():
        if isinstance(value, dict):
            deeper = get_keys_and_values(value)
            result.update(deeper)
    if all(not isinstance(v, dict) for v in d.values()):
        result.update(d)
    return result
