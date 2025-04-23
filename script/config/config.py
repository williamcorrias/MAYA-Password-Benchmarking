import yaml

def read_config(path_to_config):
    with open(path_to_config, 'r') as file_config:
        settings = yaml.safe_load(file_config)
        return settings


def read_model_args(model_settings, name):
    assert name in model_settings, f"{name} is not a valid model. Please specify a valid model name."
    sett = model_settings[name]

    assert "path_to_class" in sett, f"You must specify a path to the class of {name}, using path_to_class."
    path_to_class = sett["path_to_class"]

    assert "class_name" in sett, f"You must specify the name of the class of {name}, using class_name."
    class_name = sett["class_name"]

    assert "path_to_config" in sett, f"You must specify a path to the config file of {name}, using path_to_config."
    path_to_config = sett["path_to_config"]

    test_args = model_settings[name]["test_args"] if "test_args" in model_settings[name] else {}

    return path_to_class, class_name, path_to_config, test_args
