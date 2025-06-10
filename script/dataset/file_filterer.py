import os
from script.config.config import read_config
from script.utils.download_raw_data import dict_datasets

PATH_TO_CONFIG = "./config/dataset/dataset_settings.yaml"

class FileFilterer:
    def __init__(self):
        self.dataset_settings = read_config(PATH_TO_CONFIG)
        self.datasets_path = str(self.dataset_settings["datasets_folder"])
        self.train_and_test_path = str(self.dataset_settings["train_and_test_folder"])

    def select_from_names(self, name):
        selected_datasets = (
            os.path.join(self.datasets_path,
                         dict_datasets[name]["language"],
                         dict_datasets[name]["service"],
                         dict_datasets[name]["filename"]+".pickle")
        )

        assert selected_datasets, f"{name} is not a valid dataset."
        return selected_datasets