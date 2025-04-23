import os
from script.config.config import read_config
from script.utils.download_raw_data import dict_datasets

PATH_TO_CONFIG = "./config/dataset/dataset_settings.yaml"

class FileFilterer:
    def __init__(self):
        self.dataset_settings = read_config(PATH_TO_CONFIG)
        self.datasets_path = str(self.dataset_settings["datasets_folder"])
        self.train_and_test_path = str(self.dataset_settings["train_and_test_folder"])

        self.choose_by_categories = self.dataset_settings["choose_by_categories"]["enabled"]
        self.choose_by_names = self.dataset_settings["choose_by_names"]["enabled"]

        if self.choose_by_categories:
            self.allowed_languages = self.dataset_settings["choose_by_categories"]["languages"]
            self.allowed_services = self.dataset_settings["choose_by_categories"]["services"]

        if self.choose_by_names:
            self.allowed_names = self.dataset_settings["choose_by_names"]["names"]

    def from_paths_to_names(self, paths):
        dataset_names = [(os.path.basename(path)).rstrip(".pickle") for path in paths]
        return dataset_names

    def select_from_categories(self):
        selected_files = [
            os.path.join(lang_dir, serv_dir, file)
            for lang_dir in os.listdir(self.datasets_path)
            if lang_dir in self.allowed_languages
            for serv_dir in os.listdir(os.path.join(self.datasets_path, lang_dir))
            if serv_dir in self.allowed_services
            for file in os.listdir(os.path.join(self.datasets_path, lang_dir, serv_dir))
            if file.endswith(".pickle")
        ]
        return selected_files

    def select_from_names(self, names):
        selected_datasets = [
            os.path.join(self.datasets_path,
                         dict_datasets[dataset]["language"],
                         dict_datasets[dataset]["service"],
                         dict_datasets[dataset]["filename"]+".pickle")
            for dataset in dict_datasets if dataset in names.lower()
        ]
        assert selected_datasets, f"{names} is not a valid dataset."
        return selected_datasets

    def get_datasets_from_default_settings(self):
        selected_train_datasets = []
        if self.choose_by_categories:
            datasets = self.select_from_categories()
            selected_train_datasets_from_categories = self.from_paths_to_names(datasets)
            selected_train_datasets.extend(selected_train_datasets_from_categories)

        if self.choose_by_names:
            selected_train_datasets_from_names = self.allowed_names
            selected_train_datasets.extend(selected_train_datasets_from_names)
        return selected_train_datasets