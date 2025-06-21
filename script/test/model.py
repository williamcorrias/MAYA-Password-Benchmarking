import math
import os
import gzip
import time
import pickle
from tqdm import tqdm
import shutil
import torch
import glob

from datetime import timedelta
from script.utils.file_operations import redirect_stdout, redirect_stderr, write_to_csv
from script.utils.memory_usage import reset_memory_info, print_memory_info
from script.utils.fast_eval import check_skip_generation, sub_sample, fast_eval
from script.config.config import read_config


class Model:
    def __init__(self, s):
        self.settings = s

        self._parse_settings()
        self._prepare_paths()
        self._setup_logging()
        self._setup_device()

        # Dictionary containing the model parameters loaded from the .yaml config file.
        self.params = read_config(self.path_to_config_file)

        train_passwords = read_dataset(self.path_to_train_dataset)
        test_passwords = read_dataset(self.path_to_test_dataset)

        self.data = self.prepare_data(train_passwords, test_passwords, self.max_length)

        self._setup_checkpoint()

        status = self._run_embedding()

        if not status:
            status = self._run_fast_eval()

        if not status:
            self._run_training_and_eval()

    def _parse_settings(self):
        # --- General settings ---
        self.autoload = int(self.settings["autoload"])
        self.path_to_checkpoint = self.settings["path_to_checkpoint"]
        self.overwrite = int(self.settings["overwrite"])
        self.display_logs = int(self.settings["display_logs"])
        self.save_guesses = int(self.settings["save_guesses"])
        self.save_matches = int(self.settings["save_matches"])

        # --- Dataset related settings ---
        self.train_hash = self.settings["train_hash"]
        self.test_hash = self.settings["test_hash"]
        self.max_length = int(self.settings["max_length"])
        self.n_samples = max(self.settings["n_samples"])
        self.thresholds = sorted([s for s in self.settings["n_samples"] if s != self.n_samples])

        # --- Model related settings ---
        self.model_name = str(self.settings["model_name"])

        self.keep_uniques = False

    def _prepare_paths(self):
        # --- Dataset related paths ---
        self.path_to_train_dataset = self.settings["train_path"]
        self.path_to_test_dataset = self.settings["test_path"]

        # --- Model related paths ---
        self.path_to_config_file = self.settings["config_file"]
        self.path_to_checkpoint_dir = os.path.join("checkpoints", self.model_name, self.train_hash)
        os.makedirs(self.path_to_checkpoint_dir, exist_ok=True)

        # --- Output related paths ---
        self.path_to_results_dir = os.path.join(self.settings["output_path"], self.settings["test_hash"])
        os.makedirs(self.path_to_results_dir, exist_ok=True)
        self.path_to_output_file = os.path.join(self.path_to_results_dir, "log.out")
        self.path_to_error_file = os.path.join(self.path_to_results_dir, "log.err")
        self.path_to_guesses_dir = os.path.join(self.path_to_results_dir, "guesses")
        self.path_to_guesses_file = os.path.join(self.path_to_guesses_dir, "guesses.gz")
        self.path_to_matches_dir = os.path.join(self.path_to_results_dir, "matches")
        self.path_to_matches_file = os.path.join(self.path_to_matches_dir, "matches.gz")

    def _setup_logging(self):
        self.written_rows = {}
        # --- Redirect stderr ---
        if not self.display_logs:
            print(f"Redirecting stderr to {self.path_to_error_file}.")
            redirect_stderr(self.path_to_error_file)

        # --- Redirect stdout ---
        print(f"Redirecting stdout to {self.path_to_output_file}")
        redirect_stdout(self.path_to_output_file)
        print("-" * 40)

    def _setup_device(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        reset_memory_info(self.device)
        print(f"Selected device: {self.device}.")

    def _setup_checkpoint(self):
        next_id, latest_checkpoint = get_checkpoint_id(self.path_to_checkpoint_dir)
        self.checkpoint_name = f"checkpoint{next_id}.pt"

        if self.path_to_checkpoint:
            self.checkpoint_name = use_specified_checkpoint(self.path_to_checkpoint, self.path_to_checkpoint_dir, self.checkpoint_name)
            print(f"[I] - Using checkpoint: {self.path_to_checkpoint}")
        elif self.autoload and latest_checkpoint:
            self.checkpoint_name = latest_checkpoint
            print(f"[I] - Autoloading latest checkpoint: {latest_checkpoint}")
        else:
            print(f"[I] - No specific checkpoint or autoload passed.")

    def _run_fast_eval(self):
        n_samples_to_evaluate = sorted(self.settings.get("n_samples"))

        if not self.overwrite:
            if os.path.isfile(self.path_to_guesses_file):
                output = fast_eval(self.path_to_test_dataset, n_samples_to_evaluate, self.path_to_guesses_file)
                self.save_stats(output)
                return True

        sub_samples_from_file = str(self.settings.get("sub_samples_from_file", False))
        guesses_file = self.settings.get("guesses_file", False)

        sub_samples_from_file = check_skip_generation(sub_samples_from_file)
        guesses_file = check_skip_generation(guesses_file)

        if sub_samples_from_file:
            sub_sample(sub_samples_from_file, n_samples_to_evaluate)

        if guesses_file:
            output = fast_eval(self.path_to_test_dataset, n_samples_to_evaluate, guesses_file)
            self.save_stats(output)

        return sub_samples_from_file or guesses_file

    def _prepare_directories(self):
        if self.save_guesses:
            _create_and_clean_dir(self.path_to_guesses_dir)
        if self.save_matches:
            _create_and_clean_dir(self.path_to_matches_dir)

    def _run_training_and_eval(self):
        self._prepare_directories()

        self.start_train(self.checkpoint_name)

        start_eval_time = time.time()
        matches, match_percentage, test_size = self.start_eval(self.checkpoint_name)
        end_eval_time = time.time()
        delta = timedelta(seconds=end_eval_time - start_eval_time)
        print(f"[T] - Sampling completed after: {delta} \n.")
        print_memory_info(self.path_to_output_file, self.device)

        output = [[test_size, self.n_samples, matches, match_percentage]]
        self.save_stats(output)

        if len(self.thresholds) > 0:
            output = fast_eval(self.path_to_test_dataset, self.thresholds, self.path_to_guesses_file)
            self.save_stats(output)

    def save_stats(self, output):
        if output:
            fieldnames = ["model", "train-dataset", "test-settings", "test-hash", "test-size", "n_samples", "matches",
                          "match_percentage"]

            infos = self.settings["output_path"].split("/")
            csv_path = os.path.join(infos[0], infos[1], f"{infos[1]}.csv")
            model_name = infos[2]
            if "-" in model_name:
                model_name = model_name.replace("-", "")
            fixed_values = [model_name, infos[3], infos[4], self.test_hash]

            rows = write_to_csv(csv_path, fieldnames=fieldnames, fixed_data=fixed_values, variable_data=output)
            if csv_path not in self.written_rows:
                self.written_rows[csv_path] = []
            for row in rows:
                self.written_rows[csv_path].append(row)

    def plot_embedding(self, data, max_length):
        # you can skip implementing this
        raise NotImplementedError('This method should be implemented in the subclass.')

    def save(self, obj, mid=True):
        f_name = self.checkpoint_name if not mid else f"mid-{self.checkpoint_name}"
        save_path = os.path.join(self.path_to_checkpoint_dir, f_name)
        torch.save(obj, save_path)

    def finalize_checkpoint(self):
        source_path = os.path.join(self.path_to_checkpoint_dir, "mid-" + self.checkpoint_name)
        if os.path.isfile(source_path):
            output_path = os.path.join(self.path_to_checkpoint_dir, self.checkpoint_name)
            os.rename(source_path, output_path)

    def start_train(self, checkpoint_name):
        if checkpoint_name:
            print("[I] - Train mode selected. Searching for a checkpoint...")
            file_to_load = os.path.join(self.path_to_checkpoint_dir, (self.checkpoint_name))
            status = self.load(file_to_load)
            if not status:
                print("[I] - No checkpoints found. Proceeding with normal training.")
                self.train()
                self.finalize_checkpoint()
            else:
                print("[I] - Final checkpoint loaded successfully. Training already finished :).")

        else:
            print("[I] - Checkpoint not specified. Starting training from scratch.")
            self.train()
            self.finalize_checkpoint()

    def start_eval(self, checkpoint_name):
        print("[I] - Searching for a checkpoint for evaluation...")
        file_to_load = os.path.join(self.path_to_checkpoint_dir, self.checkpoint_name)
        status = self.load(file_to_load)

        if not status:
            print("[I] - No checkpoint found. Starting the training model normally.")
            self.start_train(checkpoint_name)
            status = self.load(file_to_load)

        print("[I] - Checkpoint loaded successfully. Initiating model evaluation.")

        matches, match_percentage, test_size = self.evaluate(self.n_samples)
        return matches, match_percentage, test_size

    def _run_embedding(self):
        if self.settings['data_to_embed']:
            try:
                file_to_load = os.path.join(self.path_to_checkpoint_dir, self.checkpoint_name)
                self.load(file_to_load)
                self.plot_embedding(self.settings['data_to_embed'], self.max_length)
                return True
            except NotImplementedError:
                print("[W] - plot_embedding method not implemented by the subclass. Skipping embedding.")
            except Exception as e:
                print(f"[E] - Error during embedding: {e}")
        return False

    def write_to_file(self, file, generated_data):
        with gzip.open(file, 'at') as file:
            for password in generated_data:
                decoded_password = self.data.decode_password(password)
                if decoded_password is None:
                    continue
                password = self.data.remove_padding(decoded_password)
                file.write(password + '\n')

    def prepare_data(self, train_passwords, test_passwords, max_length):
        """
        **TO BE IMPLEMENTED BY SUBCLASS.**

        This method should be used to create an object that manages and processes the dataset.

        The returned object must include at least the following attributes:
            - train_passwords (list): A list of training passwords.
            - test_passwords (set): A set of test passwords.

        Additionally, the object MUST implement the following methods:
            - encode_password(password): Takes a password and returns its tokenized representation.
            - decode_password(password): Takes a tokenized password and returns the decoded password.
            - remove_padding(password): Takes a password and removes any padding tokens.

        Parameters:
            - self (Model): The model instance. You can access all variables and methods defined in this class
            and in your model, including `self.params` (the configuration parameters).
            - train_passwords (list): The list of passwords used for training.
            - test_passwords (list): The list of passwords used for testing.
            - max_length (int): The maximum allowed password length.
        Returns:
             - An object containing the required attributes and methods, which will be later accessible via self.data.
        """
        raise NotImplementedError('This method should be implemented in the subclass.')

    def load(self, file_name):
        """
        **TO BE IMPLEMENTED BY SUBCLASS.**

        This method should load the model's state from the specified checkpoint file.

        Parameters:
            - self (Model): The model instance. You can access all variables and methods defined in this class
            and in your model, including `self.data` (the data object) and `self.params` (the configuration parameters).
            - file_name (str): Path to the checkpoint file.
        Returns:
            - int: Returns 1 if the model was successfully loaded, 0 otherwise.
        """
        raise NotImplementedError('This method should be implemented in the subclass.')

    def train(self):
        """
        **TO BE IMPLEMENTED BY SUBCLASS.**

        This method should train your model and save its state to a checkpoint file.

        To save a checkpoint, use the `self.save()` method from the base class by passing a dictionary containing
        all relevant model and optimizer states. For example:

        obj = {
            'generator_opt': self.generator_opt.state_dict(),
            'discriminator_opt': self.discriminator_opt.state_dict(),
            'Generator': self.Generator.state_dict(),
            'Discriminator': self.Discriminator.state_dict(),
        }
        self.save(obj)

        Parameters:
            - self (Model): The model instance. You can access all variables and methods defined in this class
            and in your model, including `self.data` (the data object) and `self.params` (the configuration parameters).
        Returns:
            - None
        """
        raise NotImplementedError('This method should be implemented in the subclass.')

    def evaluate(self, n_samples, validation_mode=False):
        print(f"Generating {n_samples} passwords...")
        save_every = 1000000
        save_guesses = self.save_guesses and not validation_mode
        save_matches = self.save_matches and not validation_mode

        evaluation_batch_size = int(self.params['eval']['evaluation_batch_size'])
        if n_samples < evaluation_batch_size:
            n_batches, evaluation_batch_size = 1, int(n_samples)
        else:
            n_batches = math.floor(n_samples / evaluation_batch_size)

        eval_dict = self.eval_init(n_samples, evaluation_batch_size)

        progress_bar = tqdm(range(n_batches))
        progress_bar.set_description(desc='Generating sample batch')

        self.guesses = []
        self.matches = set()

        for batch in range(n_batches):
            generated_passwords = self.sample(evaluation_batch_size, eval_dict)

            self.guesses.extend(generated_passwords)
            self.matches.update(generated_passwords & self.data.test_passwords)

            self.guessing_strategy(evaluation_batch_size, eval_dict)

            if save_guesses and len(self.guesses) >= save_every:
                if not self.keep_uniques:
                    self.write_to_file(self.path_to_guesses_file, self.guesses)
                    self.guesses = []

            progress_bar.set_postfix({'Matches found': {len(self.matches)},
                                      'Test set %': ({len(self.matches) / len(self.data.test_passwords) * 100.0})})
            progress_bar.update(1)

        self.post_sampling(eval_dict)

        if save_guesses and len(self.guesses) > 0:
            self.write_to_file(self.path_to_guesses_file, self.guesses)

        if save_matches:
            self.write_to_file(self.path_to_matches_file, self.matches)

        n_matches = len(self.matches)
        test_size = len(self.data.test_passwords)
        match_percentage = f'{(n_matches / test_size) * 100:.2f}%'
        print(f'{n_matches} matches found ({match_percentage} of test set).')
        return n_matches, match_percentage, test_size

    def eval_init(self, n_samples, evaluation_batch_size):
        """
        **TO BE IMPLEMENTED BY SUBCLASS.**

        This method should initialize all variables required for the evaluation process, according to your model's needs.

        For example, if you require certain variables later during the evaluation, you can initialize them here and
        return them in a dictionary.

        Parameters:
            - self (Model): The model instance. You can access all variables and methods defined in this class
            and in your model, including `self.data` (the data object) and `self.params` (the configuration parameters).
            - n_samples (int): The number of passwords to generate.
            - evaluation_batch_size (int): The batch size to use during evaluation.
        Returns:
            - eval_dict (dict): A dictionary containing all initialized resources.
        """
        raise NotImplementedError('This method should be implemented in the subclass.')

    def sample(self, evaluation_batch_size, eval_dict):
        """
        **TO BE IMPLEMENTED BY SUBCLASS.**

        This method should generate and return 'evaluation_batch_size' passwords.

        Make sure the generated passwords follow the same format of `self.data.test_passwords`;
        otherwise, no matches will be found during evaluation.

        Parameters:
            - self (Model): The model instance. You can access all variables and methods defined in this class
            and in your model, including `self.data` (the data object) and `self.params` (the configuration parameters).
            - evaluation_batch_size (int): Number of passwords to generate in this batch.
            - eval_dict (dict): Dictionary returned in `self.eval_init`.
        Returns:
            - generated_passwords (list): A list of generated passwords, matching the format of `self.data.test_passwords`.
        """
        raise NotImplementedError('This method should be implemented in the subclass.')

    def guessing_strategy(self, evaluation_batch_size, eval_dict):
        """
        **TO BE IMPLEMENTED BY SUBCLASS.**

        This method should implement your custom guessing strategy (see PassFlow with dynamic guessing as a reference).

        Leave this method blank if your model does not require a specific guessing strategy.

        Parameters:
            - self (Model): The model instance. You can access all variables and methods defined in this class
            and in your model, including `self.data` (the data object) and `self.params` (the configuration parameters).
            - evaluation_batch_size (int): Number of passwords generated in the current batch.
            - eval_dict (dict): Dictionary returned by `self.eval_init`.
        Returns:
            - None
        """
        raise NotImplementedError('This method should be implemented in the subclass.')

    def post_sampling(self, eval_dict):
        """
        **TO BE IMPLEMENTED BY SUBCLASS.**

        This method should handle any post-generation logic, such as cleaning up temporary files used during evaluation,
        resetting variables, or releasing resources.

        Parameters:
            - self (Model): The model instance. You can access all variables and methods defined in this class
            and in your model, including `self.data` (the data object) and `self.params` (the configuration parameters).
            - eval_dict (dict): Dictionary returned by `self.eval_init`.

        Returns:
            - None
        """
        raise NotImplementedError('This method should be implemented in the subclass.')


def read_dataset(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    data = data.split("\n")
    return data

def get_checkpoint_id(path):
    next_id = 1
    checkpoints = []

    for filename in os.listdir(path):
        if filename.startswith("checkpoint") and filename.endswith(".pt"):
            file_path = os.path.join(path, filename)
            if not os.path.isfile(file_path):
                continue

            id_str = filename[len("checkpoint"):-3]  # remove prefix and '.pt'
            try:
                checkpoint_id = int(id_str) if id_str else 0
                checkpoints.append((filename, checkpoint_id))
                next_id = max(next_id, checkpoint_id + 1)
            except ValueError:
                continue  # Skip malformed IDs

    # Sort checkpoints by ID descending
    checkpoints.sort(key=lambda x: x[1], reverse=True)

    # Return next available ID and latest checkpoint filename (or None)
    latest_checkpoint = checkpoints[0][0] if checkpoints else None
    return next_id, latest_checkpoint

def use_specified_checkpoint(source_path, target_dir, file_name):
    if not os.path.exists(source_path):
        raise FileNotFoundError(f"Checkpoint file not found: {source_path}")

    dest_path = os.path.join(target_dir, file_name)

    if os.path.abspath(os.path.dirname(source_path)) != os.path.abspath(target_dir):
        shutil.copyfile(source_path, dest_path)
        return file_name
    else:
        return os.path.basename(source_path)

def _create_and_clean_dir(path):
    os.makedirs(path, exist_ok=True)

    for filename in glob.glob(os.path.join(path, "*")):
        os.remove(filename)