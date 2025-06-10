import os
import time
import shutil
import torch
import glob

from datetime import timedelta
from script.utils.file_operations import redirect_stdout, redirect_stderr, write_to_csv
from script.utils.memory_usage import reset_memory_info, print_memory_info
from script.utils.fast_eval import check_skip_generation, sub_sample, fast_eval
from script.config.config import read_config

class Model:
    def __init__(self, settings):
        self.settings = settings

        self._setup_paths()
        self._setup_settings()
        self._setup_logging()
        self._setup_parameters()
        self.prepare_data()
        self._setup_device()

        print(f"Using device: {self.device}")
        reset_memory_info(self.device)

        self.eval_args, self.checkpoint_dir, self.save_guesses, self.save_matches = self.prepare_model()

        self._setup_checkpoint()

        status = self._run_embedding()

        if not status:
            status = self._run_fast_eval()

        if not status:
            self._run_training_and_eval()

    def _setup_paths(self):
        self.train_path = self.settings["train_path"]
        self.test_path = self.settings["test_path"]
        self.config_file = self.settings["config_file"]

        self.output_path = os.path.join(
            self.settings["output_path"],
            self.settings["test_hash"]
        )

        log_filename = self.settings["output_path"].replace("/", "-").replace("results-", "")
        self.log_file = os.path.join("logs", f"{log_filename}.err")

    def _setup_settings(self):
        self.max_length = int(self.settings["max_length"])
        self.n_samples = max(self.settings["n_samples"])
        self.thresholds = sorted(self.settings["n_samples"])
        self.thresholds.remove(self.n_samples)
        self.train_hash = self.settings["train_hash"]
        self.test_hash = self.settings["test_hash"]
        self.test_args = self.settings["test_args"]
        self.path_to_checkpoint = self.settings["path_to_checkpoint"]
        self.autoload = int(self.settings["autoload"])
        self.overwrite = int(self.settings["overwrite"])
        self.display_logs = int(self.settings["display_logs"])
        self.settings["same_test"] = self.settings["guesses_dir"] in os.path.join(self.output_path, "guesses") \
            if "guesses_dir" in self.settings else False

    def _setup_logging(self):
        self.written_rows = {}
        if not self.display_logs:
            print("Redirecting stderr to " + self.log_file)
            redirect_stderr(self.log_file)

        os.makedirs(self.output_path, exist_ok=True)

        self.output_file = os.path.join(self.output_path, "output.txt")

        print(f"Redirecting stdout to {self.output_file}")
        redirect_stdout(self.output_file)

        print("-" * 40)

    def _setup_parameters(self):
        self.params = read_config(self.config_file)
        self.current_epoch = 0
        self.model_loaded = False
        self.time_delta = timedelta()
        self.checkpoint_time_delta = timedelta()

    def _setup_device(self):
        if torch.cuda.is_available():
            #free_gpu = select_gpu()
            #self.device = torch.device(f'cuda:{free_gpu}')
            self.device = torch.device(f'cuda')
        #elif torch.backends.mps.is_available(): # -> functionality is not guaranteed
        #    self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def _setup_checkpoint(self):
        self.checkpoint_id, old_checkpoint_name = self.get_checkpoint_id(self.checkpoint_dir)
        self.checkpoint_name = f"checkpoint{self.checkpoint_id}.pt"

        if self.path_to_checkpoint:
            self.checkpoint_name = self.use_checkpoint(self.path_to_checkpoint)
        elif self.autoload and old_checkpoint_name:
            self.checkpoint_name = old_checkpoint_name

    def _prepare_directories(self):
        if self.save_guesses:
            self.create_guesses_dir()
        if self.save_matches:
            self.create_matches_dir()

    def _run_embedding(self):
        if self.settings['data_to_embed']:
            try:
                self.load(self.checkpoint_name)
                self.plot_embedding(self.settings['data_to_embed'])
                return 1
            except Exception as e:
                print(f"Cannot load the checkpoint: {e}")
        return 0

    def _run_fast_eval(self):
        n_samples_to_evaluate = sorted(self.settings.get("n_samples"))

        if not self.overwrite:
            guesses_file = os.path.join(self.output_path, "guesses", "guesses.gz")
            if os.path.isfile(guesses_file):
                output = fast_eval(self.test_path, n_samples_to_evaluate, guesses_file)
                self.save_stats(output)
                return True

        sub_samples_from_file = str(self.settings.get("sub_samples_from_file", False))
        guesses_file = self.settings.get("guesses_file", False)

        sub_samples_from_file = check_skip_generation(sub_samples_from_file)
        guesses_file = check_skip_generation(guesses_file)

        if sub_samples_from_file:
            sub_sample(sub_samples_from_file, n_samples_to_evaluate)

        if guesses_file:
            output = fast_eval(self.test_path, n_samples_to_evaluate, guesses_file)
            self.save_stats(output)

        return sub_samples_from_file or guesses_file

    def _run_training_and_eval(self):
        start_eval_time = time.time()

        self._prepare_directories()
        self.start_train(self.checkpoint_name)

        matches, match_percentage, test_size = self.start_eval(self.checkpoint_name)

        end_eval_time = time.time()
        delta = timedelta(seconds=end_eval_time - start_eval_time)
        print(f"[T] - Sampling completed after: {delta} \n")
        print_memory_info(self.output_file, self.device)

        output = [[test_size, self.n_samples, matches, match_percentage]]
        self.save_stats(output)

        if len(self.thresholds) > 0:
            guesses_file = os.path.join(self.guesses_dir, "guesses.gz")
            output = fast_eval(self.test_path, self.thresholds, guesses_file)
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

    def prepare_data(self):
        raise NotImplementedError('This method should be implemented in the subclass.')

    def prepare_model(self):
        raise NotImplementedError('This method should be implemented in the subclass.')

    def plot_embedding(self, data):
        # you can skip implementing this
        raise NotImplementedError('This method should be implemented in the subclass.')

    def load(self, file_name):
        raise NotImplementedError('This method should be implemented in the subclass.')

    def create_guesses_dir(self):
        self.guesses_dir = os.path.join(self.output_path, "guesses")
        if not os.path.exists(self.guesses_dir):
            os.makedirs(self.guesses_dir, exist_ok=True)
        else:
            for filename in glob.glob(os.path.join(self.guesses_dir, "*")):
                os.remove(filename)

    def create_matches_dir(self):
        self.matches_dir = os.path.join(self.output_path, "matches")
        if not os.path.exists(self.matches_dir):
            os.makedirs(self.matches_dir, exist_ok=True)
        else:
            for filename in glob.glob(os.path.join(self.matches_dir, "*")):
                os.remove(filename)

    def use_checkpoint(self, path):
        if os.path.exists(path):
            dest_path = os.path.join(self.checkpoint_dir, "checkpoint" + str(self.checkpoint_id) + ".pt")

            dir1 = os.path.dirname(os.path.abspath(path))
            dir2 = os.path.dirname(os.path.abspath(dest_path))

            if dir1 != dir2:
                shutil.copyfile(path, dest_path)
                checkpoint_name = os.path.basename(dest_path)
            else:
                checkpoint_name = os.path.basename(path)

            return checkpoint_name

        else:
            raise FileNotFoundError(f'{path} does not exist.')

    def finalize_checkpoint(self, fname):
        source_path = os.path.join(self.checkpoint_dir, "mid-" + self.checkpoint_name)
        if os.path.isfile(source_path):
            output_path = os.path.join(self.checkpoint_dir, fname)
            os.rename(source_path, output_path)

    def get_checkpoint_id(self, path):
        counter = 1
        checkpoint_list = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path) and file.startswith("checkpoint") and file.endswith(".pt"):
                id = file.replace("checkpoint", "").replace(".pt", "")
                if id:
                    try:
                        id = int(id)
                        checkpoint_list.append((file, int(id)))
                        if id >= counter:
                            counter = id + 1
                    except Exception:
                        continue
                else:
                    checkpoint_list.append((file, 0))

        checkpoint_list = sorted(checkpoint_list, key=lambda x: x[1], reverse=True)

        if len(checkpoint_list) == 0:
            return counter, None

        return counter, checkpoint_list[0][0]

    def start_train(self, checkpoint_name):
        if checkpoint_name:
            print("[I] - Train mode selected. Searching for a checkpoint...")

            status = self.load(checkpoint_name)
            if not status:
                print("[I] - No checkpoints found. Proceeding with normal training.")
                self.train()
            else:
                print("[I] - Final checkpoint loaded successfully. Training already finished :).")

        else:
            print("[I] - Checkpoint not specified. Starting training from scratch.")
            self.train()

    def start_eval(self, checkpoint_name):
        print("[I] - Searching for a checkpoint for evaluation...")
        status = self.load(checkpoint_name)

        if not status:
            print("[I] - No checkpoint found. Starting the training model normally.")
            self.start_train(checkpoint_name)
            status = self.load(checkpoint_name)

        print("[I] - Checkpoint loaded successfully. Initiating model evaluation.")

        matches, match_percentage, test_size = self.evaluate(self.eval_args, self.save_matches, self.save_guesses)
        return matches, match_percentage, test_size

    def train(self):
        raise NotImplementedError('This method should be implemented in the subclass.')

    def evaluate(self, eval_args, save_matches, save_guesses):
        raise NotImplementedError('This method should be implemented in the subclass.')
