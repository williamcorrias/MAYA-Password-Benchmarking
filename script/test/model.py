import os
import time
import shutil
import torch
import glob

from datetime import timedelta
from script.utils.file_operations import redirect_stdout, redirect_stderr, write_to_csv
from script.utils.memory_usage import reset_memory_info, print_memory_info
from script.utils.fast_eval import check_if_fast_eval, start_fast_eval
from script.config.config import read_config
from script.utils.gpu import select_gpu

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

        self._run_embedding()
        self._run_fast_eval()
        self._run_training_and_eval()

    def _setup_settings(self):
        self.max_length = int(self.settings["max_length"])
        self.n_samples = int(self.settings["n_samples"])
        self.train_hash = self.settings["train_hash"]
        self.test_hash = self.settings["test_hash"]
        self.test_args = self.settings["test_args"]
        self.path_to_checkpoint = self.settings["path_to_checkpoint"]
        self.autoload = int(self.settings["autoload"])
        self.display_logs = int(self.settings["display_logs"])
        self.modes = self.settings["modes"] if "modes" in self.settings else [""]
        self.settings["same_test"] = self.settings["guesses_dir"] in os.path.join(self.output_path, "guesses") \
            if "guesses_dir" in self.settings else False

    def _setup_paths(self):
        self.train_path = self.settings["train_path"]
        self.test_path = self.settings["test_path"]
        self.config_file = self.settings["config_file"]
        self.output_path = os.path.join(self.settings["output_path"], self.settings["train_hash"])
        self.log_file = os.path.join("logs", self.settings["output_path"].replace("/", "-").replace("results-", ""))+".err"

    def _setup_logging(self):
        if not self.display_logs:
            print("Redirecting stderr to " + self.log_file)
            redirect_stderr(self.log_file)

        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)

        self.output_file = os.path.join(self.output_path, "output.txt")

        print("Redirecting stdout to " + self.output_file)
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
        #elif torch.backends.mps.is_available():
        #    self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

    def _setup_checkpoint(self):
        self.checkpoint_id, old_checkpoint_name = self.get_checkpoint_id(self.checkpoint_dir)
        self.checkpoint_name = "checkpoint" + str(self.checkpoint_id) + ".pt"

        if self.path_to_checkpoint:
            self.checkpoint_name = self.use_checkpoint(self.path_to_checkpoint)
        elif self.autoload:
            if old_checkpoint_name is not None:
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
        fast_eval_flag, sub_sample_from, skip_generation = check_if_fast_eval(self.settings)
        if fast_eval_flag:
            output = start_fast_eval(sub_sample_from, skip_generation, self.n_samples, self.test_path, self.output_path, self.modes)
            fieldnames = ["model", "train-dataset", "test-settings", "test-hash", "test-size", "#gen", "#matches",
                          "match_percentage", "#uniques"]
            infos = self.settings["output_path"].split("/")
            csv_path = os.path.join(infos[0], infos[1], "output.csv")
            fixed_values = [infos[2], infos[3], infos[4], self.test_hash]
            write_to_csv(csv_path, fieldnames=fieldnames, fixed_data=fixed_values, variable_data=output)
            return 1
        return 0

    def _run_training_and_eval(self):
        output = []
        start_eval_time = time.time()

        self._prepare_directories()
        self.start_train(self.checkpoint_name)
        matches, match_percentage, test_size = self.start_eval(self.checkpoint_name)
        output.append([test_size, self.n_samples, matches, match_percentage])

        end_eval_time = time.time()
        delta = timedelta(seconds=end_eval_time - start_eval_time)
        print(f"[T] - Sampling completed after: {delta} \n")
        print_memory_info(self.output_file, self.device)

        fieldnames = ["model", "train-dataset", "test-settings", "test-hash", "test-size", "#gen", "#matches", "match_percentage", "#uniques"]
        infos = self.settings["output_path"].split("/")
        csv_path = os.path.join(infos[0], infos[1], "output.csv")
        fixed_values = [infos[2], infos[3], infos[4], self.test_hash]
        write_to_csv(csv_path, fieldnames=fieldnames, fixed_data=fixed_values, variable_data=output)

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
            raise FileNotFoundError('{path} does not exist.'.format(path=path))

    def finalize_checkpoint(self, fname):
        source_path = os.path.join(self.checkpoint_dir, "mid-"+self.checkpoint_name)
        if os.path.isfile(source_path):
            output_path = os.path.join(self.checkpoint_dir, fname)
            os.rename(source_path, output_path)

    def get_checkpoint_id(self, path):
        counter = 1
        checkpoint_list = []
        for file in os.listdir(path):
            file_path = os.path.join(path, file)
            if os.path.isfile(file_path) and file.startswith("checkpoint") and file.endswith(".pt"):
                id = file.replace("checkpoint","").replace(".pt", "")
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