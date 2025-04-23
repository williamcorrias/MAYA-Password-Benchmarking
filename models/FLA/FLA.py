import glob
from datetime import timedelta
import os
import torch
import time
import numpy as np
import torch.nn.functional as F
from tqdm import tqdm
import heapq
import gc
import gzip

from script.test.model import Model

from models.FLA.architecture import LSTM
from models.FLA.guesser import Guesser
from models.FLA.fla_utils.dataloader import *

from script.utils.file_operations import write_passwords_to_file

class FLA(Model):
    def __init__(self, settings):
        super().__init__(settings)

    def prepare_data(self):
        self.data = DataLoader(self.train_path, self.test_path, self.max_length, self.params)

    def prepare_model(self):
        self.model = None
        self.params_train = self.params["train"]

        self.chunk = 0
        self.generation = 0

        self.eval_args = {
            'n_samples': self.n_samples,
            'lower_probability_threshold': self.get_lower_probability_threshold(self.n_samples),
        }

        self.checkpoint_frequency = self.params['eval']['checkpoint_frequency']
        self.checkpoint_dir = os.path.join("checkpoints", "FLA", self.train_hash)
        if not os.path.exists(os.path.join(os.getcwd(), self.checkpoint_dir)):
            os.makedirs(os.path.join(os.getcwd(), self.checkpoint_dir), exist_ok=True)

        self.save_guesses = self.params['eval']['save_guesses']
        self.save_matches = self.params['eval']['save_matches']
        return self.eval_args, self.checkpoint_dir, self.save_guesses, self.save_matches

    def get_lower_probability_threshold(self, n_samples):
        n_samples = int(n_samples)
        if n_samples <= 10 ** 6:
            return 0.00000001
        elif n_samples <= 10 ** 7:
            return 0.000000001
        elif n_samples <= 5 * (10 ** 8):
            return 0.0000000001
        else:
            return 0.00000000001

    def save(self, fname):
        for filename in glob.glob(os.path.join(self.checkpoint_dir, "mid-"+self.checkpoint_name)):
            os.remove(filename)

        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, os.path.join(self.checkpoint_dir, fname))

    def load(self, fname):
        try:
            state_dicts = torch.load(os.path.join(self.checkpoint_dir, fname), map_location=self.device)
            self.model, self.optimizer, = self.build_model()
            self.model.to(self.device)
            self.model.load_state_dict(state_dicts['model'])
            self.optimizer.load_state_dict(state_dicts['optimizer'])
            return 1
        except Exception as e:
            print(f"Exception: {e}")
            return 0

    def build_model(self):
        lstm_hidden_size = self.params_train['lstm_hidden_size']
        dense_hidden_size = self.params_train['dense_hidden_size']
        context_len = self.max_length
        vocab_size = self.data.tokenizer.vocab_size

        model = LSTM(lstm_hidden_size=lstm_hidden_size,
                     dense_hidden_size=dense_hidden_size,
                     vocab_size=vocab_size,
                     context_len=context_len
                     )

        optimizer = torch.optim.Adam(model.parameters())
        return model, optimizer

    def train_step(self, x_train, y_train):
        self.model.train()
        self.optimizer.zero_grad()

        y_pred = self.model(x_train)

        train_loss = F.cross_entropy(y_pred, y_train)
        train_loss.backward()

        self.optimizer.step()

    def train(self):
        epochs = self.params_train["epochs"]
        batch_size = self.params_train["batch_size"]

        print("[I] - Launching training")
        start = time.time()

        mini_eval_args = {
            "n_samples": 10**6,
            "lower_probability_threshold": 0.0000001,
        }

        self.current_epoch = 0
        self.n_matches = 0

        self.model, self.optimizer = self.build_model()
        self.model.to(self.device)

        while self.current_epoch < epochs:
            self.current_epoch += 1

            print(f"Epoch: {self.current_epoch} / {epochs}")

            n_passwords = self.data.get_train_size()
            progress_bar = tqdm(range(n_passwords), desc="Epoch {}/{}".format(self.current_epoch, epochs))

            n_iter = 0
            for batch in self.data.get_batches(batch_size):
                x_train = np.array(batch[0])
                y_train = np.array(batch[1])

                x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
                y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

                self.train_step(x_train, y_train)
                progress_bar.update(batch_size)
                n_iter += 1


            if self.current_epoch % self.checkpoint_frequency == 0:
                matches, _, _ = self.evaluate(mini_eval_args,
                                              save_matches=False,
                                              save_guesses=False)
                if matches >= self.n_matches:
                    self.n_matches = matches
                    self.save("mid-"+self.checkpoint_name)

        end = time.time()
        self.time_delta = timedelta(seconds=end - start)

        self.finalize_checkpoint(self.checkpoint_name)

        print(f"[T] - Training completed after: {self.time_delta}")

    def evaluate(self, eval_args, save_matches, save_guesses, skip_gen=False):
        print("[I] - Generating passwords")
        output_file = os.path.join(self.guesses_dir, "total_guesses.gz")

        n_samples = eval_args["n_samples"]
        lower_probability_threshold = eval_args["lower_probability_threshold"]

        if not skip_gen:
            guesser = Guesser(model=self.model, params=self.params, data=self.data,
                              lower_probability_threshold=lower_probability_threshold, output_file=output_file, device=self.device)

            n_gen = guesser.complete_guessing()
        else:
            with gzip.open(output_file, 'rt') as f:
                n_gen = sum(1 for line in f)

        print(f"[I] - Generated {n_gen} passwords")

        min_heap_n_most_prob = []

        with gzip.open(output_file, "rt") as f_out:
            for line in f_out:
                line = line.split(" ")
                if len(line) != 2:
                    continue

                password, prob = line[0].replace("~", ""), float(line[1])

                if len(min_heap_n_most_prob) < n_samples:
                    heapq.heappush(min_heap_n_most_prob, (prob, password))
                else:
                    if prob > min_heap_n_most_prob[0][0]:
                        heapq.heappushpop(min_heap_n_most_prob, (prob, password))

        n_most_prob_psw = {password for prob, password in heapq.nlargest(n_samples, min_heap_n_most_prob)}
        current_match_n_most_prob_psw = n_most_prob_psw & self.data.test_passwords

        if save_guesses:
            file_name = os.path.join(self.guesses_dir, "guesses-most_prob_n_psw.gz")
            write_passwords_to_file(file_name, n_most_prob_psw)

        if save_matches:
            file_name = os.path.join(self.matches_dir, "matches-most_prob_n_psw.gz")
            write_passwords_to_file(file_name, current_match_n_most_prob_psw)

        matches = len(current_match_n_most_prob_psw)
        match_percentage = f'{(matches / self.data.get_test_size()) * 100:.2f}%'

        print(f'{matches} matches found ({match_percentage} of test set). '
              f'total generated samples: {n_gen}. Samples generated by sampling the n most probable passwords.')

        del n_most_prob_psw
        gc.collect()
        os.remove(output_file)

        test_size = self.data.get_test_size()

        match_percentage = f'{(matches / test_size) * 100:.2f}%'
        return matches, match_percentage, test_size