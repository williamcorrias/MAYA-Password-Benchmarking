from datetime import timedelta
import os
import torch
import time
import torch.nn.functional as F
from tqdm import tqdm
import heapq
import gc
import gzip

from script.test.model import Model

from models.FLA.architecture import LSTM
from models.FLA.guesser import Guesser
from models.FLA.fla_utils.dataloader import *


def get_lower_probability_threshold(n_samples):
    n_samples = int(n_samples)
    if n_samples <= 10 ** 6:
        return 0.00000001
    elif n_samples <= 10 ** 7:
        return 0.000000001
    elif n_samples <= 5 * (10 ** 8):
        return 0.0000000001
    else:
        return 0.00000000001

class FLA(Model):
    def __init__(self, settings):
        self.model = None
        self.optimizer = None

        super().__init__(settings)

    def prepare_data(self, train_passwords, test_passwords, max_length):
        return DataLoader(train_passwords, test_passwords, max_length, self.params)

    def load(self, file_to_load):
        try:
            self.init_model()
            state_dicts = torch.load(file_to_load, map_location=self.device)
            self.model.load_state_dict(state_dicts['model'])
            self.optimizer.load_state_dict(state_dicts['optimizer'])
            return 1
        except Exception as e:
            print(f"Exception: {e}")
            return 0

    def init_model(self):
        self.params['eval']['evaluation_batch_size'] = self.n_samples + 1
        lstm_hidden_size = self.params["train"]['lstm_hidden_size']
        dense_hidden_size = self.params["train"]['dense_hidden_size']
        context_len = self.data.max_length
        vocab_size = self.data.tokenizer.vocab_size

        self.model = LSTM(lstm_hidden_size=lstm_hidden_size,
                     dense_hidden_size=dense_hidden_size,
                     vocab_size=vocab_size,
                     context_len=context_len
                     )

        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train_step(self, x_train, y_train):
        self.model.train()
        self.optimizer.zero_grad()

        y_pred = self.model(x_train)

        train_loss = F.cross_entropy(y_pred, y_train)
        train_loss.backward()

        self.optimizer.step()

    def train(self):
        print("[I] - Launching training")

        epochs = self.params["train"]["epochs"]
        batch_size = self.params["train"]["batch_size"]

        start = time.time()

        current_epoch = 0
        n_matches = 0
        n_passwords = self.data.get_train_size()

        checkpoint_frequency = self.params['eval']['checkpoint_frequency']

        self.init_model()

        while current_epoch < epochs:
            print(f"Epoch: {current_epoch + 1} / {epochs}")

            progress_bar = tqdm(range(n_passwords), desc="Epoch {}/{}".format(current_epoch, epochs))

            n_iter = 0
            for batch in self.data.get_batches(batch_size):
                x_train = np.array(batch[0])
                y_train = np.array(batch[1])

                x_train = torch.tensor(x_train, dtype=torch.float32).to(self.device)
                y_train = torch.tensor(y_train, dtype=torch.float32).to(self.device)

                self.train_step(x_train, y_train)
                progress_bar.update(batch_size)
                n_iter += 1

            if current_epoch % checkpoint_frequency == 0:
                matches, _, _ = self.evaluate(n_samples=10 ** 6, validation_mode=True)
                if matches >= n_matches:
                    n_matches = matches
                    obj = {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                    }
                    self.save(obj)

            current_epoch += 1

        end = time.time()
        time_delta = timedelta(seconds=end - start)
        print(f"[T] - Training completed after: {time_delta}")

    def eval_init(self, n_samples, evaluation_batch_size):
        self.model.eval()
        eval_dict = {
            'n_samples': n_samples,
            'output_file': os.path.join(self.path_to_guesses_dir, "total_guesses.gz"),
        }
        return eval_dict

    def sample(self, evaluation_batch_size, eval_dict):
        lower_probability_threshold = get_lower_probability_threshold(eval_dict['n_samples'])
        guesser = Guesser(model=self.model, params=self.params, data=self.data,
                          lower_probability_threshold=lower_probability_threshold, output_file=eval_dict['output_file'],
                          device=self.device)
        n_gen = guesser.complete_guessing()

        print(f"[I] - Generated {n_gen} passwords")

        min_heap_n_most_prob = []

        with gzip.open(eval_dict['output_file'], "rt") as f_out:
            for line in f_out:
                line = line.split(" ")
                if len(line) != 2:
                    continue

                password, prob = line[0].replace("~", ""), float(line[1])

                if len(min_heap_n_most_prob) < eval_dict['n_samples']:
                    heapq.heappush(min_heap_n_most_prob, (prob, password))
                else:
                    if prob > min_heap_n_most_prob[0][0]:
                        heapq.heappushpop(min_heap_n_most_prob, (prob, password))

        n_most_prob_psw = {password for prob, password in heapq.nlargest(eval_dict['n_samples'], min_heap_n_most_prob)}

        return n_most_prob_psw

    def guessing_strategy(self, evaluation_batch_size, eval_dict):
        pass

    def post_sampling(self, eval_dict):
        gc.collect()
        os.remove(eval_dict['output_file'])
        pass