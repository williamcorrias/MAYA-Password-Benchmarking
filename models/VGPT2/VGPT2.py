import math
import gzip
from tqdm import tqdm
import torch
import glob
import time
from datetime import timedelta
import os

from models.VGPT2.src.models.autoencoders import VAE

from script.test.model import Model
from models.VGPT2.src.data.dataloader import TokenizedTextDataLoader

class VGPT2(Model):

    def __init__(self, settings):
        super().__init__(settings)

    def prepare_data(self):
        params_dataloader = self.params['dataloader']
        TOKENIZER_MAX_LEN = int(self.max_length) + 2
        self.data = TokenizedTextDataLoader(self.train_path, self.test_path, TOKENIZER_MAX_LEN, params_dataloader)

    def prepare_model(self):
        self.params_model = self.params["model"]
        self.params_train = self.params["train"]

        self.checkpoint_frequency = self.params['eval']['checkpoint_frequency']
        self.checkpoint_dir = os.path.join("checkpoints", "VGPT2", self.train_hash)
        if not os.path.exists(os.path.join(os.getcwd(), self.checkpoint_dir)):
            os.makedirs(os.path.join(os.getcwd(), self.checkpoint_dir), exist_ok=True)

        self.evaluation_batch_size = self.params['eval']['evaluation_batch_size']
        self.save_to_file_batch_size = self.params['eval']['save_to_file_batch_size']

        self.eval_args = {
            "n_samples": self.n_samples,
            "evaluation_batch_size": self.evaluation_batch_size,
        }

        self.save_guesses = self.params['eval']['save_guesses']
        self.save_matches = self.params['eval']['save_matches']
        return self.eval_args, self.checkpoint_dir, self.save_guesses, self.save_matches

    def save(self, fname):
        for filename in glob.glob(os.path.join(self.checkpoint_dir, "mid-"+self.checkpoint_name)):
            os.remove(filename)

        torch.save({
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, os.path.join(self.checkpoint_dir, fname))

    def load(self, fname):
        try:
            state_dicts = torch.load(os.path.join(self.checkpoint_dir, fname), map_location=self.device)

            self.model, self.optimizer, self.scheduler = self.build_model()
            self.model.to(self.device)

            self.model.load_state_dict(state_dicts['model'])
            self.optimizer.load_state_dict(state_dicts['optimizer'])
            self.scheduler.load_state_dict(state_dicts['scheduler'])

            self.model_loaded = True
            return 1
        except Exception:
            return 0

    def build_model(self):
        vae_args = dict()
        vae_args["vocab_dim"] = self.data.vocab_size
        vae_args["latent_dim"] = self.params_model["latent_dim"]
        vae_args["embedding_dim"] = self.params_model['embedding_dim']
        vae_args["encoder"] = self.params_model['encoder']
        vae_args["decoder"] = self.params_model['decoder']
        vae_args["parameter_schedulers"] = self.params_model['parameter_schedulers']
        vae_args["max_sequence_length"] = self.data.max_sequence_length
        vae_args["sos_index"] = self.data.sos_index
        vae_args["eos_index"] = self.data.eos_index
        vae_args["pad_index"] = self.data.pad_index
        vae_args["unk_index"] = self.data.unk_index

        model = VAE(vae_args, self.device).to(self.device)

        optimizer, scheduler = model.configure_optimizers()

        return model, optimizer, scheduler

    def train(self):
        if self.model_loaded == False:
            self.current_epoch = 0
            self.n_matches = 0
            self.model, self.optimizer, self.scheduler = self.build_model()
            self.model.to(self.device)
            print("[I] - Starting the training")
        else:
            print("[I] - Continuing the training...")

        torch.set_grad_enabled(True)

        epochs = self.params_train["epochs"]
        batch_size = self.params_train["batch_size"]

        print("[I] - Launching training")
        start = time.time()

        mini_eval_args = {
            "n_samples": 10**6,
            "evaluation_batch_size": self.evaluation_batch_size,
        }

        while self.current_epoch < epochs:
            self.current_epoch += 1

            print(f"Epoch: {self.current_epoch} / {epochs}")

            self.model.train()

            n_passwords = self.data.get_train_size()
            progress_bar = tqdm(range(n_passwords), desc="Epoch {}/{}".format(self.current_epoch, epochs))

            for batch in self.data.get_batches(batch_size):
                self.optimizer.zero_grad()

                step_results = self.model.training_step(batch)

                loss = step_results["loss"].to(self.device)
                loss.backward()

                self.optimizer.step()

                self.model.clip_gradients(clip_value=0.1)
                self.model.global_step += 1

                progress_bar.update(len(batch[0]))

            if self.current_epoch % self.checkpoint_frequency == 0:
                matches, _ , _ = self.evaluate(mini_eval_args, save_matches=False, save_guesses=False)
                if matches >= self.n_matches:
                    self.n_matches = matches

                    checkpoint_time = time.time()
                    delta = timedelta(seconds=checkpoint_time - start)
                    self.checkpoint_time_delta = self.time_delta + delta

                    self.save("mid-"+self.checkpoint_name)

            self.scheduler.step()

        end = time.time()
        delta = timedelta(seconds=end - start)
        self.time_delta = self.time_delta + delta

        self.finalize_checkpoint(self.checkpoint_name)

        print(f"[T] - Training completed after: {self.time_delta}")

    def sample(self, batch_size):
        with torch.no_grad():
            batch_indices, batch_lengths = self.model.generate(batch_size)
            generated_data = [(self.data.tokenizer.decode(indices)[:length]) for indices, length in
                              zip(batch_indices, batch_lengths)]
        return generated_data

    def write_guesses(self, generated_data):
        with gzip.open(os.path.join(self.guesses_dir, "Epoch" + str(self.current_epoch) + ".gz"), 'at') as file:
            for row in generated_data:
                file.write(''.join(map(str, row)) + '\n')

    def evaluate(self, eval_args, save_matches, save_guesses):
        print("[I] - Generating passwords")
        self.model.eval()

        n_samples = eval_args["n_samples"]
        evaluation_batch_size = eval_args["evaluation_batch_size"]

        evaluation_batch_size = int(evaluation_batch_size)
        if n_samples < evaluation_batch_size:
            batches, evaluation_batch_size = 1, int(n_samples)
        else:
            batches = math.floor(n_samples / evaluation_batch_size)

        matches = set()

        progress_bar = tqdm(range(batches))
        progress_bar.set_description(desc='Generating sample batch')

        generated_passwords = []

        for batch in range(batches):
            generated_psw = self.sample(evaluation_batch_size)

            generated_passwords.extend(generated_psw)
            if save_matches and len(generated_passwords) >= self.save_to_file_batch_size:
                self.write_guesses(generated_passwords)
                generated_passwords = []

            current_match = set(generated_psw) & self.data.test_passwords
            matches.update(current_match)

            progress_bar.set_postfix({'Matches found': {len(matches)},
                                      'Test set %': ({len(matches) / self.data.get_test_size() * 100.0})})

            progress_bar.update(1)

        if save_matches and len(generated_passwords) > 0:
            self.write_guesses(generated_passwords)
            generated_passwords = []

        if save_matches == True:
            with gzip.open(os.path.join(self.matches_dir, "Epoch" + str(self.current_epoch) + ".gz"), 'at') as file:
                for match in matches:
                    string_list = [str(element) for element in match]
                    result_string = ''.join(string_list)
                    file.write(str(result_string) + '\n')

        matches = len(matches)
        test_size = self.data.get_test_size()
        match_percentage = f'{(matches / test_size) * 100:.2f}%'
        print(f'{matches} matches found ({match_percentage} of test set).')
        return matches, match_percentage, test_size