from tqdm import tqdm
import torch
import time
from datetime import timedelta
import os

import sys
sys.path.append(os.path.join("models", "VGPT2"))

from models.VGPT2.src.models.autoencoders import VAE

from script.test.model import Model
from models.VGPT2.src.data.dataloader import TokenizedTextDataLoader

class VGPT2(Model):

    def __init__(self, settings):
        self.model = None
        self.optimizer = None
        self.scheduler = None
        super().__init__(settings)

    def prepare_data(self, train_passwords, test_passwords, max_length):
        params_dataloader = self.params['dataloader']
        TOKENIZER_MAX_LEN = max_length + 2
        self.data = TokenizedTextDataLoader(train_passwords, set(test_passwords), TOKENIZER_MAX_LEN, params_dataloader)
        return self.data

    def load(self, file_to_load):
        try:
            self.init_model()
            state_dicts = torch.load(file_to_load, map_location=self.device)
            self.model.load_state_dict(state_dicts['model'])
            self.optimizer.load_state_dict(state_dicts['optimizer'])
            self.scheduler.load_state_dict(state_dicts['scheduler'])
            return 1
        except Exception as e:
            print(f"Exception: {e}")
            return 0

    def init_model(self):
        vae_args = dict()
        vae_args["vocab_dim"] = self.data.vocab_size
        vae_args["latent_dim"] = self.params["model"]["latent_dim"]
        vae_args["embedding_dim"] = self.params["model"]['embedding_dim']
        vae_args["encoder"] = self.params["model"]['encoder']
        vae_args["decoder"] = self.params["model"]['decoder']
        vae_args["parameter_schedulers"] = self.params["model"]['parameter_schedulers']
        vae_args["max_sequence_length"] = self.data.max_sequence_length
        vae_args["sos_index"] = self.data.sos_index
        vae_args["eos_index"] = self.data.eos_index
        vae_args["pad_index"] = self.data.pad_index
        vae_args["unk_index"] = self.data.unk_index

        self.model = VAE(vae_args, self.device).to(self.device)

        self.optimizer, self.scheduler = self.model.configure_optimizers()

    def train(self):
        print("[I] - Launching training")
        torch.set_grad_enabled(True)

        current_epoch = 0
        epochs = self.params["train"]["epochs"]
        batch_size = self.params["train"]["batch_size"]
        n_passwords = self.data.get_train_size()
        n_matches = 0

        checkpoint_frequency = self.params['eval']['checkpoint_frequency']

        start = time.time()

        self.init_model()

        while current_epoch < epochs:
            print(f"Epoch: {current_epoch + 1} / {epochs}")
            self.model.train()

            progress_bar = tqdm(range(n_passwords), desc="Epoch {}/{}".format(current_epoch, epochs))

            for batch in self.data.get_batches(batch_size):
                self.optimizer.zero_grad()

                step_results = self.model.training_step(batch)

                loss = step_results["loss"].to(self.device)
                loss.backward()

                self.optimizer.step()

                self.model.clip_gradients(clip_value=0.1)
                self.model.global_step += 1

                progress_bar.update(len(batch[0]))

            if current_epoch % checkpoint_frequency == 0:
                matches, _, _ = self.evaluate(n_samples=10 ** 6, validation_mode=True)
                if matches >= n_matches:
                    n_matches = matches
                    obj = {
                        'model': self.model.state_dict(),
                        'optimizer': self.optimizer.state_dict(),
                        'scheduler': self.scheduler.state_dict(),
                    }
                    self.save(obj)

            self.scheduler.step()
            current_epoch += 1

        end = time.time()
        time_delta = timedelta(seconds=end - start)
        print(f"[T] - Training completed after: {time_delta}")

    def eval_init(self, n_samples, evaluation_batch_size):
        self.model.eval()
        eval_dict = {
            'n_samples': n_samples,
        }
        return eval_dict

    def sample(self, evaluation_batch_size, eval_dict):
        with torch.no_grad():
            generated_passwords, _ = self.model.generate(evaluation_batch_size)
            generated_passwords = {self.data.tokenizer.decode(psw) for psw in generated_passwords}
        return generated_passwords

    def guessing_strategy(self, evaluation_batch_size, eval_dict):
        pass

    def post_sampling(self, eval_dict):
        pass