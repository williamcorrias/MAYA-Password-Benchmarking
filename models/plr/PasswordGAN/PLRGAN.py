import copy
import glob
import math
import os
import gzip
import torch.nn.functional as F
import time

from tqdm import tqdm
from datetime import timedelta

from script.test.model import Model
from script.dataset.dataset import Dataset
from models.plr.PasswordGAN.architecture import Generator, Discriminator
from models.plr.PasswordGAN.DPG import *

class PLRGAN(Model):

    def __init__(self, settings):
        super().__init__(settings)

    def prepare_data(self):
        self.data = Dataset(self.train_path, self.test_path, self.max_length, self.test_hash, skip_unk=True)

    def prepare_model(self):
        self.dict_size = self.data.charmap_size
        self.LAMBDA = self.params['train']['lambda']
        self.learning_rate = self.params['train']['learning_rate']
        self.evaluation_batch_size = self.params['eval']['evaluation_batch_size']
        self.save_to_file_batch_size = self.params['eval']['save_to_file_batch_size']
        self.eval_args = {
            "DYNAMIC": bool(self.settings["ds"]),
            "alpha": self.params['eval']['alpha'],
            "sigma": self.params['eval']['sigma'],
            "n_samples": self.n_samples,
            "evaluation_batch_size": self.evaluation_batch_size,
        }
        self.DYNAMIC = self.eval_args["DYNAMIC"]

        self.optimizer = torch.optim.Adam

        self.checkpoint_frequency = self.params['eval']['checkpoint_frequency']
        self.checkpoint_dir = os.path.join("checkpoints", "plr", "gan", self.train_hash)
        if not os.path.exists(os.path.join(os.getcwd(), self.checkpoint_dir)):
            os.makedirs(os.path.join(os.getcwd(), self.checkpoint_dir), exist_ok=True)

        self.save_guesses = self.params['eval']['save_guesses']
        self.save_matches = self.params['eval']['save_matches']

        return self.eval_args, self.checkpoint_dir, self.save_guesses, self.save_matches

    def save(self, fname):
        # remove old checkpoints
        for filename in glob.glob(os.path.join(self.checkpoint_dir, "mid-"+self.checkpoint_name)):
            os.remove(filename)

        # save last checkpoint
        torch.save({
            'generator_opt': self.generator_opt.state_dict(),
            'discriminator_opt': self.discriminator_opt.state_dict(),
            'Generator': self.Generator.state_dict(),
            'Discriminator': self.Discriminator.state_dict(),
        }, os.path.join(self.checkpoint_dir, fname))

    def load(self, fname):
        try:
            self.build_model("train")
            state_dicts = torch.load(os.path.join(self.checkpoint_dir, fname), map_location=self.device, weights_only=False)
            self.Generator.load_state_dict(state_dicts['Generator'])
            self.Discriminator.load_state_dict(state_dicts['Discriminator'])
            self.generator_opt.load_state_dict(state_dicts['generator_opt'])
            self.discriminator_opt.load_state_dict(state_dicts['discriminator_opt'])
            return 1
        except Exception as e:
            print(f"Exception: {e}")
            return 0

    def train_discriminator(self, real_data):
        self.Discriminator.train()

        self.discriminator_opt.zero_grad()
        real_data = self.transform_truedata(real_data.to(torch.int64), self.dict_size)
        disc_real = self.Discriminator(real_data)
        noise = self.generate_random_noise(self.batch_size).to(self.device)
        fake_data = self.Generator(noise)
        disc_fake = self.Discriminator(fake_data.detach())
        disc_cost, gradient_penalty = self.compute_disc_wgangp_loss(fake_data, real_data, disc_real, disc_fake, self.LAMBDA)
        disc_cost.backward()
        self.discriminator_opt.step()
        return disc_cost

    def train_generator(self):
        self.Generator.train()

        self.generator_opt.zero_grad()
        noise = self.generate_random_noise(self.batch_size).to(self.device)
        fake_data = self.Generator(noise)
        disc_fake = self.Discriminator(fake_data)
        gen_cost = self.compute_gen_wgangp_loss(disc_fake)
        gen_cost.backward()
        self.generator_opt.step()
        return gen_cost

    def train(self):
        num_gen_training_steps = self.params['train']['num_gen_training_steps']
        disc_iters4iteration = self.params['train']['D_iters']
        train_size = self.data.get_train_size()
        self.batch_size = self.params['train']['batch_size']

        batch_4_epochs = math.ceil(train_size / self.batch_size)
        epochs = math.ceil(num_gen_training_steps * disc_iters4iteration / batch_4_epochs)
        progress_bar = tqdm(range(num_gen_training_steps))

        print("[I] - Launching training")

        start = time.time()

        mini_eval_args = copy.deepcopy(self.eval_args)
        mini_eval_args.update({
            "n_samples": 10**6,
            "evaluation_batch_size": self.evaluation_batch_size,
        })

        self.gen_iteration_counter = 0
        self.disc_iteration_counter = 0
        self.build_model("train")
        self.n_matches = 0

        while self.current_epoch < epochs:

            print(f"Epoch: {self.current_epoch + 1} / {epochs}")

            for real_data in self.data.get_batches(batch_size=self.batch_size):
                real_data = torch.tensor(real_data).to(self.device)

                disc_cost = self.train_discriminator(real_data)
                self.disc_iteration_counter += 1

                if self.disc_iteration_counter % disc_iters4iteration == 0:
                    gen_cost = self.train_generator()

                    self.gen_iteration_counter += 1

                    if self.gen_iteration_counter % self.checkpoint_frequency == 0:
                        matches, _, _ = self.evaluate(mini_eval_args, save_matches=False, save_guesses=False)
                        if matches >= self.n_matches:
                            self.n_matches = matches
                            self.save("mid-"+self.checkpoint_name)

                    progress_bar.update(1)

                    if self.gen_iteration_counter >= num_gen_training_steps:
                        break

            self.current_epoch += 1

        end = time.time()
        self.time_delta = timedelta(seconds=end - start)

        self.finalize_checkpoint(self.checkpoint_name)

        print("[T] - Training completed after: " + str(self.time_delta))

    def build_model(self, mode):
        is_training = mode == "train"

        x_size = self.data.max_length
        z_size = self.params['train']['z_size']

        self.dict_size = self.data.charmap_size

        self.Generator = Generator(x_size, self.dict_size, layer_dim=z_size, is_training=is_training).to(self.device)

        self.G_vars = [p for p in self.Generator.parameters() if p.requires_grad]

        if is_training:
            self.Discriminator = Discriminator(x_size, self.dict_size, layer_dim=z_size, is_training=is_training).to(self.device)
            self.D_vars = [p for p in self.Discriminator.parameters() if p.requires_grad]
            self.create_optimizers()

        else:
            self.Discriminator = None
            self.D_vars = None


    def transform_truedata(self, x, dict_size):
        x = F.one_hot(x, dict_size).to(self.device)
        # label smoothing
        gamma = self.params['train']['gamma']
        batch_size = self.params['train']['batch_size']
        x_size = self.data.max_length
        if gamma:
            x = x + (torch.rand((batch_size, x_size, dict_size)).to(self.device) * gamma)
            # normalize
            x = x / torch.sum(x, dim=2, keepdim=True)
        return x

    def create_optimizers(self):
        lr = self.learning_rate
        beta1 = self.params['train']['beta1']
        beta2 = self.params['train']['beta2']

        self.generator_opt = self.optimizer(self.G_vars, lr=lr, betas=(beta1, beta2))
        self.discriminator_opt = self.optimizer(self.D_vars, lr=lr, betas=(beta1, beta2))

    def compute_disc_wgangp_loss(self, fake_data, real_data, disc_real, disc_fake, LAMBDA):
        batch_size = self.params['train']['batch_size']

        disc_cost = torch.mean(disc_fake) - torch.mean(disc_real)

        alpha = torch.rand(batch_size, 1, 1).to(self.device)

        differences = fake_data - real_data
        interpolates = (real_data + (alpha*differences))

        D = self.Discriminator(interpolates)

        gradients = torch.autograd.grad(outputs=D, inputs=interpolates, grad_outputs=torch.ones_like(D), retain_graph=True, create_graph=True)[0]
        slopes = torch.sqrt(torch.sum(torch.square(gradients), dim=2))
        gradient_penality = LAMBDA * torch.mean((slopes - 1.) ** 2)
        disc_cost += gradient_penality
        return disc_cost, gradient_penality

    def compute_gen_wgangp_loss(self, disc_fake):
        gen_cost = -torch.mean(disc_fake)
        return gen_cost

    def generate_random_noise(self, batch_size):
        z_prior = self.params['train']['z_prior']
        z_size = self.params['train']['z_size']
        z = torch.normal(mean=0, std=z_prior, dtype=torch.float32, size=(batch_size, z_size))
        return z

    def sample(self, batch_size):
        with torch.no_grad():
            if self.DYNAMIC:
                z = self.state.guess().to(self.device)
            else:
                z = self.generate_random_noise(batch_size).to(self.device)

            generated_data = self.Generator(z)
            generated_data = torch.argmax(generated_data, 2)
            generated_data = generated_data.type(torch.uint8)
            generated_data = set([tuple(row.tolist()) for row in generated_data])

        return z, generated_data

    def write_guesses(self, encoded_passwords):
        with gzip.open(os.path.join(self.guesses_dir, "guesses.gz"), 'at') as file:
            for psw in encoded_passwords:
                result_string = ''.join(map(str, self.data.decode_password(psw)))
                cleaned_string = result_string.replace('`', '')
                file.write(cleaned_string + '\n')

    def evaluate(self, eval_args, save_matches, save_guesses):
        print("[I] - Generating passwords")
        self.Generator.eval()

        n_samples = eval_args['n_samples']
        evaluation_batch_size = eval_args['evaluation_batch_size']

        evaluation_batch_size = int(evaluation_batch_size)
        if n_samples < evaluation_batch_size:
            batches, evaluation_batch_size = 1, int(n_samples)
        else:
            batches = math.floor(n_samples / evaluation_batch_size)

        matches = set()

        progress_bar = tqdm(range(batches))
        progress_bar.set_description(desc='Generating sample batch')

        z_size = self.params['train']['z_size']
        z_prior = self.params['train']['z_prior']

        hot_start = eval_args["alpha"]
        dpg_z_prior = eval_args["sigma"]

        self.state = DPG(z_size, z_prior, dpg_z_prior, hot_start, self.n_samples, evaluation_batch_size, self.data.test_passwords.copy(), device=self.device)

        encoded_guesses = []

        for batch in range(batches):
            z, generated_psw = self.sample(evaluation_batch_size)

            encoded_guesses.extend(generated_psw)
            if save_matches and len(encoded_guesses) >= self.save_to_file_batch_size:
                self.write_guesses(encoded_guesses)
                encoded_guesses = []

            if self.DYNAMIC:
                for z, x in zip(z, generated_psw):
                    self.state(z, x)

            current_match = generated_psw & self.data.test_passwords
            matches.update(current_match)

            progress_bar.set_postfix({'Matches found': {len(matches)},
                                      'Test set %': ({len(matches) / self.data.get_test_size() * 100.0})})

            progress_bar.update(1)

        if save_matches and len(encoded_guesses) > 0:
            self.write_guesses(encoded_guesses)
            encoded_guesses = []

        if save_matches == True:
            with gzip.open(os.path.join(self.matches_dir, "matches.gz"), 'at') as file:
                for match in matches:
                    decoded_match = self.data.decode_password(match)
                    result_string = ''.join(map(str, decoded_match))
                    cleaned_string = result_string.rstrip('`')
                    file.write(cleaned_string + '\n')

        matches = len(matches)
        test_size = self.data.get_test_size()

        match_percentage = f'{(matches / test_size) * 100:.2f}%'
        print(f'{matches} matches found ({match_percentage} of test set)')
        return matches, match_percentage, test_size