import math
import torch.nn.functional as F
import time

from tqdm import tqdm
from datetime import timedelta

from script.test.model import Model
from script.dataset.dataset import Dataset
from models.PLRGAN.architecture import Generator, Discriminator
from models.PLRGAN.DPG import *

class PLRGAN(Model):
    def __init__(self, settings):
        self.Generator = None
        self.generator_opt = None
        self.Discriminator = None
        self.discriminator_opt = None

        super().__init__(settings)

    def prepare_data(self, path_to_train_dataset, path_to_test_dataset, max_length):
        return Dataset(path_to_train_dataset, path_to_test_dataset, max_length, self.test_hash)

    def load(self, file_to_load):
        try:
            self.init_model()
            state_dicts = torch.load(file_to_load, map_location=self.device)
            self.Generator.load_state_dict(state_dicts['Generator'])
            self.Discriminator.load_state_dict(state_dicts['Discriminator'])
            self.generator_opt.load_state_dict(state_dicts['generator_opt'])
            self.discriminator_opt.load_state_dict(state_dicts['discriminator_opt'])
            return 1
        except Exception as e:
            print(f"Exception: {e}")
            return 0

    def init_model(self):
        optimizer = torch.optim.Adam
        x_size = self.data.max_length
        z_size = self.params['train']['z_size']
        dict_size = self.data.charmap_size

        lr = self.params['train']['learning_rate']
        beta1 = self.params['train']['beta1']
        beta2 = self.params['train']['beta2']

        self.Generator = Generator(x_size, dict_size, layer_dim=z_size).to(self.device)
        self.Discriminator = Discriminator(x_size, dict_size, layer_dim=z_size).to(self.device)
        g_vars = [p for p in self.Generator.parameters() if p.requires_grad]
        d_vars = [p for p in self.Discriminator.parameters() if p.requires_grad]

        self.generator_opt = optimizer(g_vars, lr=lr, betas=(beta1, beta2))
        self.discriminator_opt = optimizer(d_vars, lr=lr, betas=(beta1, beta2))

    def train_discriminator(self, real_data):
        self.Discriminator.train()
        LAMBDA = self.params['train']['lambda']
        batch_size = self.params['train']['batch_size']

        self.discriminator_opt.zero_grad()
        real_data = self.transform_truedata(real_data.to(torch.int64), self.data.charmap_size)
        disc_real = self.Discriminator(real_data)
        noise = self.generate_random_noise(batch_size).to(self.device)
        fake_data = self.Generator(noise)
        disc_fake = self.Discriminator(fake_data.detach())
        disc_cost, gradient_penalty = self.compute_disc_wgangp_loss(fake_data, real_data, disc_real, disc_fake, LAMBDA)
        disc_cost.backward()
        self.discriminator_opt.step()
        return disc_cost

    def train_generator(self):
        self.Generator.train()
        batch_size = self.params['train']['batch_size']

        self.generator_opt.zero_grad()
        noise = self.generate_random_noise(batch_size).to(self.device)
        fake_data = self.Generator(noise)
        disc_fake = self.Discriminator(fake_data)
        gen_cost = self.compute_gen_wgangp_loss(disc_fake)
        gen_cost.backward()
        self.generator_opt.step()
        return gen_cost

    def train(self):
        print("[I] - Launching training")

        train_size = self.data.get_train_size()
        batch_size = self.params['train']['batch_size']
        batch_4_epochs = math.ceil(train_size / batch_size)
        num_gen_training_steps = self.params['train']['num_gen_training_steps']
        disc_iters4iteration = self.params['train']['D_iters']
        epochs = math.ceil(num_gen_training_steps * disc_iters4iteration / batch_4_epochs)
        progress_bar = tqdm(range(num_gen_training_steps))

        start = time.time()

        current_epoch = 0
        gen_iteration_counter = 0
        disc_iteration_counter = 0
        n_matches = 0

        checkpoint_frequency = self.params['eval']['checkpoint_frequency']

        self.init_model()

        while current_epoch < epochs:

            print(f"Epoch: {current_epoch + 1} / {epochs}")

            for real_data in self.data.get_batches(batch_size=batch_size):
                real_data = torch.tensor(real_data).to(self.device)

                _ = self.train_discriminator(real_data)
                disc_iteration_counter += 1

                if disc_iteration_counter % disc_iters4iteration == 0:
                    _ = self.train_generator()
                    gen_iteration_counter += 1

                    if gen_iteration_counter % checkpoint_frequency == 0:
                        matches, _, _ = self.evaluate(n_samples=10**6, validation_mode=True)
                        if matches >= n_matches:
                            n_matches = matches

                            obj = {
                                'generator_opt': self.generator_opt.state_dict(),
                                'discriminator_opt': self.discriminator_opt.state_dict(),
                                'Generator': self.Generator.state_dict(),
                                'Discriminator': self.Discriminator.state_dict(),
                            }
                            self.save(obj)

                    progress_bar.update(1)

                    if gen_iteration_counter >= num_gen_training_steps:
                        break

            current_epoch += 1

        end = time.time()
        time_delta = timedelta(seconds=end - start)
        print(f"[T] - Training completed after: {time_delta}")

    def transform_truedata(self, x, dict_size):
        x = F.one_hot(x, dict_size).to(self.device)
        gamma = self.params['train']['gamma']
        batch_size = self.params['train']['batch_size']
        x_size = self.data.max_length
        if gamma:
            x = x + (torch.rand((batch_size, x_size, dict_size)).to(self.device) * gamma)
            x = x / torch.sum(x, dim=2, keepdim=True)
        return x

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

    def eval_init(self, n_samples, evaluation_batch_size):
        self.Generator.eval()

        eval_dict = {
            'DYNAMIC': bool(self.settings.get("ds", False)),
            'state': DPG(
                self.params['train']['z_size'],
                self.params['train']['z_prior'],
                self.params['eval']['sigma'],
                self.params['eval']['alpha'],
                n_samples,
                evaluation_batch_size,
                self.data.test_passwords.copy(),
                device=self.device)
        }

        return eval_dict

    def sample(self, evaluation_batch_size, eval_dict):
        with torch.no_grad():
            if eval_dict['DYNAMIC']:
                z = eval_dict['state'].guess().to(self.device)
            else:
                z = self.generate_random_noise(evaluation_batch_size).to(self.device)
            generated_passwords = self.Generator(z)
            generated_passwords = torch.argmax(generated_passwords, 2)
            generated_passwords = generated_passwords.type(torch.uint8)
            generated_passwords = set([tuple(row.tolist()) for row in generated_passwords])
            eval_dict['z'] = z
            eval_dict['generated_passwords'] = generated_passwords

        return generated_passwords

    def guessing_strategy(self, evaluation_batch_size, eval_dict):
        if eval_dict['DYNAMIC']:
            for z, x in zip(eval_dict['z'], eval_dict['generated_passwords']):
                eval_dict['state'](z, x)

    def post_sampling(self, eval_dict):
        pass