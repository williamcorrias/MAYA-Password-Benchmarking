import copy

import torch
import os
import torch.optim as optim
from tqdm import tqdm
import numpy as np
import torch.nn.functional as F

from script.metrics.psw_classifier.architecture import CNNClassifier
from script.metrics.utils.common_op import get_batches
from script.utils.gpu import select_gpu


class PasswordClassifier:

    def __init__(self, train_dataset, test_dataset, tokenizer, name, params):
        if torch.cuda.is_available():
            free_gpu = select_gpu()
            self.device = torch.device(f'cuda:{free_gpu}')
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.tokenizer = tokenizer

        self.name = name

        self.params = params

        self.model = self.build_model()
        self.ema_model = copy.deepcopy(self.model)

        self.model.to(self.device)
        self.ema_model.to(self.device)

        self.optimizer = self.build_optimizer()

        self.checkpoint_dir = os.path.join(os.getcwd(), "script", "metrics", "psw_classifier", "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def save(self, fname):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'ema_state_dict': self.ema_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'tokenizer': self.tokenizer,

        }, os.path.join(self.checkpoint_dir, fname))

    def load(self, fname):
        state_dicts = torch.load(os.path.join(self.checkpoint_dir, fname), map_location=self.device)
        self.model.load_state_dict(state_dicts['model_state_dict'])
        self.ema_model.load_state_dict(state_dicts['ema_state_dict'])
        self.optimizer.load_state_dict(state_dicts['optimizer_state_dict'])
        self.tokenizer = state_dicts['tokenizer']

    def build_model(self):
        input_dim = self.tokenizer.vocab_size
        conv1_1dim = self.params["train"]["conv1_1dim"]
        conv1_2dim = self.params["train"]["conv1_2dim"]
        conv1_3dim = self.params["train"]["conv1_3dim"]
        output_size = self.params["train"]["output_size"]
        context_len = self.params["data"]["max_length"]
        model = CNNClassifier(input_dim, conv1_1dim, conv1_2dim, conv1_3dim, context_len, output_size)
        return model

    def build_optimizer(self):
        return optim.Adam(self.model.parameters(), lr=self.params["train"]["learning_rate"])

    def update_ema(self, model, ema_model, ema_decay):
        with torch.no_grad():
            for ema_param, param in zip(ema_model.parameters(), model.parameters()):
                ema_param.data.copy_(ema_param.data * ema_decay + (1 - ema_decay) * param.data)

    def transform_data(self, data, vocab_size):
        return F.one_hot(data, vocab_size).to(self.device)

    def compute_wgan_loss(self, fake_data, real_data, fake_output, real_output, LAMBDA=10, requires_grad=True):
        batch_size = real_data.size(0)

        loss = torch.mean(fake_output) - torch.mean(real_output)

        alpha = torch.rand(batch_size, 1, 1).to(self.device)

        differences = fake_data - real_data
        interpolates = (real_data + (alpha * differences)).requires_grad_(requires_grad)

        output = self.model(interpolates)

        gradients = torch.autograd.grad(outputs=output, inputs=interpolates, grad_outputs=torch.ones_like(output),
                                        retain_graph=requires_grad, create_graph=requires_grad)[0]

        slopes = torch.sqrt(torch.sum(torch.square(gradients), dim=2))
        gp = LAMBDA * torch.mean((slopes - 1.) ** 2)
        loss += gp
        return loss

    def train(self):
        current_iter = 0

        iterations = self.params["train"]["iterations"]
        batch_size = self.params["train"]["batch_size"]
        ema_decay = self.params["train"]["ema_decay"]
        checkpoint_frequency = int(self.params["train"]["checkpoint_frequency"])

        progress_bar = tqdm(range(iterations), desc="Training Progress")

        generated_passwords = self.train_dataset[0]
        human_passwords = self.train_dataset[1]

        while current_iter < iterations:

            for fake_data, real_data in zip(get_batches(generated_passwords, batch_size),
                                            get_batches(human_passwords, batch_size),
                                            ):

                if current_iter >= iterations:
                    break

                self.model.train()

                fake_data = [tuple(self.tokenizer.encode_data(password)) for password in fake_data]
                fake_data = torch.tensor(np.array(fake_data), device=self.device)
                fake_data = self.transform_data(fake_data.to(torch.int64), self.tokenizer.vocab_size).to(torch.float32)

                real_data = [tuple(self.tokenizer.encode_data(password)) for password in real_data]
                real_data = torch.tensor(np.array(real_data), device=self.device)
                real_data = self.transform_data(real_data.to(torch.int64), self.tokenizer.vocab_size).to(torch.float32)

                fake_output = self.model(fake_data)
                real_output = self.model(real_data)

                loss = self.compute_wgan_loss(fake_data, real_data, fake_output, real_output)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                self.update_ema(self.model, self.ema_model, ema_decay)

                if current_iter % checkpoint_frequency == 0:
                    self.save(self.name+".pt")

                current_iter += 1
                progress_bar.update(1)

        self.save(self.name+".pt")

    def eval(self):
        batch_size = self.params["train"]["batch_size"]

        self.ema_model.eval()
        self.ema_model.to(self.device)

        generated_passwords = self.test_dataset[0]
        human_passwords = self.test_dataset[1]

        current_iter = 0
        iterations = min(len(generated_passwords) // batch_size, len(human_passwords) // batch_size) + 1

        progress_bar_eval = tqdm(range(iterations), desc="Eval Progress")

        loss_array = []

        with torch.no_grad():

            for fake_data, real_data in zip(get_batches(generated_passwords, batch_size),
                                            get_batches(human_passwords, batch_size),
                                            ):

                if current_iter >= iterations:
                    break

                fake_data = [tuple(self.tokenizer.encode_data(password)) for password in fake_data]
                fake_data = torch.tensor(np.array(fake_data), device=self.device)
                fake_data = self.transform_data(fake_data.to(torch.int64), self.tokenizer.vocab_size).to(torch.float32)

                real_data = [tuple(self.tokenizer.encode_data(password)) for password in real_data]
                real_data = torch.tensor(np.array(real_data), device=self.device)
                real_data = self.transform_data(real_data.to(torch.int64), self.tokenizer.vocab_size).to(torch.float32)

                fake_output = self.ema_model(fake_data)
                real_output = self.ema_model(real_data)

                loss = torch.mean(fake_output) - torch.mean(real_output)
                loss_array.append(loss.item())

                progress_bar_eval.update(1)
                current_iter += 1

        avg_loss = np.abs(np.mean(loss_array))
        print("Divergence: ", avg_loss)
        return avg_loss