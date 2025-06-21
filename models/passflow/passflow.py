import sys
import time
from datetime import timedelta
import numpy as np
import torch
from torch.nn import functional as F
from script.utils.file_operations import read_files

from tqdm import tqdm

from models.passflow.src.real_nvp.real_nvp import RealNVP
from models.passflow.src.real_nvp.coupling_layer import AffineTransform, MaskType

from script.test.model import Model
from script.dataset.dataset import Dataset
from script.plotters.various_plot import tsne_plot


def get_evaluation_params(n_samples):
    n_samples = int(n_samples)
    if n_samples <= 10 ** 5:
        return {
            "alpha": 1,
            "sigma": 0.12,
            "gamma": 2
        }
    elif n_samples <= 10 ** 6:
        return {
            "alpha": 5,
            "sigma": 0.12,
            "gamma": 10
        }
    elif n_samples <= 10 ** 7:
        return {
            "alpha": 50,
            "sigma": 0.12,
            "gamma": 10
        }
    else:
        return {
            "alpha": 50,
            "sigma": 0.15,
            "gamma": 10
        }


class PassFlow(Model):
    def __init__(self, settings):
        self.model = None
        self.optimizer = None
        super().__init__(settings)

    def prepare_data(self, train_passwords, test_passwords, max_length):
        data = Dataset(train_passwords, test_passwords, max_length, self.test_hash)
        data.test_passwords = {np.array(password, dtype=np.uint8).tobytes() for password in data.test_passwords}
        return data

    def load(self, file_to_load):
        try:
            self.init_model()
            state_dicts = torch.load(file_to_load, map_location=self.device)
            self.model.load_state_dict(state_dicts['net'])
            self.model.to(self.device)
            self.optimizer.load_state_dict(state_dicts['optimizer'])
            return 1
        except Exception as e:
            print(f"Exception: {e}")
            return 0

    def init_model(self):
        self.keep_uniques = True

        optimizer = torch.optim.Adam
        dim = self.data.max_length
        lr = self.params['train']['learning_rate']
        num_coupling = self.params['train']['num_coupling']
        mask_pattern = self.params['train']['mask_pattern'] != 'None'
        weight_decay = self.params['train']['weight_decay']
        n_hidden = self.params['train']['n_hidden']
        hidden_size = self.params['train']['hidden_size']
        architecture = "resnet"

        mask = ['left', 'right']
        if not mask_pattern:
            mask_pattern = [MaskType.CHECKERBOARD] * num_coupling

        flows = [AffineTransform(dim, self.device, mask[i % 2], mask_pattern[i], architecture,
                                 n_hidden=n_hidden, hidden_size=hidden_size) for i in range(num_coupling)]

        self.model = RealNVP(dim, self.device, flows).to(self.device)
        trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]

        self.optimizer = optimizer(trainable_parameters, lr=lr, weight_decay=weight_decay)

    def preprocess(self, x, reverse=False):
        charmap_size = float(self.data.charmap_size)

        if reverse:
            x = 1.0 / (1 + torch.exp(-x))
            x -= 0.05
            x /= 0.9

            x *= charmap_size
            return x
        else:
            x /= charmap_size

            x *= 0.9
            x += 0.05
            logit = torch.log(x) - torch.log(1.0 - x)
            log_det = F.softplus(logit) + F.softplus(-logit) + torch.log(torch.tensor(0.9)) \
                      - torch.log(torch.tensor(charmap_size))

            return logit, torch.sum(log_det, dim=1)

    def plot_embedding(self, data_paths, max_length):
        def get_batches(data, batch_size=2048):
            for i in range(0, len(data) - batch_size + 1, batch_size):
                yield np.array([np.array(pwd) for pwd in data[i:i + batch_size]], dtype='uint8')

        datasets = {}
        for idx, dataset in enumerate(data_paths):
            dataset = read_files(dataset)
            datasets[str(idx)] = [self.data.encode_password(str(password) + ("`" * (int(max_length) - len(str(password)))))
                         for password in dataset][:20000]

        embeddings = []
        labels = []

        for idx in datasets.keys():
            for b in get_batches(datasets[idx], batch_size=2048):
                b = torch.tensor(b).to(self.device).float().contiguous()
                logit_x, _ = self.preprocess(b)
                z, _ = self.model.flow(logit_x)

                embeddings.append(z.detach().cpu().numpy())
                labels.extend([idx] * z.shape[0])

        tsne_plot(embeddings, labels, datasets.keys(), data_paths)

    def train(self):
        print("[I] - Launching training")

        dim = self.data.max_length
        batch_size = self.params['train']['batch_size']
        n_epochs = self.params['train']['epochs']

        early_stop_epoch = n_epochs // 2
        early_stop_counter = 0

        start = time.time()

        current_epoch = 0
        n_matches = 0
        checkpoint_frequency = self.params['eval']['checkpoint_frequency']

        self.init_model()

        while current_epoch < n_epochs:

            print(f"Epoch: {current_epoch + 1} / {n_epochs}")
            self.model.train()

            with tqdm(total=self.data.get_train_size()) as bar:
                bar.set_description(f'Epoch {current_epoch}')
                batch_loss_history = []

                for b in self.data.get_batches(batch_size):
                    b = torch.tensor(b).to(self.device).float().contiguous()

                    logit_x, log_det = self.preprocess(b)
                    log_prob = self.model.log_prob(logit_x)
                    log_prob += log_det

                    loss = -torch.mean(log_prob) / float(dim)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    batch_loss = float(loss.data)
                    batch_loss_history.append(batch_loss)

                    epoch_loss = np.mean(batch_loss_history)
                    bar.set_postfix(loss=epoch_loss)
                    bar.update(b.size(0))

                if current_epoch % checkpoint_frequency == 0:
                    matches, _, _ = self.evaluate(n_samples=10 ** 6, validation_mode=True)
                    self.model.reset_prior()

                    if current_epoch >= early_stop_epoch:
                        threshold = int(n_matches + (n_matches * 0.05))
                        if matches < threshold:
                            early_stop_counter += 1
                        else:
                            early_stop_counter = 0

                        if early_stop_counter >= 10:
                            print("[I] - Early stopping")
                            break

                    if matches >= n_matches:
                        n_matches = matches
                        obj = {
                            'optimizer': self.optimizer.state_dict(),
                            'net': self.model.state_dict(),
                        }
                        self.save(obj)

            current_epoch += 1

        end = time.time()
        time_delta = timedelta(seconds=end - start)
        print(f"[T] - Training completed after: {time_delta}")

    def smoothen_samples(self, samples, uniques):
        for idx, sample in enumerate(samples):
            sample = np.around(samples[idx]).astype(np.uint8)
            decoded_sample = sample.tobytes()
            counter = 0
            noise_d = 0.0
            while decoded_sample in uniques:
                if counter > 20:
                    noise_d += 0.05
                    counter = 0
                    if noise_d >= 0.20:
                        break
                sample = np.around(samples[idx] + np.random.normal(0.0, self.params['train']['noise'] + noise_d, self.data.max_length)).astype(np.uint8)
                decoded_sample = sample.tobytes()
                counter += 1
            samples[idx] = sample
        return samples.astype(np.uint8)

    def around_sampling(self, password, num_samples, temperature=0.05):
        self.model.eval()

        pwd = self.data.pad_password(tuple(password))
        pwd = np.array([self.data.encode_password(pwd)] * num_samples).astype(float)

        x = torch.FloatTensor(pwd).to(self.device)
        with torch.no_grad():
            x, _ = self.preprocess(x)
            z, _ = self.model.flow(x)

            z += torch.distributions.Uniform(low=-temperature, high=temperature).sample(z.shape).to(self.device)

            x = self.model.invert_flow(z)
            x = self.preprocess(x, reverse=True)

            return np.around(x.cpu().numpy()).astype(int)

    def interpolate(self, start, target, steps=50):
        self.model.eval()

        with torch.no_grad():
            start = self.data.pad_password(tuple(start))
            target = self.data.pad_password(tuple(target))
            start = np.array([self.data.encode_password(start)]).astype(float)
            target = np.array([self.data.encode_password(target)]).astype(float)

            x1 = torch.FloatTensor(start).to(self.device)
            x2 = torch.FloatTensor(target).to(self.device)

            x1, _ = self.preprocess(x1)
            x2, _ = self.preprocess(x2)

            latents = []

            z1, _ = self.model.flow(x1)
            z2, _ = self.model.flow(x2)

            delta = (z2 - z1) / float(steps)
            latents.append(z1)
            for j in range(1, steps):
                latents.append(z1 + delta * float(j))
            latents.append(z2)

            latents = torch.cat(latents, dim=0)
            logit_results = self.model.invert_flow(latents)
            results = self.preprocess(logit_results, reverse=True)

            return np.around(results.cpu().numpy()).astype(int)

    def eval_init(self, n_samples, evaluation_batch_size):
        self.model.eval()

        ds = bool(self.settings.get("ds", False))
        gs = bool(self.settings.get("gs", False))

        eval_dict = {
            'gs': gs,
            'ds': ds,
        }

        if ds or gs:
            p = get_evaluation_params(n_samples)
            alpha, sigma, gamma = p['alpha'], p['sigma'], p['gamma']
            matched_history = dict()
            count_samples = 0

            eval_dict.update({
                'alpha': alpha,
                'sigma': sigma,
                'gamma': gamma,
                'matched_history': matched_history,
                'count_samples': count_samples,
                'old_matches': set(),
                'dim': self.data.max_length,
            })

        sys.stdout.flush()
        return eval_dict

    def sample(self, evaluation_batch_size, eval_dict):
        with torch.no_grad():
            raw_samples = self.model.sample(evaluation_batch_size)
            samples = self.preprocess(raw_samples, reverse=True).to('cpu').numpy()

            if eval_dict['gs'] and self.guesses is not None and self.params['train']['noise'] != 0:
                samples = self.smoothen_samples(samples, self.guesses)
            else:
                samples = np.around(samples).astype(np.uint8)

            encoded_samples = {x.tobytes() for x in samples}
            return encoded_samples

    def guessing_strategy(self, evaluation_batch_size, eval_dict):
        if eval_dict['gs'] or eval_dict['ds']:
            self.dynamic_sampling(evaluation_batch_size, eval_dict)

    def post_sampling(self, eval_dict):
        self.guesses = [tuple(np.frombuffer(x, dtype=np.uint8)) for x in self.guesses]
        self.matches = [tuple(np.frombuffer(x, dtype=np.uint8)) for x in self.matches]

        self.model.reset_prior()

    def dynamic_sampling(self, evaluation_batch_size, eval_dict):
        new_matches = set(self.matches) - eval_dict['old_matches']

        for match in new_matches:
            eval_dict['matched_history'][match] = 0

        with torch.no_grad():
            if len(self.matches) >= eval_dict['alpha'] and len(eval_dict['matched_history']) > 0:
                idxs = np.random.randint(0, len(eval_dict['matched_history']), evaluation_batch_size, np.int32)

                key_list = np.fromiter(eval_dict['matched_history'].keys(), dtype=object)[idxs]
                encoded_key_list = np.array([np.frombuffer(key, dtype=np.uint8) for key in key_list])

                for key in key_list:
                    if key in eval_dict['matched_history']:
                        eval_dict['matched_history'][key] += 1
                        if eval_dict['matched_history'][key] > eval_dict['gamma']:
                            del eval_dict['matched_history'][key]

                x = torch.FloatTensor(encoded_key_list).to(self.device)
                x, _ = self.preprocess(x)
                matched_z = self.model.flow(x)[0].to('cpu').numpy()

                dynamic_mean = matched_z
                dynamic_var = np.full((evaluation_batch_size, eval_dict['dim']), eval_dict['sigma'], dtype=np.float32)
                self.model.set_prior(dynamic_mean, dynamic_var)
            else:
                self.model.reset_prior()

        eval_dict['old_matches'] = set(self.matches)