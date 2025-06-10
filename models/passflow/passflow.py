import copy
import glob
import math
import os
import gzip
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

class PassFlow(Model):
    def __init__(self, settings):
        super().__init__(settings)

    def prepare_data(self):
        self.data = Dataset(self.train_path, self.test_path, self.max_length, self.test_hash, max_train_size=0, max_test_size=0, skip_unk=True)
        self.data.test_passwords = {np.array(password, dtype=np.uint8).tobytes() for password in
                                    self.data.test_passwords}

    def prepare_model(self):
        self.dim = self.data.max_length
        self.learning_rate = self.params['train']['learning_rate']
        self.num_coupling = self.params['train']['num_coupling']

        mask_pattern = self.params['train']['mask_pattern'] != 'None'
        weight_decay = self.params['train']['weight_decay']
        n_hidden = self.params['train']['n_hidden']
        hidden_size = self.params['train']['hidden_size']
        architecture = "resnet"

        mask = ['left', 'right']
        if not mask_pattern:
            mask_pattern = [MaskType.CHECKERBOARD] * self.num_coupling

        self.mask_pattern = ''.join(str(int(m)) for m in mask_pattern)

        flows = [AffineTransform(self.dim, self.device, mask[i % 2], mask_pattern[i], architecture,
                                 n_hidden=n_hidden, hidden_size=hidden_size) for i in range(self.num_coupling)]

        self.model = RealNVP(self.dim, self.device, flows)
        self.model.to(self.device)

        self.trainable_parameters = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.Adam(self.trainable_parameters, lr=self.learning_rate, weight_decay=weight_decay)

        self.noise = self.params['train']['noise']

        self.eval_args = {
            "alpha": 1,
            "sigma": 0.12,
            "gamma": 2,
            "ds": self.settings["ds"] if "ds" in self.settings else False,
            "gs": self.settings["gs"] if "gs" in self.settings else False,
            "evaluation_batch_size": self.params['eval']['evaluation_batch_size'],
            "n_samples": self.n_samples
        }
        self.eval_args.update(self.get_evaluation_params(self.n_samples))

        self.checkpoint_frequency = self.params['eval']['checkpoint_frequency']
        self.checkpoint_dir = './checkpoints/passflow/'+str(self.train_hash)
        if not os.path.exists(os.path.join(os.getcwd(), self.checkpoint_dir)):
            os.makedirs(os.path.join(os.getcwd(), self.checkpoint_dir), exist_ok=True)

        self.evaluation_batch_size = self.params['eval']['evaluation_batch_size']

        self.save_guesses = self.params['eval']['save_guesses']
        self.save_matches = self.params['eval']['save_matches']
        return self.eval_args, self.checkpoint_dir, self.save_guesses, self.save_matches

    def get_evaluation_params(self, n_samples):
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

    def save(self, fname):
        for filename in glob.glob(os.path.join(self.checkpoint_dir, "mid-"+self.checkpoint_name)):
            os.remove(filename)

        torch.save({
            'optimizer': self.optimizer.state_dict(),
            'net': self.model.state_dict(),
        }, os.path.join(self.checkpoint_dir, fname))

    def load(self, fname):
        try:
            state_dicts = torch.load(os.path.join(self.checkpoint_dir, fname), map_location=self.device, weights_only=False)
            self.model.load_state_dict(state_dicts['net'])
            self.model.to(self.device)
            self.optimizer.load_state_dict(state_dicts['optimizer'])
            return 1
        except Exception as e:
            print(f"Exception: {e}")
            return 0

    def preprocess(self, x, reverse=False):
        charmap_size = float(self.data.get_charmap_size())

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

    def plot_embedding(self, data_paths):
        def get_batches(data, batch_size=2048):
            for i in range(0, len(data) - batch_size + 1, batch_size):
                yield np.array([np.array(pwd) for pwd in data[i:i + batch_size]], dtype='uint8')

        datasets = {}
        for idx, dataset in enumerate(data_paths):
            dataset = read_files(dataset)
            datasets[str(idx)] = [self.data.encode_password(str(password) + ("`" * (int(self.max_length) - len(str(password)))))
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
        self.model.train()

        self.batch_size = self.params['train']['batch_size']
        n_epochs = self.params['train']['epochs']

        train_losses = []

        print("[I] - Launching training")
        start = time.time()

        early_stop_epoch = n_epochs // 2
        early_stop_counter = 0

        mini_eval_args = copy.deepcopy(self.eval_args)
        mini_eval_args['n_samples'] = 10**6

        self.n_matches = 0

        while self.current_epoch < n_epochs:
            self.current_epoch += 1

            print(f"Epoch: {self.current_epoch} / {n_epochs}")

            self.model.train()

            with tqdm(total=self.data.get_train_size()) as bar:
                bar.set_description(f'Epoch {self.current_epoch}')
                batch_loss_history = []

                for b in self.data.get_batches(self.batch_size):
                    b = torch.tensor(b).to(self.device).float().contiguous()

                    logit_x, log_det = self.preprocess(b)
                    log_prob = self.model.log_prob(logit_x)
                    log_prob += log_det

                    loss = -torch.mean(log_prob) / float(self.dim)
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    batch_loss = float(loss.data)
                    batch_loss_history.append(batch_loss)

                    epoch_loss = np.mean(batch_loss_history)
                    bar.set_postfix(loss=epoch_loss)
                    bar.update(b.size(0))

                train_losses.append(epoch_loss)

                if self.current_epoch % self.checkpoint_frequency == 0:
                    matches, _, _ = self.evaluate(mini_eval_args, save_matches=False, save_guesses=False)

                    if self.current_epoch >= early_stop_epoch:
                        threshold = int(self.n_matches + (self.n_matches * 0.05))
                        if matches < threshold:
                            early_stop_counter += 1
                        else:
                            early_stop_counter = 0

                        if early_stop_counter >= 10:
                            print("[I] - Early stopping")
                            break

                    if matches >= self.n_matches:
                        self.n_matches = matches
                        self.save("mid-"+self.checkpoint_name)

        end = time.time()
        self.time_delta = timedelta(seconds=end - start)

        self.finalize_checkpoint(self.checkpoint_name)

        print("[T] - Training completed after: " + str(self.time_delta))
        return train_losses

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
                sample = np.around(samples[idx] + np.random.normal(0.0, self.noise + noise_d, self.dim)).astype(np.uint8)
                decoded_sample = sample.tobytes()
                counter += 1
            samples[idx] = sample
        return samples.astype(np.uint8)

    def sample(self, num_samples, save_guesses, uniques=None):
        with torch.no_grad():
            raw_samples = self.model.sample(num_samples)
            samples = self.preprocess(raw_samples, reverse=True).to('cpu').numpy()

            if uniques is not None and self.noise != 0:
                samples = self.smoothen_samples(samples, uniques)
            else:
                samples = np.around(samples).astype(np.uint8)

            encoded_samples = set()

            if save_guesses:
                with gzip.open(os.path.join(self.guesses_dir, "guesses.gz"),'at') as file:
                    for password in samples:
                        encoded_samples.add(password.tobytes())
                        decoded_password = ''.join(map(str, self.data.decode_password(password))).replace('`', '')
                        file.write(decoded_password + '\n')
            else:
                encoded_samples = {x.tobytes() for x in samples}

            return encoded_samples

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

    def evaluate(self, eval_args, save_matches, save_guesses):
        if eval_args["ds"]:
            matches, match_percentage, test_size = self.dynamic_sampling(eval_args["n_samples"],
                                                                       eval_args["evaluation_batch_size"],
                                                                       sigma=eval_args["sigma"],
                                                                       alpha=eval_args["alpha"],
                                                                       gamma=eval_args["gamma"],
                                                                       save_matches=save_matches,
                                                                       save_guesses=save_guesses)
        elif eval_args["gs"]:
            matches, match_percentage, test_size = self.dynamic_sampling_gs(eval_args["n_samples"],
                                                                          eval_args["evaluation_batch_size"],
                                                                          sigma=eval_args["sigma"],
                                                                          alpha=eval_args["alpha"],
                                                                          gamma=eval_args["gamma"],
                                                                          save_matches=save_matches,
                                                                          save_guesses=save_guesses)
        else:
            matches, match_percentage, test_size = self.evaluate_sampling(eval_args["n_samples"],
                                                                        eval_args["evaluation_batch_size"],
                                                                        save_matches=save_matches,
                                                                        save_guesses=save_guesses
                                                                        )

        return matches, match_percentage, test_size

    def evaluate_sampling(self, n_samples, max_batch_size, save_matches, save_guesses):
        print("[I] - Generating passwords")
        self.model.eval()

        max_batch_size = int(max_batch_size)
        if n_samples < max_batch_size:
            batches, max_batch_size = 1, int(n_samples)
        else:
            batches = math.floor(n_samples / max_batch_size)

        matches = set()
        with (tqdm(range(batches)) as pbar):
            pbar.set_description(desc='Generating sample batch')
            for _ in pbar:
                generated_psw = self.sample(max_batch_size, save_guesses)
                current_match = generated_psw & self.data.test_passwords
                matches.update(current_match)
                pbar.set_postfix({'Matches found': {len(matches)},
                                  'Test set %': ({len(matches) / self.data.get_test_size() * 100.0})})

        if save_matches == True:
            with gzip.open(os.path.join(self.matches_dir, "matches.gz"), 'at') as file:
                for match in matches:
                    decoded_match = np.frombuffer(match, dtype=np.uint8)
                    decoded_match = ''.join(map(str, self.data.decode_password(decoded_match))).replace('`', '')
                    file.write(decoded_match + '\n')

        matches = len(matches)
        test_size = self.data.get_test_size()
        match_percentage = f'{(matches / test_size) * 100:.2f}%'
        print(f'{matches} matches found ({match_percentage} of test set).')
        self.model.reset_prior()

        return matches, match_percentage, test_size


    def dynamic_sampling(self, n_samples, max_batch_size, sigma, alpha, gamma, save_matches, save_guesses):
        print("[I] - Generating passwords")
        self.model.eval()

        matches = set()
        matched_history = dict()
        batches = math.floor(n_samples / max_batch_size)
        count_samples = 0
        sys.stdout.flush()

        with torch.no_grad():
            with tqdm(range(batches)) as pbar:
                for _ in pbar:
                    generated_psw = self.sample(max_batch_size, save_guesses)
                    count_samples += len(generated_psw)

                    current_match = generated_psw & self.data.test_passwords
                    new_matches = current_match - matches

                    matches.update(current_match)

                    for match in new_matches:
                        matched_history[match] = 0

                    match_set = len(matched_history)

                    if len(matches) >= alpha and len(matched_history) > 0:
                        idxs = np.random.randint(0, len(matched_history), max_batch_size, np.int32)

                        key_list = np.fromiter(matched_history.keys(), dtype=object)[idxs]
                        encoded_key_list = np.array([np.frombuffer(key, dtype=np.uint8) for key in key_list])

                        for key in key_list:
                            if key in matched_history:
                                matched_history[key] += 1
                                if matched_history[key] > gamma:
                                    del matched_history[key]

                        x = torch.FloatTensor(encoded_key_list).to(self.device)
                        x, _ = self.preprocess(x)
                        matched_z = self.model.flow(x)[0].to('cpu').numpy()

                        dynamic_mean = matched_z
                        dynamic_var = np.full((max_batch_size, self.dim), sigma, dtype=np.float32)
                        self.model.set_prior(dynamic_mean, dynamic_var)
                    else:
                        self.model.reset_prior()

                    pbar.set_postfix({'Num samples': count_samples,
                                      'Matches found': {len(matches)},
                                      'Test set %': ({len(matches) / self.data.get_test_size() * 100.0})})

                if save_matches == True:
                    with gzip.open(os.path.join(self.matches_dir, "matches.gz"), 'at') as file:
                        for match in matches:
                            decoded_match = np.frombuffer(match, dtype=np.uint8)
                            decoded_match = ''.join(map(str, self.data.decode_password(decoded_match))).replace('`', '')
                            file.write(decoded_match + '\n')

                matches = len(matches)
                test_size = self.data.get_test_size()
                match_percentage = f'{(matches / test_size) * 100:.2f}%'
                print(f'{matches} matches found ({match_percentage} of test set). ')
                self.model.reset_prior()

                return matches, match_percentage, test_size

    def dynamic_sampling_gs(self, n_samples, max_batch_size, sigma, alpha, gamma, save_matches, save_guesses, running_mean_len=16):
        print("[I] - Generating passwords")
        self.model.eval()

        matches = set()
        uniques = set()
        matched_history = dict()
        batches = math.floor(n_samples / max_batch_size)
        count_samples = 0
        running_matches = np.full(running_mean_len, 0)
        running_idx = 0
        sys.stdout.flush()

        with torch.no_grad():
            with tqdm(range(batches)) as pbar:
                for _ in pbar:
                    generated_psw = self.sample(max_batch_size, save_guesses, uniques=uniques)
                    count_samples += len(generated_psw)

                    current_match = generated_psw & self.data.test_passwords
                    new_matches = current_match - matches
                    prev_matches = len(matches)

                    matches.update(current_match)
                    uniques.update(generated_psw)

                    running_matches[running_idx] = len(matches) - prev_matches
                    running_idx = (running_idx + 1) % running_mean_len

                    for match in new_matches:
                        matched_history[match] = 0

                    match_set_len = len(matched_history)

                    if len(matches) >= alpha and len(matched_history) > 0:
                        idxs = np.random.randint(0, len(matched_history), max_batch_size, np.int32)

                        key_list = np.fromiter(matched_history.keys(), dtype=object)[idxs]
                        encoded_key_list = np.array([np.frombuffer(key, dtype=np.uint8) for key in key_list])

                        for key in key_list:
                            if key in matched_history:
                                matched_history[key] += 1
                                if matched_history[key] > gamma:
                                    del matched_history[key]

                        x = torch.FloatTensor(encoded_key_list).to(self.device)
                        x, _ = self.preprocess(x)
                        matched_z = self.model.flow(x)[0].to('cpu').numpy()

                        dynamic_mean = matched_z
                        dynamic_var = np.full((max_batch_size, self.dim), sigma, dtype=np.float32)
                        self.model.set_prior(dynamic_mean, dynamic_var)

                    else:
                        self.model.reset_prior()

                    pbar.set_postfix({'Num samples': count_samples,
                                      'Matches found': {len(matches)},
                                      'Unique samples': {len(uniques)},
                                      'Running Matches': {np.average(running_matches)},
                                      'Match Set': {match_set_len},
                                      'Test set %': ({len(matches) / self.data.get_test_size() * 100.0})})

                if save_matches == True:
                    with gzip.open(os.path.join(self.matches_dir, "matches.gz"), 'at') as file:
                        for match in matches:
                            decoded_match = np.frombuffer(match, dtype=np.uint8)
                            decoded_match = ''.join(map(str, self.data.decode_password(decoded_match))).rstrip('`')
                            file.write(decoded_match + '\n')

                matches = len(matches)
                uniques = len(uniques)
                test_size = self.data.get_test_size()
                match_percentage = f'{(matches / test_size) * 100:.2f}%'
                print(f'{matches} matches found ({match_percentage} of test set). out of {uniques} unique samples.')
                self.model.reset_prior()

                return matches, match_percentage, test_size