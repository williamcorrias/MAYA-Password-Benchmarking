import torch
import numpy as np
import os
from tqdm import tqdm
from script.metrics.OCC.architecture import OneClass
from script.metrics.utils.common_op import get_batches
from script.utils.gpu import select_gpu
from sklearn.neighbors import NearestNeighbors


class OCC:
    def __init__(self, data, params, name):
        if torch.cuda.is_available():
            free_gpu = select_gpu()
            self.device = torch.device(f'cuda:{free_gpu}')
        elif torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        self.name = name

        self.generated_passwords, self.real_passwords = data

        self.params = params
        self.loss_fn = self.params['train']['loss_fn']

        tmp_array = [1, ] * params['train']['output_dim']
        self.center = torch.tensor([tmp_array]).to(self.device)
        self.radius = self.params['hyperparams']['Radius']
        self.nu = self.params['hyperparams']['nu']

        self.model = self.build_model().to(self.device)

        lr = int(params['train']['lr_rate'])
        weight_decay = float(params['train']['weight_decay'])
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr, weight_decay=weight_decay)

        self.checkpoint_dir = os.path.join(os.getcwd(), "script", "metrics", "OCC", "checkpoints")
        if not os.path.exists(self.checkpoint_dir):
            os.mkdir(self.checkpoint_dir)

    def save(self, fname):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, os.path.join(self.checkpoint_dir, fname))

    def load(self, fname):
        state_dicts = torch.load(os.path.join(self.checkpoint_dir, fname), map_location=self.device)
        self.model.load_state_dict(state_dicts['model_state_dict'])
        self.optimizer.load_state_dict(state_dicts['optimizer_state_dict'])

    def build_model(self):
        input_dim = self.params['train']['input_dim']
        num_hidden = self.params['train']['num_hidden']
        output_dim = self.params['train']['output_dim']
        num_layers = self.params['train']['num_layers']
        dropout_active = self.params['train']['dropout_active']
        dropout_prob = self.params['train']['dropout_prob']

        model = OneClass(input_dim, num_hidden, output_dim, num_layers, dropout_active, dropout_prob)
        return model

    def train(self):
        warm_up_epochs = self.params['train']['warm_up_epochs']
        batch_size = min(self.params['train']['batch_size'], len(self.real_passwords))

        losses = []
        epochs = self.params['train']['epochs']
        progress_bar = tqdm(total=epochs * len(self.real_passwords))

        self.model.train()

        for epoch in range(epochs):
            for x in get_batches(self.real_passwords, batch_size):
                x = torch.tensor(np.array(x), device=self.device).float()

                self.model.zero_grad()
                self.optimizer.zero_grad()

                outputs = self.model(x)

                if self.loss_fn == "SoftBoundary":
                    loss = self.SoftBoundaryLoss(outputs, self.radius, self.center, self.nu)
                elif self.loss_fn == "OneClass":
                    loss = self.OneClassLoss(outputs, self.center)

                self.optimizer.zero_grad()

                loss.backward(retain_graph=True)
                losses.append(loss.detach().cpu().numpy())

                self.optimizer.step()

                if (epoch >= warm_up_epochs) and (self.loss_fn == "SoftBoundary"):
                    dist = torch.sum((outputs - self.center) ** 2, dim=1)
                    self.radius = torch.tensor(self.get_radius(dist, self.nu))

                progress_bar.update(batch_size)

            self.save(self.name + ".pt")

    def eval(self):
        self.model.eval()

        batch_size = int(self.params['test']['batch_size'])

        k1 = min(len(self.generated_passwords), batch_size)
        k2 = min(len(self.real_passwords), batch_size)

        real_passwords = np.array(self.real_passwords)
        generated_passwords = np.array(self.generated_passwords)

        p_idx = np.random.choice(generated_passwords.shape[0], k1, replace=False)
        q_idx = np.random.choice(real_passwords.shape[0], k2, replace=False)

        x_real = torch.tensor(generated_passwords[p_idx], device=self.device).float()
        x_gen = torch.tensor(real_passwords[q_idx], device=self.device).float()

        x_true_out = self.model(x_real).float().detach().cpu().numpy()
        x_gen_out = self.model(x_gen).float().detach().cpu().numpy()

        alphas, alpha_precision_curve, delta_alpha_precision, beta_recall_curve, delta_beta_recall, authen = (
            self.compute_metrics(x_true_out, x_gen_out, self.center))

        return alphas, alpha_precision_curve, delta_alpha_precision, beta_recall_curve, delta_beta_recall, authen

    def OneClassLoss(self, outputs, center):
        dist = torch.sum((outputs - center) ** 2, dim=1)
        loss = torch.mean(dist)
        return loss

    def SoftBoundaryLoss(self, outputs, radius, center, nu):
        dist = torch.sum((outputs - center) ** 2, dim=1)
        scores = dist - radius ** 2
        loss = radius ** 2 + (1 / nu) * torch.mean(torch.max(torch.zeros_like(scores), scores))
        return loss

    def get_radius(self, dist, nu):
        return np.quantile(np.sqrt(dist.clone().data.float().cpu().numpy()), 1 - nu)

    def compute_metrics(self, real_data, synthetic_data, emb_center):
        n_steps = 30
        alphas = np.linspace(0, 1, n_steps)
        radius = np.quantile(torch.sqrt(torch.sum((torch.tensor(real_data).float() - emb_center.cpu()) ** 2, dim=1)),
                             alphas)
        alpha_precision_curve, delta_alpha_precision = self.compute_alpha_precision(real_data, synthetic_data,
                                                                                    emb_center, radius, alphas)

        beta_recall_curve, delta_beta_recall = self.compute_beta_recall(real_data, synthetic_data, radius, alphas)

        authenticity = self.compute_authenticity(real_data, synthetic_data)

        return alphas, alpha_precision_curve, delta_alpha_precision, beta_recall_curve, delta_beta_recall, authenticity

    def compute_alpha_precision(self, real_data, synthetic_data, emb_center, radius, alphas):
        alpha_precision_curve = []
        synth_dist_to_center = torch.sqrt(
            torch.sum((torch.tensor(synthetic_data).float() - emb_center.cpu()) ** 2, dim=1))
        for k in range(len(radius)):
            precision_audit_mask = (synth_dist_to_center <= radius[k]).detach().float().cpu().numpy()
            alpha_precision = np.mean(precision_audit_mask)
            alpha_precision_curve.append(alpha_precision)

        delta_alpha_precision = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(alpha_precision_curve))) * (
                alphas[1] - alphas[0])
        return alpha_precision_curve, delta_alpha_precision

    def tune_nearest_neighbors(self, real_data_train, real_data_test, radius, alpha):
        k_vals = list(range(2, 10))
        epsilon = 0.025
        errs = []

        for k in k_vals:
            beta_coverage = self.evaluate_beta_recall_kNN(real_data_train, real_data_test, radius, alpha, k)
            errs.append(beta_coverage)

        k_opt = k_vals[np.argmin(np.abs(np.array(errs) - (alpha * (1 - epsilon))))]
        return k_opt

    def evaluate_beta_recall_kNN(self, real_data, synthetic_data, radius, alpha, k):
        synth_center = torch.tensor(np.mean(synthetic_data, axis=0)).float()

        neigh_real = NearestNeighbors(n_neighbors=k, n_jobs=-1, p=2).fit(real_data)
        real_to_real_distances, _ = neigh_real.kneighbors(real_data)

        neigh_synth = NearestNeighbors(n_neighbors=1, n_jobs=-1, p=2).fit(synthetic_data)
        real_to_synth_distances, real_to_synth_idx = neigh_synth.kneighbors(real_data)

        real_to_real_distances = torch.from_numpy(real_to_real_distances[:, -1].squeeze())
        real_to_synth_distances = torch.from_numpy(real_to_synth_distances.squeeze())
        real_to_synth_idx = real_to_synth_idx.squeeze()

        real_synth_closest = synthetic_data[real_to_synth_idx]
        real_synth_closest_d = torch.sqrt(
            torch.sum((torch.tensor(real_synth_closest).float() - synth_center) ** 2, dim=1))

        closest_synth_Radii = np.quantile(
            torch.sqrt(torch.sum((torch.tensor(synthetic_data).float() - synth_center) ** 2, dim=1)), [alpha])

        beta_coverage = np.mean(((real_to_synth_distances <= real_to_real_distances) * (
                real_synth_closest_d <= closest_synth_Radii[0])).detach().float().cpu().numpy())

        return beta_coverage

    def compute_beta_recall(self, real_data, synthetic_data, radius, alphas):
        beta_recall_curve = []

        n_data = real_data.shape[0]
        real_data_train = real_data[: int(np.floor(n_data / 2)), :]
        real_data_test = real_data[int(np.floor(n_data / 2)):, :]

        for u in range(len(radius)):
            k_opt = self.tune_nearest_neighbors(real_data_train, real_data_test, radius[u], alphas[u])
            beta_coverage = self.evaluate_beta_recall_kNN(real_data, synthetic_data, radius[u], alphas[u], k_opt)
            beta_recall_curve.append(beta_coverage)

        delta_beta_recall = 1 - 2 * np.sum(np.abs(np.array(alphas) - np.array(beta_recall_curve))) * (
                alphas[1] - alphas[0])

        return beta_recall_curve, delta_beta_recall

    def compute_authenticity(self, real_data, synthetic_data):
        neigh_real = NearestNeighbors(n_neighbors=2, n_jobs=-1, p=2).fit(real_data)

        dist_synth_to_real, neigh_synth_to_real = neigh_real.kneighbors(synthetic_data)
        dist_synth_to_real = torch.from_numpy(dist_synth_to_real[:, 0].squeeze())
        neigh_synth_to_real = neigh_synth_to_real[:, 0].squeeze()

        dist_real_to_real, neigh_real_to_real = neigh_real.kneighbors(real_data)
        dist_real_to_real = torch.from_numpy(dist_real_to_real[:, 1].squeeze())

        authen = dist_real_to_real[neigh_synth_to_real] < dist_synth_to_real
        authenticity = np.mean(authen.numpy())
        return authenticity
