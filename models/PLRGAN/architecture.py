from torch import nn
import torch.nn.functional as F

def softmax(logits, num_classes):
    logits_reshaped = logits.reshape(-1, num_classes)
    softmax_output = F.softmax(logits_reshaped, dim=1)
    return softmax_output.view(logits.size())

class ResidualBlock(nn.Module):
    def __init__(self, layer_dim, batch_norm):
        super(ResidualBlock, self).__init__()
        self.dim_BNK = layer_dim // 2

        self.batch_norm = batch_norm
        if batch_norm:
            self.res_block = nn.Sequential(
                nn.BatchNorm1d(layer_dim, affine=True),
                nn.ReLU(),
                nn.Conv1d(in_channels=layer_dim, out_channels=self.dim_BNK, kernel_size=3, padding='same'),
                nn.BatchNorm1d(self.dim_BNK, affine=True),
                nn.ReLU(),
                nn.Conv1d(in_channels=self.dim_BNK, out_channels=self.dim_BNK, kernel_size=5, padding='same'),
                nn.BatchNorm1d(self.dim_BNK, affine=True),
                nn.ReLU(),
                nn.Conv1d(in_channels=self.dim_BNK, out_channels=layer_dim, kernel_size=3, padding='same'),
            )
        else:
            self.res_block = nn.Sequential(
                nn.ReLU(),
                nn.Conv1d(in_channels=layer_dim, out_channels=self.dim_BNK, kernel_size=3, padding='same'),
                nn.ReLU(),
                nn.Conv1d(in_channels=self.dim_BNK, out_channels=self.dim_BNK, kernel_size=5, padding='same'),
                nn.ReLU(),
                nn.Conv1d(in_channels=self.dim_BNK, out_channels=layer_dim, kernel_size=3, padding='same'),
            )

    def forward(self, x):
        inputs = x
        out = self.res_block(x)
        return inputs + (0.3 * out)


class Generator(nn.Module):
    def __init__(self, max_length, n_chars, layer_dim=128, with_logits=False):
        super(Generator, self).__init__()

        self.max_length = max_length
        self.n_chars = n_chars
        self.layer_dim = layer_dim
        self.with_logits = with_logits
        self.batch_norm = True

        self.fc1 = nn.Linear(layer_dim, max_length * layer_dim)
        self.res_block = nn.Sequential(
            ResidualBlock(layer_dim, batch_norm=True),
            ResidualBlock(layer_dim, batch_norm=True),
            ResidualBlock(layer_dim, batch_norm=True),
            ResidualBlock(layer_dim, batch_norm=True),
            ResidualBlock(layer_dim, batch_norm=True),
        )
        self.conv1 = nn.Conv1d(in_channels=layer_dim, out_channels=n_chars, kernel_size=1, padding='same')

    def forward(self, x):
        x = self.fc1(x)
        x = x.reshape(-1, self.layer_dim, self.max_length)
        x = self.res_block(x)
        logits = self.conv1(x)
        logits = logits.transpose(1, 2)
        x = softmax(logits, self.n_chars)
        if self.with_logits:
            return x, logits
        return x

class Discriminator(nn.Module):
    def __init__(self, max_length, dict_size, layer_dim=128):
        super(Discriminator, self).__init__()
        self.max_length = max_length
        self.dict_size = dict_size
        self.layer_dim = layer_dim
        self.batch_norm = True

        self.conv1 = nn.Conv1d(in_channels=dict_size, out_channels=layer_dim, kernel_size=1, padding='same')
        self.res_block = nn.Sequential(
            ResidualBlock(layer_dim, batch_norm=False),
            ResidualBlock(layer_dim, batch_norm=False),
            ResidualBlock(layer_dim, batch_norm=False),
            ResidualBlock(layer_dim, batch_norm=False),
            ResidualBlock(layer_dim, batch_norm=False),
        )
        self.fc = nn.Linear(layer_dim * max_length, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv1(x)
        x = self.res_block(x)
        x = x.view(-1, self.max_length * self.layer_dim)
        logits = self.fc(x)
        return logits
