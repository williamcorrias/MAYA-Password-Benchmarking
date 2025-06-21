import torch.nn as nn
import torch


class CNNClassifier(nn.Module):

    def __init__(self, input_dim, conv1_1dim, conv1_2dim, conv1_3dim, context_len, output_size):
        super(CNNClassifier, self).__init__()

        self.conv1_1 = nn.Conv1d(in_channels=input_dim, out_channels=conv1_1dim, kernel_size=5, padding="same")
        self.conv1_2 = nn.Conv1d(in_channels=conv1_1dim, out_channels=conv1_2dim, kernel_size=5, padding="same")
        self.conv1_3 = nn.Conv1d(in_channels=conv1_2dim, out_channels=conv1_3dim, kernel_size=5, padding="same")

        self.flatten = nn.Flatten()
        self.fc = nn.Linear(conv1_3dim * context_len, output_size)

        self.he_style_initialization()

    def he_style_initialization(self):
        for layer in self.modules():
            if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.Linear):
                torch.nn.init.kaiming_normal_(layer.weight)
                if layer.bias is not None:
                    torch.nn.init.zeros_(layer.bias)

    def swish_activation(self, x):
        return x * torch.sigmoid(x)

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.swish_activation(x)
        x = self.conv1_1(x)
        x = self.swish_activation(x)
        x = self.conv1_2(x)
        x = self.swish_activation(x)
        x = self.conv1_3(x)
        x = self.swish_activation(x)
        x = x.transpose(1, 2)
        x = self.flatten(x)
        x = self.fc(x)
        return x
