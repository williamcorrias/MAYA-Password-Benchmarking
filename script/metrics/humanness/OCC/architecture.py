import torch.nn as nn

class OneClass(nn.Module):

    def __init__(self, input_dim, num_hidden, output_dim, num_layers, dropout_active, dropout_prob):
        super(OneClass, self).__init__()

        layers = []

        if dropout_active:
            layers.append(nn.Dropout(dropout_prob))

        layers.append(nn.Linear(input_dim, num_hidden))
        layers.append(nn.Tanh())
        for i in range(num_layers - 1):

            if dropout_active:
                layers.append(nn.Dropout(dropout_prob))

            layers.append(nn.Linear(num_hidden, num_hidden))
            layers.append(nn.Tanh())

        layers.append(nn.Linear(num_hidden, output_dim))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


