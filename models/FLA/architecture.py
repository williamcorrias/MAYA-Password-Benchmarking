from torch import nn

class LSTM(nn.Module):
    def __init__(self, lstm_hidden_size, dense_hidden_size, vocab_size, context_len, train_backwards=True):
        super(LSTM, self).__init__()
        self.train_backwards = train_backwards

        self.lstm = nn.LSTM(input_size=vocab_size, hidden_size=lstm_hidden_size, num_layers=3, batch_first=True)

        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(in_features=(lstm_hidden_size * context_len), out_features=dense_hidden_size)
        self.fc2 = nn.Linear(in_features=dense_hidden_size, out_features=vocab_size)

    def forward(self, x):
        if self.train_backwards:
            x = x.flip(1)

        x, _ = self.lstm(x)

        if self.train_backwards:
            x = x.flip(1)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x
