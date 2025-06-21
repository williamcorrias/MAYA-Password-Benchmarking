import gzip
import pickle
from collections import Counter
from various_plot import plot_distribution

file = "datasets/ru/mail/mailru.pickle"
model_name = "mailru"

def read_files(path):
    data = []

    if path.endswith('.gz'):
        with gzip.open(path, 'rt') as f:
            data = f.read().split("\n")

    elif path.endswith('.txt'):
        with open(path, 'r') as f:
            data = f.read().split("\n")

    elif path.endswith('.pickle'):
        with open(path, 'rb') as f:
            data = pickle.load(f)
            data = [password.strip() for password in data]
    return data

data = read_files(file)

counter = Counter(data)

frequencies = sorted(counter.values(), reverse=True)
total_uniques = len(frequencies)
frequencies = [freq for freq in frequencies if freq >= 3]

x_axis = [(x / total_uniques) for x in range(len(frequencies))]

plot_distribution(x_data=x_axis,
                  y_data=frequencies,
                  y_log_scale=True,
                  x_caption='Fraction of passwords',
                  y_caption='Frequencies',
                  dest_path=f"script/plotters/output/zipf/{model_name}.pdf",
                  color="#1f77b4"
                  )

