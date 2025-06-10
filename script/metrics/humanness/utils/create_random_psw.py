import random
from tqdm import tqdm

random.seed(42)

allowed_lengths = [6, 7, 8, 9, 10, 11, 12]
charbag = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789~!@#$%^&*(),.<>/?'\"{}[]\\|-_=+;:`"

size = 10 ** 7
path = "10million_fully_random_psw.txt.txt"

random_psw = []
progress_bar = tqdm(total=size)

for i in range(size):
    max_len = random.choice(allowed_lengths)
    psw = random.sample(charbag, max_len)
    psw = "".join(psw) + "\n"

    random_psw.append(psw)
    progress_bar.update(1)

with open(path, "w") as output_file:
    output_file.write("".join(random_psw))