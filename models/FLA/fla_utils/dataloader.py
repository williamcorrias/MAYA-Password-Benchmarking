import numpy as np

from models.FLA.fla_utils.tokenizer import Tokenizer

class DataLoader():
    def __init__(self, train_passwords, test_passwords, max_length, params):
        self.PASSWORD_END = '\n'

        self.max_length = max_length

        self.char_bag = params['data']['char_bag']
        self.tokenizer = Tokenizer(self.char_bag, self.max_length, self.PASSWORD_END, padding_character=False)

        self.train_passwords = train_passwords
        self.test_passwords = set(test_passwords)

    def prepare_y_data(self, y_str_list):
        y_vec = np.zeros((len(y_str_list), self.tokenizer.vocab_size), dtype=np.bool_)
        self.tokenizer.y_encode_into(y_vec, y_str_list)
        return y_vec

    def prepare_x_data(self, x_strs):
        return self.tokenizer.encode_many(x_strs)

    def prepare_data(self, data):
        x_str = [""]
        y_str_list = []
        for password in data:
            current_password = ""
            for char in password:
                current_password = current_password + char
                x_str.append(current_password[:])
                y_str_list.append(char)
            x_str.append("")
            y_str_list.append("\n")

        x_vec = self.prepare_x_data(x_str[:-1])
        y_vec = self.prepare_y_data(y_str_list)
        return x_vec, y_vec

    def get_batches(self, batch_size=128, is_train=True):
        data = self.train_passwords if is_train else self.test_passwords

        for i in range(0, len(data) - batch_size + 1, batch_size):
            batch = [pwd for pwd in data[i:i + batch_size]]
            x_vec, y_vec = self.prepare_data(batch)
            yield x_vec, y_vec

    def get_test_size(self):
        return len(self.test_passwords)

    def get_train_size(self):
        return len(self.train_passwords)

    def remove_padding(self, password):
        # bos eos and pad tokens are already removed
        return password

    def decode_password(self, password):
        # passwords are already decoded
        return password
