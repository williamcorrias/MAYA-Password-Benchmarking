import os
import gzip
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import TrainingArguments
import time
from datetime import timedelta
import numpy as np
import random
from tqdm import trange
from transformers import Trainer

from script.test.model import Model
from models.PassGPT.passgpt_utils import *
from models.PassGPT.create_tokenizer import create_tokenizer, load_tokenizer, create_dataset

class PassGPT(Model):
    def __init__(self, settings):
        self.model = None
        self.tokenizer = None
        super().__init__(settings)

    def prepare_data(self, train_passwords, test_passwords, max_length):
        tokenizer_path = self.params['config']['tokenizer_path']
        tokenizer_path = os.path.join("models", "PassGPT", tokenizer_path, self.test_hash)
        train_dataset = create_dataset(train_passwords, "train")
        test_dataset = create_dataset(test_passwords, "test")
        data = (tokenizer_path, train_dataset, test_dataset)
        return data

    def save(self, file, mid=False):
        self.model.save_pretrained(file)

    def load(self, file_to_load):
        try:
            self.model = GPT2LMHeadModel.from_pretrained(file_to_load).eval().to(self.device)
            return 1
        except Exception as e:
            print(f"Exception: {e}")
            return 0

    def init_tokenizer(self):
        TOKENIZER_MAX_LEN = int(self.max_length) + 2
        tokenizer_path, train_dataset, test_dataset = self.data
        if not os.path.exists(tokenizer_path):
            tokenizer_path = create_tokenizer(train_dataset, tokenizer_path, self.max_length)
        self.tokenizer = load_tokenizer(tokenizer_path, TOKENIZER_MAX_LEN)

    def init_model(self, model_args):
        config = GPT2Config(
            vocab_size=self.tokenizer.vocab_size,
            bos_token_id=self.tokenizer.bos_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **model_args
        )

        self.model = GPT2LMHeadModel(config).to(self.device)
        print("Model initialized with {} parameters".format(sum(t.numel() for t in self.model.parameters())))

    def train(self):
        tokenizer_path, train_dataset, test_dataset = self.data

        training_args = self.params['train']
        training_args['output_dir'] = self.path_to_checkpoint_dir
        model_args = self.params['model_args']
        config_args = self.params['config']
        max_length = int(self.max_length)
        TOKENIZER_MAX_LEN = self.max_length + 2
        seed = config_args['seed']

        self.init_tokenizer()

        def preprocess_function(entries):
            """
            This function tokenizes a list of passwords. It appends the end of password token to each of them before processing.
            """
            to_tokenize = ['<s>' + p[:int(max_length)] + '</s>' for p in entries['text']]
            return self.tokenizer(to_tokenize,
                             truncation=True,
                             padding="max_length",
                             max_length=TOKENIZER_MAX_LEN,
                             add_special_tokens=False,
                             return_special_tokens_mask=False)

        print("[I] - Processing data")
        tokenized_datasets = train_dataset.map(preprocess_function, batched=True,
                                         remove_columns=train_dataset["train"].column_names)
        tokenized_datasets = tokenized_datasets.shuffle(seed=seed)
        tokenized_datasets.set_format(type="torch")

        print("[I] - Initializing model")
        self.init_model(model_args)

        print("[I] - Preparing training")
        # Define the data collator. In charge of hiding tokens to be predicted.
        data_collator = PasswordDataCollator(tokenizer=self.tokenizer, mlm=False)

        train_args = TrainingArguments(**training_args)

        trainer = Trainer(
            model=self.model,
            data_collator=data_collator,
            args=train_args,
            train_dataset=tokenized_datasets["train"]
        )

        print("[I] - Launching training")
        start = time.time()
        trainer.train()
        end = time.time()

        print("[T] - Training completed after {}. Storing last version.".format(str(timedelta(seconds=end - start))))
        path = os.path.join(self.path_to_checkpoint_dir, self.checkpoint_name)
        self.save(path)

    def write_to_file(self, file, data):
        with gzip.open(file, 'wt') as f:
            for psw in data:
                f.write("{}\n".format(psw))

    def evaluate(self, n_samples, validation_mode=False):
        tokenizer_path, train_dataset, test_dataset = self.data
        TOKENIZER_MAX_LEN = int(self.max_length) + 2

        if not os.path.exists(tokenizer_path):
            tokenizer_path = create_tokenizer(train_dataset, tokenizer_path, self.max_length)

        self.tokenizer = load_tokenizer(tokenizer_path, TOKENIZER_MAX_LEN)

        evaluation_batch_size = self.params["eval"]["evaluation_batch_size"]
        num_beams = self.params["eval"]['num_beams']
        temperature = self.params["eval"]['temperature']
        top_p = self.params["eval"]['top_p']
        top_k = self.params["eval"]['top_k']
        seed = self.params["eval"]["seed"]

        # Init random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        assert n_samples % evaluation_batch_size == 0, "Number of passwords to generate should be divisible by batch size"

        # Passwords generation
        generations = []
        print("bos_token", self.tokenizer.bos_token_id)

        save_guesses = self.save_guesses and not validation_mode
        save_matches = self.save_matches and not validation_mode

        for i in trange(int(n_samples / evaluation_batch_size)):
            # Set seed for reproducibility
            torch.manual_seed(seed + i)

            with torch.no_grad():
                # Generate tokens sampling from the distribution of codebook indices
                g = self.model.generate(torch.tensor([[self.tokenizer.bos_token_id]]).cuda(device=self.device), do_sample=True,
                                   max_length=TOKENIZER_MAX_LEN, pad_token_id=self.tokenizer.pad_token_id,
                                   bad_words_ids=[[self.tokenizer.bos_token_id]], num_return_sequences=evaluation_batch_size,
                                   num_beams=num_beams, top_p=top_p / 100, top_k=top_k,
                                   temperature=temperature)

                # Remove start of sentence token
                g = g[:, 1:]

            decoded = self.tokenizer.batch_decode(g.tolist())
            decoded_clean = [i.split("</s>")[0] for i in decoded]  # Get content before end of password token

            generations += decoded_clean

            del g
            del decoded
            del decoded_clean

        if save_guesses:
            self.write_to_file(self.path_to_guesses_file, generations)

        eval_passwords = set(test_dataset["test"]["text"])
        matches = eval_passwords.intersection(set(generations))

        if save_matches:
            self.write_to_file(self.path_to_matches_file, list(matches))

        matches = len(matches)
        test_size = len(eval_passwords)
        match_percentage = f'{(matches / test_size) * 100:.2f}%'
        print(f'{matches} matches found ({match_percentage} of test set).')
        return matches, match_percentage, test_size
        