import os
import gzip
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import TrainingArguments
import time
from datetime import timedelta
import numpy as np
import random
import glob
from tqdm import trange
from transformers import Trainer

from script.test.model import Model
from models.PassGPT.passgpt_utils import *
from models.PassGPT.create_tokenizer import create_tokenizer, load_tokenizer, create_dataset

class PassGPT(Model):
    def __init__(self, settings):
        super().__init__(settings)

    def prepare_data(self):
        self.eval_args = self.params['eval']
        seed = self.params['config']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        tokenizer_path = self.params['config']['tokenizer_path']
        tokenizer_path = os.path.join("models", "PassGPT", tokenizer_path, self.test_hash)
        training_args = self.params['train']

        if int(self.max_length) > 10:
            if int(self.max_length) < 13:
                training_args['per_device_train_batch_size'] = 1024
            else:
                training_args['per_device_train_batch_size'] = 256

        training_args["output_dir"] = os.path.join(training_args["output_dir"], self.train_hash)
        if not os.path.exists(training_args["output_dir"]):
            os.makedirs(training_args["output_dir"], exist_ok=True)

        self.TOKENIZER_MAX_LEN = int(self.max_length) + 2

        train_dataset = create_dataset(self.train_path, "train")
        test_dataset = create_dataset(self.test_path, "test")

        self.data = (tokenizer_path, train_dataset, test_dataset)

    def prepare_model(self):
        self.tokenizer_path = self.data[0]

        self.eval_args = self.params['eval']
        self.params['max_length'] = self.max_length
        self.params['TOKENIZER_MAX_LEN'] = self.TOKENIZER_MAX_LEN

        seed = self.params['config']['seed']
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.checkpoint_dir = os.path.join("checkpoints", "PassGPT", self.train_hash)
        if not os.path.exists(os.path.join(os.getcwd(), self.checkpoint_dir)):
            os.makedirs(os.path.join(os.getcwd(), self.checkpoint_dir), exist_ok=True)

        self.eval_args['n_samples'] = self.n_samples

        self.save_guesses = self.params['eval']['save_guesses']
        self.save_matches = self.params['eval']['save_matches']
        return self.eval_args, self.checkpoint_dir, self.save_guesses, self.save_matches


    def save(self):
        path_to_gpt_model = os.path.join(self.checkpoint_dir, "checkpoint.pt")
        self.model.save_pretrained(path_to_gpt_model)

    def load(self, fname):
        try:
            self.prepare_model()
            path_to_gpt_model = os.path.join(self.checkpoint_dir, "checkpoint.pt")
            self.model = GPT2LMHeadModel.from_pretrained(path_to_gpt_model).eval().to(self.device)
            return 1
        except Exception as e:
            print(f"Exception: {e}")
            return 0

    def build_model(self, tokenizer, model_args):
        config = GPT2Config(
            vocab_size=tokenizer.vocab_size,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            **model_args
        )

        model = GPT2LMHeadModel(config)
        print("Model initialized with {} parameters".format(sum(t.numel() for t in model.parameters())))

        return model

    def train(self):
        self.tokenizer_path, train_dataset, test_dataset = self.data

        if not os.path.exists(self.tokenizer_path):
            self.tokenizer_path = create_tokenizer(train_dataset, self.tokenizer_path, self.max_length)

        self.tokenizer = load_tokenizer(self.tokenizer_path, self.TOKENIZER_MAX_LEN)

        training_args = self.params['train']
        model_args = self.params['model_args']
        config_args = self.params['config']
        max_length = self.params['max_length']
        TOKENIZER_MAX_LEN = self.params['TOKENIZER_MAX_LEN']
        output_dir = training_args["output_dir"]
        seed = config_args['seed']

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

        self.model = self.build_model(self.tokenizer, model_args).to(self.device)

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
        self.save()

    def save_passwords(self, passwords, output_path, mode):
        dir = os.path.join(output_path, str(mode))
        if not os.path.exists(dir):
            os.makedirs(dir, exist_ok=True)
        else:
            for filename in glob.glob(os.path.join(dir, "*.gz")):
                os.remove(filename)

        with gzip.open(os.path.join(dir, mode + ".gz"), 'wt') as f:
            for psw in passwords:
                f.write("{}\n".format(psw))

    def evaluate(self, eval_args, save_matches, save_guesses):
        self.tokenizer_path, train_dataset, test_dataset = self.data

        if not os.path.exists(self.tokenizer_path):
            self.tokenizer_path = create_tokenizer(train_dataset, self.tokenizer_path, self.max_length)

        self.tokenizer = load_tokenizer(self.tokenizer_path, self.TOKENIZER_MAX_LEN)

        output_path = self.output_path
        max_length = self.params['max_length']

        n_samples = eval_args['n_samples']
        evaluation_batch_size = eval_args["evaluation_batch_size"]
        num_beams = eval_args['num_beams']
        temperature = eval_args['temperature']
        top_p = eval_args['top_p']
        top_k = eval_args['top_k']
        seed = eval_args["seed"]

        # Init random seeds
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        assert n_samples % evaluation_batch_size == 0, "Number of passwords to generate should be divisible by batch size"

        # Passwords generation
        generations = []
        print("bos_token", self.tokenizer.bos_token_id)

        for i in trange(int(n_samples / evaluation_batch_size)):
            # Set seed for reproducibility
            torch.manual_seed(seed + i)

            with torch.no_grad():
                # Generate tokens sampling from the distribution of codebook indices
                g = self.model.generate(torch.tensor([[self.tokenizer.bos_token_id]]).cuda(device=self.device), do_sample=True,
                                   max_length=max_length + 2, pad_token_id=self.tokenizer.pad_token_id,
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
            self.save_passwords(generations, output_path, mode="guesses")

        eval_passwords = set(test_dataset["test"]["text"])
        matches = eval_passwords.intersection(set(generations))

        if save_matches:
            self.save_passwords(list(matches), output_path, mode="matches")

        matches = len(matches)
        test_size = len(eval_passwords)
        match_percentage = f'{(matches / test_size) * 100:.2f}%'
        print(f'{matches} matches found ({match_percentage} of test set).')
        return matches, match_percentage, test_size
        