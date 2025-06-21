import argparse
import sys
import random
import pickle
import os
from pathlib import Path
from tokenizers import ByteLevelBPETokenizer, trainers
from transformers import RobertaTokenizerFast, AutoTokenizer
from pathlib import Path
from datasets import DatasetDict, Dataset

#### It is strongly recommended to use this script with a training set including unique passwords, not frequencies.

class PassTokenizer(ByteLevelBPETokenizer):
    """ByteLevelBPETokenizer
    Represents a Byte-level BPE as introduced by OpenAI with their GPT-2 model
    """

    def train_from_iterator(
            self,
            iterator,
            vocab_size: int = 30000,
            min_frequency: int = 2,
            show_progress: bool = True,
            special_tokens=[],
            length=None,
    ):
        """Train the model using the given iterator"""

        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=min_frequency,
            show_progress=show_progress,
            special_tokens=special_tokens,
            initial_alphabet=[],
        )
        self._tokenizer.train_from_iterator(
            iterator,
            trainer=trainer,
            length=length,
        )

def create_dataset(data, mode=""):
    dataset = {"text": data}
    dataset = Dataset.from_dict(dataset)
    dataset = DatasetDict({mode: dataset})
    return dataset


def create_tokenizer(train_data, output_path, max_length):
    print("===> Reading passwords")

    # Filter printable passwords
    ascii_printable = []
    train_data = train_data["train"]
    for k in train_data:
        v = k["text"]
        if all(32 < ord(c) < 128 for c in v) and (len(v) <= int(max_length)):
            ascii_printable.append(v)

    # Log information about your data
    all_chars = ''.join(ascii_printable)  # concatenate all strings into a single string
    unique_chars = set(all_chars)
    count = len(unique_chars)
    print(f"The number of distinct letters in all strings is {count}")

    # Customize training
    special_tokens = [
        "<s>",
        "<pad>",
        "</s>",
        "<unk>",
        "<mask>",
    ]

    # Create BPE tokenizer
    print("===> Training tokenizer")
    tokenizer = PassTokenizer()

    # Customize training
    tokenizer.train_from_iterator(ascii_printable, vocab_size=count + len(special_tokens), min_frequency=1,
                                  special_tokens=special_tokens)

    print("===> Tokenizer trained with vocabulary")
    vocab = tokenizer.get_vocab()
    print(sorted(vocab, key=lambda x: vocab[x]))

    Path(output_path).mkdir(parents=True, exist_ok=True)

    # Export
    tokenizer.save_model(output_path)
    print("===> Tokenizer exported succesfully")

    return output_path


def load_tokenizer(tokenizer_path, TOKENIZER_MAX_LEN):
    print("[I] - Loading tokenizer")

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    except:

        tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_path,
                                                         max_len=TOKENIZER_MAX_LEN,
                                                         mask_token="<mask>",
                                                         unk_token="<unk>",
                                                         pad_token="<pad>",
                                                         clean_up_tokenization_spaces=True,
                                                         )
        tokenizer.save_pretrained(tokenizer_path)
    return tokenizer
