from functools import partial

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset

from models.tokenizer import WordTokenizer
from .utils import collate_fn


def get_imdb_loader(
    split: str, 
    tokenizer: WordTokenizer,
    batch_size: int = 64,
) -> DataLoader:
    dataset = load_dataset("stanfordnlp/imdb", split=split)

    my_collate_fn = partial(collate_fn, tokenizer=tokenizer)
    loader = DataLoader(
        dataset, batch_size=batch_size, num_workers=8, shuffle=True, collate_fn=my_collate_fn
    )
    return loader
