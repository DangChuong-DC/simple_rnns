from typing import Sequence, Any
from functools import partial
from tqdm import tqdm
import re

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, SGD, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from datasets import load_dataset
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from models.simp_rnn import SimpClassificationRNN, TorchClassificationRNN
from models.simp_gru import SimpleClassifierGRU
from tokenizer import WordTokenizer

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


def clean_text(text: str) -> list[str]:
    """
    Clean and tokenize the input text.
    """
    text = re.sub(r'<[^>]+>', ' ', text) # strip HTML
    text = text.lower() # convert to lowercase
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # strip URLs
    text = re.sub(r'[^a-z0-9\s]', ' ', text)  # remove punctuation/special chars
    text = re.sub(r'\s+', ' ', text).strip()

    # Tokenize into words
    # tokens = word_tokenize(text)
    tokens = wordpunct_tokenize(text)  # Alternative tokenization

    # # Remove stopwords and very short tokens
    stops = set(stopwords.words('english'))
    tokens = [t for t in tokens if t not in stops and len(t) > 2]

    # # Lemmatize using WordNet
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(t, pos=wordnet.NOUN) for t in tokens]
    return tokens


def collate_fn(
    batch: Sequence[dict[str, Any]], tokenizer: WordTokenizer
) -> dict[str, torch.Tensor]:
    texts = [clean_text(item["text"]) for item in batch]
    max_length = max(len(text) for text in texts)
    # tokenize text and pad sequences to the maximum length
    input_ids = [tokenizer.encode(text) for text in texts]
    input_ids = [[tokenizer.get_padding_index()] * (max_length - len(ids)) + ids for ids in input_ids]
    input_ids = torch.tensor(input_ids, dtype=torch.long)

    labels = [item["label"] for item in batch]
    labels = torch.tensor(labels, dtype=torch.float)

    output = {
        "input_ids": input_ids,
        "labels": labels,
    }
    return output


def eval(
    epoch: int,
    model: nn.Module,
    val_dataloader: DataLoader,
    device: str,
    criterion: nn.Module,
) -> None:
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in tqdm(val_dataloader):
            seq_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            logits = model(seq_ids).squeeze(-1)  # (B,)

            loss = criterion(logits, labels)
            total_loss += loss.item()

            # Convert logits to probabilities (since using BCEWithLogitsLoss)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    
    avg_eval_loss = total_loss / len(val_dataloader)
    accuracy = correct / total if total > 0 else 0
    print(f">>> [Eval Epoch {epoch}] loss: {avg_eval_loss:.9f} | accuracy: {accuracy*100:.2f}% <<<")


def train(
    epoch: int,
    model: nn.Module, 
    train_dataloader: DataLoader, 
    optimizer: Optimizer,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: str, 
    criterion: nn.Module,
    log_iter: int = 5000,
) -> None:
    model.train()
    total_loss = 0.0

    for i, batch in enumerate(tqdm(train_dataloader)):
        # move data to right device
        seq_ids = batch["input_ids"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()

        logits = model(seq_ids).squeeze(-1)  # (B,)
        assert torch.isfinite(logits).all(), "NaN in logits!"

        loss = criterion(logits, labels)
        assert torch.isfinite(loss).all(), "loss became NaN!"
        loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        lr_scheduler.step()

        total_loss += loss.item()
        if i % log_iter == 0:
            print(f"[Iter {i}] --- loss: {loss.item()}")
    
    avg_train_loss = total_loss / len(train_dataloader)
    print(f"> 🎓 [Epoch {epoch}] train loss: {avg_train_loss:.9f} <")


def main():
    # ___ Training Config ___
    num_epoch = 9
    batch_size = 64
    max_learning_rate = 1e-3
    min_learning_rate = 5e-6

    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

    # ___ Model Config ___
    model_dimension = 512
    RNN_hidden_dimension = 1024
    RNN_num_layer = 1
    num_class = 1


    train_dataset = load_dataset("stanfordnlp/imdb", split="train")
    val_dataset = load_dataset("stanfordnlp/imdb", split="test")

    tokenizer = WordTokenizer("/home/dc/self_studies/simple_rnns/.checkpoints/vocabulary.json")
    my_collate_fn = partial(collate_fn, tokenizer=tokenizer)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, num_workers=8, shuffle=True, collate_fn=my_collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, num_workers=4, shuffle=False, collate_fn=my_collate_fn
    )

    rnn_model = SimpleClassifierGRU(
        len(tokenizer), model_dimension, RNN_hidden_dimension, 
        num_layer=RNN_num_layer, num_class=num_class,
    )
    rnn_model.to(device=device)

    optimizer = Adam(
        rnn_model.parameters(), lr=max_learning_rate, betas=(0.9, 0.95)
    )
    total_steps = num_epoch * len(train_loader)
    lr_scheduler = CosineAnnealingLR(optimizer, total_steps, min_learning_rate)

    criterion = nn.BCEWithLogitsLoss()

    for e in range(num_epoch):
        eval(e, rnn_model, val_loader, device, criterion)
        train(e, rnn_model, train_loader, optimizer, lr_scheduler, device, criterion)
    
    # Final evaluation
    print("Final evaluation:")
    eval(e + 1, rnn_model, val_loader, device, criterion)


if __name__ == "__main__":
    main()
