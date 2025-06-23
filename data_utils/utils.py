from typing import Sequence, Any
import re

import torch
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

from models.tokenizer import WordTokenizer

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
