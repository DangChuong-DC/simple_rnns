import pandas as pd
import re
import string
from collections import Counter
import json

from datasets import load_dataset
from torch.utils.data import DataLoader
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')


def clean_text(text: str) -> list[str]:
    text = re.sub(r'<[^>]+>', ' ', text) # strip HTML
    text = text.lower() # convert to lowercase
    text = re.sub(r'https?://\S+|www\.\S+', ' ', text)  # strip URLs
    # text = text.translate(str.maketrans('', '', string.punctuation)) # remove punctuation
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


def main():
    # # Load a Hugging Face dataset
    # dataset = load_dataset("stanfordnlp/imdb", split="train")
    # # Set the format to PyTorch
    # dataset.set_format(type="torch", columns=["text", "label"])

    # # Create a PyTorch DataLoader
    # dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # for batch in dataloader:
    #     texts = batch["text"]
    #     labels = batch["label"]

    #     print(texts)
    #     print(word_tokenize(texts[0]))
    #     print(wordpunct_tokenize(texts[0]))
    #     print(type(texts[0]))

    #     break

    # Load a CSV file into a DataFrame
    df = pd.read_csv(
        "/DATA01/dc/datasets/imdb_movie_reviews/IMDB-Dataset.csv",
        names=['text', 'label'],
    )

    df['text'] = df['text'].apply(lambda x: clean_text(x))
    print("Finished cleaning text!!!")

    word_counter = Counter(word for phrase in df['text'] for word in phrase)

    REQUIRED_FREQ = 999
    filtered_words = [word for word, count in word_counter.items() if count >= REQUIRED_FREQ]
    vocabulary = {word: idx + 1 for idx, word in enumerate(filtered_words)}
    vocabulary["<unk>"] = 0

    print(vocabulary)
    print(f"Vocabulary size: {len(vocabulary)}")

    # saving vocabulary to a file
    save_path = "/home/dc/self_studies/simple_rnns/.checkpoints"
    with open(f"{save_path}/vocabulary.json", "w") as f:
        json.dump(vocabulary, f)
    print(f"Vocabulary saved to {save_path}/vocabulary.json")


if __name__ == "__main__":
    main()
