import nltk
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')

import spacy


def main():
    lemmatizer = WordNetLemmatizer()

    words1 = ["python", "pythoner", "pythoning", "pythonly", "pythons"]
    words2 = ["studies", "studying", "studied", "study", "studious"]
    words3 = ["happily", "happy", "happier", "happiest"]

    lemmatized_words1 = [lemmatizer.lemmatize(w, pos='n') for w in words1]
    lemmatized_words2 = [lemmatizer.lemmatize(w, pos='v') for w in words2]
    lemmatized_words3 = [lemmatizer.lemmatize(w, pos='a') for w in words3]  # 'a' for adjectives
    print("Lemmatization:")
    print("Words1:", words1, lemmatized_words1)
    print("Words2:", words2, lemmatized_words2)
    print("Words3:", words3, lemmatized_words3)


    # Using spaCy for lemmatization
    nlp = spacy.load("en_core_web_sm")

    doc = nlp("Apples and oranges are similar. Boots and hippos aren't.")
    print(doc.text)
    
    lemmas = [token.lemma_ for token in doc]
    print(" ".join(lemmas))
        

if __name__ == "__main__":
    main()
