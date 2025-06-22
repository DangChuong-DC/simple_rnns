import nltk
nltk.download('punkt')
from nltk.stem import PorterStemmer, SnowballStemmer, LancasterStemmer


def main():
    words1 = ["python", "pythoner", "pythoning", "pythonly", "pythons"]
    words2 = ["studies", "studying", "studied", "study", "studious"]
    words3 = ["happily", "happy", "happier", "happiest"]

    ps = PorterStemmer()
    ps_words1 = [ps.stem(w) for w in words1]
    ps_words2 = [ps.stem(w) for w in words2]
    ps_words3 = [ps.stem(w) for w in words3]
    print("Porter Stemmer:")
    print("Words1:", words1, ps_words1)
    print("Words2:", words2, ps_words2)
    print("Words3:", words3, ps_words3)

    ss = SnowballStemmer("english")
    ss_words1 = [ss.stem(w) for w in words1]
    ss_words2 = [ss.stem(w) for w in words2]
    ss_words3 = [ss.stem(w) for w in words3]
    print("\nSnowball Stemmer:")
    print("Words1:", words1, ss_words1)
    print("Words2:", words2, ss_words2)
    print("Words3:", words3, ss_words3)

    ls = LancasterStemmer()
    ls_words1 = [ls.stem(w) for w in words1]
    ls_words2 = [ls.stem(w) for w in words2]
    ls_words3 = [ls.stem(w) for w in words3]
    print("\nLancaster Stemmer:")
    print("Words1:", words1, ls_words1)
    print("Words2:", words2, ls_words2)
    print("Words3:", words3, ls_words3)

if __name__ == "__main__":
    main()
