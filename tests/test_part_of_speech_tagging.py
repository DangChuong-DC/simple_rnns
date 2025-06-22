import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger


def main():
    sentence = "The children studies are better than the studies done before. He was running, swimming, and eating. Sony gets a contract with Jonhny Depth."

    tokens = nltk.word_tokenize(sentence) # Tokenize the sentence into words
    pos_tags = nltk.pos_tag(tokens)  # Get the part-of-speech tags for each token

    print("Tokens:", tokens)
    print(type(tokens[0]))

    print("---")
    print("POS Tags:", pos_tags)
    print(type(pos_tags[0]))


    # Initialize the Stanford POS Tagger
    # Make sure to set the correct path to your Stanford POS Tagger model and jar files
    model_path = "/home/dc/nltk_data/taggers/stanford-postagger-full-2015-04-20/models/english-bidirectional-distsim.tagger"  # Path to the POS tagger model
    jar_path = "/home/dc/nltk_data/taggers/stanford-postagger-full-2015-04-20/stanford-postagger.jar"
    sfd_pos_tag = StanfordPOSTagger(model_path, jar_path)
    sfd_pos_tags = sfd_pos_tag.tag(tokens)  # Get the POS tags using Stanford POS Tagger
    print("---")
    print("Stanford POS Tags:", sfd_pos_tags)
    print(type(sfd_pos_tags[0]))

    # Example of using the NLTK NER tagger
    model_path = "/home/dc/nltk_data/taggers/stanford-ner-2015-04-20/classifiers/english.all.3class.distsim.crf.ser.gz"
    jar_path = "/home/dc/nltk_data/taggers/stanford-ner-2015-04-20/stanford-ner.jar"
    sfd_ner_tag = StanfordNERTagger(model_path, jar_path)
    ner_tags = sfd_ner_tag.tag(tokens)
    print("---")
    print("Stanford NER Tags:", ner_tags)
    print(type(ner_tags[0]))


if __name__ == "__main__":
    main()
