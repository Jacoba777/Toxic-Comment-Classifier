import re
import nltk
from nltk.stem import PorterStemmer
from nltk.tokenize import RegexpTokenizer
from datetime import datetime

nltk.download('punkt')

tokenizer = RegexpTokenizer(r'\w+')
ps = PorterStemmer()


# Represents a single document from the corpus.
class Document:
    id: str
    text: str

    toxic: bool
    severe_toxic: bool
    obscene: bool
    threat: bool
    insult: bool
    identity_hate: bool

    dictionary: set
    dictionary_bigram: set
    dictionary_trigram: set

    def __init__(self, id, text,
                 toxic=False, severe_toxic=False, obscene=False, threat=False, insult=False, identity_hate=False):
        self.id = id
        self.text = text

        self.toxic = toxic
        self.severe_toxic = severe_toxic
        self.obscene = obscene
        self.threat = threat
        self.insult = insult
        self.identity_hate = identity_hate

        self.dictionary = set()
        self.dictionary_bigram = set()
        self.dictionary_trigram = set()
        self.freq_term = {}

        self.create_dictionaries()

    # not used
    def get_term_frequency(self, word):
        return self.freq_term[word]

    # Creates dictionaries for each unique unigram, bigram, and trigram encountered in the document's text.
    def create_dictionaries(self):
        words = tokenizer.tokenize(self.text)

        for w in range(len(words)):
            word = words[w]
            if len(word) > 900:
                word = word[:900]
            word = ps.stem(word)
            self.dictionary.add(word)

            if w > 0:
                word_prev = ps.stem(words[w-1])
                self.dictionary_bigram.add('{} {}'.format(word_prev, word))

            if w > 1:
                word_prev = ps.stem(words[w - 1])
                word_prev_prev = ps.stem(words[w-2])
                self.dictionary_trigram.add('{} {} {}'.format(word_prev_prev, word_prev, word))

    # Returns true if the document belongs to at least one of the six toxic subcategories.
    def is_toxic(self):
        return self.toxic or self.severe_toxic or self.obscene or self.threat or self.insult or self.identity_hate


# Reads the documents from the corpus, and generates Document objects to represent each of them.
# lim can be used to limit the reader to only use the first [lim] documents (if lim is positive)
# or the last [lim] documents (if lim is negative)
def create_docs_from_corpus(corpus: str, lim=None):
    docs = []

    corpus = corpus.replace('""""', "''")
    corpus = corpus.replace('"""', '"')
    comments = re.split(r'("[^"]*","[^"]*",\d,\d,\d,\d,\d,\d)\n', corpus)

    print('Found {} documents'.format(len(comments)))

    last = False
    if lim is not None:
        if lim < 0:
            lim = -lim
            last = True

        if len(comments) > lim:
            if not last:
                comments = comments[:2*lim]
                print('Limiting to only use the first {} documents'.format(lim))
            else:
                comments = comments[-2*lim:]
                print('Limiting to only use the last {} documents'.format(lim))

    for i in range(len(comments)):
        if i % 40000 == 0:
            print(datetime.now(), '| Creating document {} of {}'.format(i // 2, lim))
        comment = comments[i]
        data = re.split(r'"([^"]*)","([^"]*)",(\d),(\d),(\d),(\d),(\d),(\d)', comment)
        if len(data) > 1:
            data = data[1:-1]  # Remove the empty data in the first and last spots in the list
            doc = create_doc_from_data(data)
            docs.append(doc)

    return docs


# Converts a 1 or 0 to True or False, respectively.
def flag2bool(flag: str):
    return flag == '1'


# Creates a Document object from a single line from the corpus.
def create_doc_from_data(data: list):
    id = data[0]
    text = data[1].replace("'", '').replace('"', '')
    f1 = flag2bool(data[2])
    f2 = flag2bool(data[3])
    f3 = flag2bool(data[4])
    f4 = flag2bool(data[5])
    f5 = flag2bool(data[6])
    f6 = flag2bool(data[7])

    return Document(id, text, f1, f2, f3, f4, f5, f6)
