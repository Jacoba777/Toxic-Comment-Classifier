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

        self.create_dictionaries(stem_words=False)

    # not used
    def get_term_frequency(self, word):
        return self.freq_term[word]

    # Creates dictionaries for each unique unigram, bigram, and trigram encountered in the document's text.
    def create_dictionaries(self, stem_words=True):
        words = tokenizer.tokenize(self.text)

        for w in range(len(words)):
            word = words[w].strip().upper()
            if len(word) > 900:
                word = word[:900]
            if stem_words:
                word = ps.stem(word)
            self.dictionary.add(word)

            if w > 0:
                word_prev = words[w - 1].strip().upper()
                if stem_words:
                    word_prev = ps.stem(word_prev)
                self.dictionary_bigram.add('{} {}'.format(word_prev, word))

            if w > 1:
                word_prev = words[w - 1].strip().upper()
                word_prev_prev = words[w - 2].strip().upper()
                if stem_words:
                    word_prev = ps.stem(word_prev)
                    word_prev_prev = ps.stem(word_prev_prev)
                self.dictionary_trigram.add('{} {} {}'.format(word_prev_prev, word_prev, word))

    # Returns true if the document belongs to at least one of the six toxic subcategories.
    def is_toxic(self):
        return self.toxic or self.severe_toxic or self.obscene or self.threat or self.insult or self.identity_hate


# Equivalent to the Dictionary in the train.py file; Represents a single dictionary file.
class NgramFile:
    gram: int  # 1 for unigram, 2 for bigram ... etc
    classification: str  # The name of the dictionary, corresponds to an element of dict_types
    data: dict  # dictionary that maps an n-gram to a count

    def __init__(self, filename: str, dictionary_directory: str):
        self.classification = filename.replace('unigram_', '').replace('bigram_', '').replace('trigram_', '')\
            .replace('.txt', '')
        self.gram = 0
        self.data = {}

        f = open(dictionary_directory + '/' + filename, encoding="UTF-8")
        data_raw = f.read()
        f.close()

        if len(data_raw) > 0:
            self.gram = len(data_raw.split('\n')[0].split(' ')) - 1
            self.set_data(data_raw)

    # Reads the text from the file, and generates its own internal data accordingly.
    def set_data(self, data: str):
        ngram_rows = data.split('\n')
        for ngram_row in ngram_rows:
            freq = int(ngram_row.split(' ')[-1])
            ngram = ngram_row.replace(' {}'.format(freq), '')
            self.data[ngram] = freq

    # Returns the number of documents that had this n-gram, and belonged to this object's classification type.
    def get_freq(self, ngram: str):
        if self.data.__contains__(ngram):
            return self.data[ngram]
        return 0


# A data structure that stores all of the dictionaries from file
class NgramData:
    ngram_files: list

    def __init__(self):
        self.ngram_files = []

    # Adds a single Ngram file to the data structure
    def add_ngram_file(self, ngram_file: NgramFile):
        self.ngram_files.append(ngram_file)

    def get_ngram_file(self, gram: int, classification: str):
        for ngram_file in self.ngram_files:
            if ngram_file.gram == gram and ngram_file.classification == classification:
                return ngram_file

    # Finds the relevant dictionary given the n-gram's length and the desired classification, and returns the frequency.
    def get_ngram_freq(self, ngram: str, classification: str):
        gram = ngram.count(' ') + 1

        ngram_file = self.get_ngram_file(gram, classification)
        if ngram_file:
            return ngram_file.get_freq(ngram)
        else:
            return 0

    # Returns a list of tuples, where each tuple is a frequency and
    def get_ngrams_freqs_with_prefix(self, prefix: str, classification: str):
        gram = prefix.count(' ') + 2
        ngrams = []
        weights = []

        ngram_file = self.get_ngram_file(gram, classification)

        for ngram in ngram_file.data.keys():
            if ngram.startswith(prefix):
                ngrams.append(ngram)
                weights.append(ngram_file.data[ngram])
        return ngrams, weights


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
