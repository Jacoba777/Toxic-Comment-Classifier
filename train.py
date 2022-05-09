import operator
from corpus_utils import *

# The location of the dataset to train the model off of
input_file = './dataset.txt'

# Restrict the training dataset to be this size.
# A positive number will use that many documents reading from the top of the file.
# A negative number will use that many documents reading from the bottom of the file.
train_dataset_size = 474807

# The folder to write the unigram, bigram, trigram, and count dictionaries to.
output_directory = './dictionaries'

#
dict_types = ['all', 'nontoxic', 'all_toxic', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# Represents a single dictionary
class Dictionary:
    gram: int  # 1 for unigram, 2 for bigram ... etc
    name: str  # The name of the dictionary, corresponds to an element of dict_types
    dictionary: dict  # dictionary that maps an n-gram to a count

    def __init__(self, gram: int, name: str):
        self.gram = gram
        self.name = name
        self.dictionary = dict()

    # Returns the string representation of the n-gram.
    def gram_name(self):
        if self.gram == 1:
            return 'uni'
        elif self.gram == 2:
            return 'bi'
        elif self.gram == 3:
            return 'tri'
        else:
            return str(self.gram)

    # Returns the file location that this dictionary should be saved to based on the n-gram and dictionary type.
    def get_dict_filename(self):
        return '{}/{}gram_{}.txt'.format(output_directory, self.gram_name(), self.name)


# Increments the count of an n-gram in a dictionary by 1. Null-safe.
def inc_word_freq(dict: Dictionary, word, conditional=True):
    if conditional:
        if dict.dictionary.__contains__(word):
            dict.dictionary[word] += 1
        else:
            dict.dictionary[word] = 1


# Writes the contents of the dictionaries to the output directory
def write_dicts_to_file(dicts: dict):
    for dict_row in dicts.values():
        for dict_to_sort in dict_row.values():
            file = dict_to_sort.get_dict_filename()
            print('Writing to', file)

            # Sorts by n-gram frequency, with most frequent first
            file_txt = ''
            dict_sorted = dict(sorted(dict_to_sort.dictionary.items(), key=operator.itemgetter(1), reverse=True))

            i = 0
            for word in dict_sorted.keys():
                if i % 10000:
                    print('Writing word {}/{}'.format(i, len(dict_sorted)))
                file_txt += '{} {}\n'.format(word, dict_sorted[word])
                i += 1
            file_txt = file_txt.rstrip('\n')

            f = open(file, 'w', encoding="UTF-8")
            f.write(file_txt)
            f.close()


# Writes the document counts to the output directory
def write_counts_to_file(counts):
    counts_txt = ''
    for count_name in counts.keys():
        counts_txt += '{} {}\n'.format(count_name, counts[count_name])
    counts_txt = counts_txt.rstrip()
    filename = output_directory + '/counts.txt'
    f = open(filename, 'w')
    f.write(counts_txt)
    f.close()


# Generates all of the necessary dictionaries from the dataset and writes them to file
def create_dictionaries(docs: list):
    dicts = {1: {}, 2: {}, 3: {}}
    counts = {}

    for dict_name in dict_types:
        dicts[1][dict_name] = Dictionary(1, dict_name)
        dicts[2][dict_name] = Dictionary(2, dict_name)
        dicts[3][dict_name] = Dictionary(3, dict_name)
        counts[dict_name] = 0

    for i in range(len(docs)):
        if i % 1000 == 0:
            print(datetime.now(), '| Processing Document', i, '/', len(docs))
        doc = docs[i]

        # The conditional that must evaluate to true in order to increment a counter
        conditionals = {'all': True,
                        'nontoxic': not doc.is_toxic(),
                        'all_toxic': doc.is_toxic(),
                        'toxic': doc.toxic,
                        'severe_toxic': doc.severe_toxic,
                        'obscene': doc.obscene,
                        'threat': doc.threat,
                        'insult': doc.insult,
                        'identity_hate': doc.identity_hate}

        # Increment each n-gram counter in the doc to the dictionaries if the document is of that type
        # Also increment the document counter accordingly
        for dict_name in dict_types:
            if conditionals[dict_name]:
                counts[dict_name] += 1
                for word in doc.dictionary:
                    inc_word_freq(dicts[1][dict_name], word)
                for bigram in doc.dictionary_bigram:
                    inc_word_freq(dicts[2][dict_name], bigram)
                for trigram in doc.dictionary_trigram:
                    inc_word_freq(dicts[3][dict_name], trigram)

    write_counts_to_file(counts)
    write_dicts_to_file(dicts)


# Main process function; Reads the input, generates documents from it, and creates the dictionaries.
def process_file():
    f = open(input_file, encoding="UTF-8")
    corpus = f.read()
    f.close()

    docs = create_docs_from_corpus(corpus, train_dataset_size)
    create_dictionaries(docs)


process_file()
