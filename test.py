from corpus_utils import *
import os
import math

# The location of the dataset to test the model off of
input_file = './dataset.txt'

# Restrict the test dataset to be this size.
# A positive number will use that many documents reading from the top of the file.
# A negative number will use that many documents reading from the bottom of the file.
test_dataset_size = -100000

# The folder to read the unigram, bigram, trigram, and count dictionaries from.
dictionary_directory = './dictionaries'

# The file to write the test results file to.
output_file = './test_results.txt'

# The different categories to judge the model off of.
dict_types = ['nontoxic', 'all_toxic', 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']


# Equivalent to the Dictionary in the train.py file; Represents a single dictionary file.
class NgramFile:
    gram: int  # 1 for unigram, 2 for bigram ... etc
    classification: str  # The name of the dictionary, corresponds to an element of dict_types
    data: dict  # dictionary that maps an n-gram to a count

    def __init__(self, filename: str):
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

    # Finds the relevant dictionary given the n-gram's length and the desired classification, and returns the frequency.
    def get_ngram_freq(self, ngram: str, classification: str):
        gram = ngram.count(' ') + 1

        for ngram_file in self.ngram_files:
            if ngram_file.gram == gram and ngram_file.classification == classification:
                return ngram_file.get_freq(ngram)
        return 0


# A data structure that contains the predictions for a single document.
class ModelPrediction:
    nontoxic: float
    all_toxic: float

    toxic: float
    severe_toxic: float
    obscene: float
    threat: float
    insult: float
    identity_hate: float

    def __init__(self):
        self.nontoxic = 0.0
        self.all_toxic = 0.0
        self.toxic = 0.0
        self.severe_toxic = 0.0
        self.obscene = 0.0
        self.threat = 0.0
        self.insult = 0.0
        self.identity_hate = 0.0

    # The document is likely toxic if the model is more confident it is in the all_toxic classification than nontoxic
    def predict_all_toxic(self):
        return self.all_toxic > self.nontoxic

    def predict_toxic(self):
        return self.toxic > 0

    def predict_severe_toxic(self):
        return self.severe_toxic > 0

    def predict_obscene(self):
        return self.obscene > 0

    def predict_threat(self):
        return self.threat > 0

    def predict_insult(self):
        return self.insult > 0

    def predict_identity_hate(self):
        return self.identity_hate > 0

    def __getitem__(self, item):
        if item == 'nontoxic':
            return self.nontoxic
        elif item == 'all_toxic':
            return self.all_toxic
        elif item == 'toxic':
            return self.toxic
        elif item == 'severe_toxic':
            return self.severe_toxic
        elif item == 'obscene':
            return self.obscene
        elif item == 'threat':
            return self.threat
        elif item == 'insult':
            return self.insult
        elif item == 'identity_hate':
            return self.identity_hate

    def __setitem__(self, key, value):
        if key == 'nontoxic':
            self.nontoxic = value
        elif key == 'all_toxic':
            self.all_toxic = value
        elif key == 'toxic':
            self.toxic = value
        elif key == 'severe_toxic':
            self.severe_toxic = value
        elif key == 'obscene':
            self.obscene = value
        elif key == 'threat':
            self.threat = value
        elif key == 'insult':
            self.insult = value
        elif key == 'identity_hate':
            self.identity_hate = value


# Given a classification and a bunch of different predictive models (unigram, bigram, trigram),
# return the prediction that is the most confident for the classification.
def get_most_confident_prediction(dict_type: str, mps: list):
    predictions = []
    for mp in mps:
        predictions.append(mp[dict_type])

    p_max = max(predictions)
    p_min = min(predictions)

    if p_max > -p_min:
        return p_max
    else:
        return p_min


# Create's a user-friendly string to represent the ModelPrediction's guess for a document.
# Will be N if it believes the doc is nontoxic,
# and will be some combo of TSODIH for the different subcategories otherwise.
def get_prediction_str(mp: ModelPrediction):
    prediction_str = ''

    if mp.predict_all_toxic():
        if mp.predict_toxic(): prediction_str += 'T'
        if mp.predict_severe_toxic(): prediction_str += 'S'
        if mp.predict_obscene(): prediction_str += 'O'
        if mp.predict_threat(): prediction_str += 'D'
        if mp.predict_insult(): prediction_str += 'I'
        if mp.predict_identity_hate(): prediction_str += 'H'
    else:
        prediction_str = 'N'

    return prediction_str


# Create's a user-friendly string to represent the document's true classifications.
# Will be N if the doc is nontoxic, and will be some combo of TSODIH for the different subcategories otherwise.
def get_actual_str(doc: Document):
    actual_str = ''

    if doc.is_toxic():
        if doc.toxic: actual_str += 'T'
        if doc.severe_toxic: actual_str += 'S'
        if doc.obscene: actual_str += 'O'
        if doc.threat: actual_str += 'D'
        if doc.insult: actual_str += 'I'
        if doc.identity_hate: actual_str += 'H'
    else:
        actual_str = 'N'

    return actual_str


# Reads the document counts from file and stores/returns them in a dictionary.
def get_doc_counts():
    counts = {}

    f = open(dictionary_directory + '/counts.txt', encoding="UTF-8")
    counts_raw = f.read().split('\n')
    f.close()

    for count in counts_raw:
        data = count.split(' ')
        label = data[0]
        value = int(data[1])
        counts[label] = value

    return counts


# Creates the NgramData data structure from all of the dictionary files.
def build_ngram_data_from_dictionaries():
    ngram_data = NgramData()

    for file in os.listdir(dictionary_directory):
        if file != 'counts.txt':
            print('Creating Ngram data from', file)
            ngramfile = NgramFile(file)
            ngram_data.add_ngram_file(ngramfile)

    return ngram_data


# Predicts the document's classifications using the model.
# The raw score is in log-space, and approximates log( P(doc|C) / P(doc) ),
# where C indicates that the document is in the tested category.
# A positive raw score means the model believes the document is in the tested classification, and negative if not.
# To convert the raw score to a confidence proportion, calc 1-(1/(e^x+1)), where x is the raw score
def get_model_predictions(model: NgramData, doc_count: dict, doc: Document, gram: int):
    model_prediction = ModelPrediction()
    smoothing = 1 / doc_count['all']

    ngram_dict = doc.dictionary
    if gram == 2:
        ngram_dict = doc.dictionary_bigram
    elif gram == 3:
        ngram_dict = doc.dictionary_trigram

    for ngram in ngram_dict:
        p_ngram = model.get_ngram_freq(ngram, 'all') / doc_count['all']

        for dict_type in dict_types:
            p_ngram_given_dict_type = model.get_ngram_freq(ngram, dict_type) / doc_count[dict_type]
            model_prediction[dict_type] += math.log(p_ngram_given_dict_type + smoothing)
            model_prediction[dict_type] -= math.log(p_ngram + smoothing)

    return model_prediction


# Increments a metric by 1 depending on the relationship between the predicted and actual classification
def apply_metric(metrics: dict, dict_type: str, predicted: bool, actual: bool, id: str):
    if predicted == actual:
        if predicted:
            metrics[dict_type]['true_pos'] += 1
        else:
            metrics[dict_type]['true_neg'] += 1
    else:
        if predicted:
            metrics[dict_type]['false_pos'] += 1
            if dict_type != 'total':
                print('FALSE POS {} | {}'.format(id, dict_type))
        else:
            metrics[dict_type]['false_neg'] += 1
            if dict_type != 'total':
                print('FALSE NEG {} | {}'.format(id, dict_type))


# Test the model on the dataset and returns the results
def test_model(docs: list):
    model = build_ngram_data_from_dictionaries()

    doc_count = get_doc_counts()
    metric_categories = dict_types.copy()
    metric_categories.append('total')
    metric_categories.remove('nontoxic')

    # metrics is a 2D dictionary; The first dimension are the classifications,
    # and the 2nd dimension is the relevant metric for that classification
    # So metrics['insult']['true_pos'] will return the number of docs correctly predicted as an insult
    metrics = {}
    for cat in metric_categories:
        metrics[cat] = {}
        metrics[cat]['true_pos'] = 0
        metrics[cat]['true_neg'] = 0
        metrics[cat]['false_pos'] = 0
        metrics[cat]['false_neg'] = 0

    for i in range(len(docs)):
        doc = docs[i]

        predictions_unigram = get_model_predictions(model, doc_count, doc, 1)
        predictions_bigram = get_model_predictions(model, doc_count, doc, 2)
        predictions_trigram = get_model_predictions(model, doc_count, doc, 3)
        candidate_prediction_models = [predictions_unigram, predictions_bigram, predictions_trigram]

        predictions_final = ModelPrediction()

        for dict_type in dict_types:
            predictions_final[dict_type] = get_most_confident_prediction(dict_type, candidate_prediction_models)

        total_correct = get_prediction_str(predictions_final) == get_actual_str(doc)
        apply_metric(metrics, 'total', total_correct, True, doc.id)

        apply_metric(metrics, 'all_toxic', predictions_final.predict_all_toxic(), doc.is_toxic(), doc.id)

        apply_metric(metrics, 'toxic', predictions_final.predict_toxic(), doc.toxic, doc.id)
        apply_metric(metrics, 'severe_toxic', predictions_final.predict_severe_toxic(), doc.severe_toxic, doc.id)
        apply_metric(metrics, 'obscene', predictions_final.predict_obscene(), doc.obscene, doc.id)
        apply_metric(metrics, 'threat', predictions_final.predict_threat(), doc.threat, doc.id)
        apply_metric(metrics, 'insult', predictions_final.predict_insult(), doc.insult, doc.id)
        apply_metric(metrics, 'identity_hate', predictions_final.predict_identity_hate(), doc.identity_hate, doc.id)

    return metrics


# Write the metrics to the output file.
def export_results(metrics: dict):
    txt = 'category name, true positives, true negatives, false positives, false negatives\n'

    for cat in metrics:
        txt += '{}, {}, {}, {}, {}\n'.format(cat, metrics[cat]['true_pos'], metrics[cat]['true_neg'], metrics[cat]['false_pos'], metrics[cat]['false_neg'])
    txt = txt.rstrip('\n')

    f = open(output_file, 'w')
    f.write(txt)
    f.close()


def process_file():
    f = open(input_file, encoding="UTF-8")
    corpus = f.read()
    f.close()

    docs = create_docs_from_corpus(corpus, test_dataset_size)
    metrics = test_model(docs)
    export_results(metrics)


process_file()
