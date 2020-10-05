import random
import copy

import numpy as np

import nltk
import nltk.stem

import nltk.corpus
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')

import sklearn
import sklearn.naive_bayes
import sklearn.ensemble
import sklearn.metrics
import sklearn.feature_extraction.text
import sklearn.utils
import sklearn.utils.testing
import sklearn.exceptions

random_state = 1111
np.random.seed(random_state)
random.seed(random_state)

# ================
# Pre-process data
# ================

def read_lines(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as file_handler:
        return [sentence.rstrip() for sentence in file_handler.readlines()]

class Lemmatizer:
    def __init__(self):
        self.normalizer = nltk.stem.WordNetLemmatizer()
        self.tag_prefix_dict = {
            'J': nltk.corpus.wordnet.ADJ,
            'N': nltk.corpus.wordnet.NOUN,
            'V': nltk.corpus.wordnet.VERB,
            'R': nltk.corpus.wordnet.ADV
        }
    
    def __call__(self, document):
        tokens = nltk.word_tokenize(document)
        return [
            self.normalizer.lemmatize(token, pos=self.get_tag_class(tag))
            for token, tag in nltk.pos_tag(tokens)
        ]
    
    def get_tag_class(self, tag):
        prefix = tag[0].upper()
        return self.tag_prefix_dict.get(prefix, nltk.corpus.wordnet.NOUN)

class Stemmer:
    def __init__(self):
        self.normalizer = nltk.stem.PorterStemmer()
    
    def __call__(self, document):
        return [
            self.normalizer.stem(token)
            for token in nltk.word_tokenize(document)
        ]

def fit_vectorizer(X_data, tokenizer, stop_words, min_df):    
    vectorizer = sklearn.feature_extraction.text.CountVectorizer(
        tokenizer=tokenizer,
        stop_words=stop_words,
        min_df=min_df
    )
    vectorizer.fit_transform(X_data)
    return vectorizer

data_folder = 'E:/Repos/comp550/assignment1/data'
positives_file = 'rt-polarity.pos'
negatives_file = 'rt-polarity.neg'

splits = [0.8, 0.9]

negatives_data = [
    (document, 'negative')
    for document in read_lines(f'{data_folder}/{negatives_file}')
]
positives_data = [
    (document, 'positive')
    for document in read_lines(f'{data_folder}/{positives_file}')
]
all_data = sklearn.utils.shuffle(np.array(negatives_data + positives_data), random_state=random_state)
n = all_data.shape[0]

X_train_raw, y_train = all_data[:int(splits[0]*n), 0], all_data[:int(splits[0]*n), 1]
X_valid_raw, y_valid = all_data[int(splits[0]*n):int(splits[1]*n), 0], all_data[int(splits[0]*n): int(splits[1]*n), 1]
X_test_raw, y_test = all_data[int(splits[1]*n):, 0], all_data[int(splits[1]*n):, 1]

# ===================================================
# Validation set hyper-parameter search across models
# ===================================================

@sklearn.utils.testing.ignore_warnings(category=sklearn.exceptions.ConvergenceWarning)
def random_search(models, search_params, n_datasets=1, n_models=1):
    data_sets = []
    results = [{} for i in range(n_datasets)]

    data_set_variations = [
        choose_random_params(search_params['data'])
        for i in range(n_datasets)
    ]
    model_variations = {
        model_name: [choose_random_params(search_params['model'][model_name]) for i in range(n_models)]
        for model_name in search_params['model'].keys()
    }
    
    for i in range(n_datasets):
        print(f'Data set variation {i+1}/{n_datasets}')

        data_params = data_set_variations[i]
        data_sets.append(data_params)

        print('\tFitting vectorizer...')
        vectorizer = fit_vectorizer(X_train_raw, **data_params)

        X_train = vectorizer.transform(X_train_raw)
        X_valid = vectorizer.transform(X_valid_raw)
        X_test = vectorizer.transform(X_test_raw)

        for model_name, model_class in models.items():
            for j in range(n_models):
                print(f'\t{model_name} {j+1}/{n_models}')

                model_params = model_variations[model_name][j]

                model = model_class(**model_params)

                model.fit(X_train, y_train)
                valid_predictions = model.predict(X_valid)
                test_predictions = model.predict(X_test)
                
                valid_accuracy = sklearn.metrics.accuracy_score(y_valid, valid_predictions)
                
                # This number is only looked at once at the very end when the best models have been chosen based on validation accuracy
                test_accuracy = sklearn.metrics.accuracy_score(y_test, test_predictions)
                test_confusion_matrix = sklearn.metrics.confusion_matrix(y_test, test_predictions)

                if results[i].get(model_name, None) is None:
                    results[i][model_name] = []

                results[i][model_name].append({
                    'model_params': model_params,
                    'valid_accuracy': valid_accuracy,
                    'test_accuracy': test_accuracy,
                    'test_confusion_matrix': test_confusion_matrix
                })
    
    return data_sets, results

def choose_random_params(parameters):
    return {
        name: np.random.choice(values)
        for name, values in parameters.items()
    }

search_params = {
    'data': {
        'tokenizer': [Lemmatizer(), Stemmer()],
        'stop_words': ['english', None],
        'min_df': [1, 2, 3] # Minimum token frequency
    },
    'model': {
        'logistic_regression': {
            'eta0': [1e-3, 1e-2, 1e-1], # learning rate
            'alpha': [1e-3, 1e-2, 1e-1], # regularization
            'max_iter': np.arange(start=1, stop=5), # epochs
            'random_state': [random_state]
        },
        'linear_support_vector_machine': {
            'kernel': ['linear'],
            'max_iter': np.arange(start=1, stop=5), # epochs
            'C': [1e-3, 1e-2, 1e-1], # L2 regularization
            'random_state': [random_state]
        },
        'naive_bayes': {
            'alpha': np.arange(start=0.1, stop=1.1, step=0.1)
        },
        'random_forest': {
            'n_estimators': np.arange(start=10, stop=1000, step=10),
            'max_depth': np.append(np.array(None), np.arange(start=1, stop=5, step=2)),
            'random_state': [random_state]
        }
    }
}

models = {
    'logistic_regression': sklearn.linear_model.SGDClassifier,
    'linear_support_vector_machine': sklearn.svm.SVC,
    'naive_bayes': sklearn.naive_bayes.MultinomialNB,
    'random_forest': sklearn.ensemble.RandomForestClassifier
}

data_sets, results = random_search(models, search_params, n_datasets=3, n_models=4)

for i, params in enumerate(data_sets):
    print(f'\nDataset variation {i+1}: {params}')
    model_results = results[i]
    for model_name, params in model_results.items():
        print(f'\t{model_name}:')
        for j in range(len(params)):
            print('\t\t{}'.format({
                    'model_params': params[j]['model_params'],
                    'valid_accuracy': params[j]['valid_accuracy']
            }))

# ================
# Test set results
# ================

best_models = {
    'logistic_regression': 1,
    'naive_bayes': 0,
    'random_forest': 0
}

for model_name, model_index in best_models.items():
    print(f'Best {model_name} test sets accuracies:')
    for i in range(len(results)):
        print(f'\t{i+1}: {results[i][model_name][model_index]["test_accuracy"]}')

print(f'No best linear_support_vector_machine test sets accuracies')

random_predictions = np.random.choice(['negative', 'positive'], size=y_test.shape[0])
random_accuracy = sklearn.metrics.accuracy_score(y_test, random_predictions)
print(f'Random classifier test set accuracy: {random_accuracy}')

print('Best model (Naive Bayes) confusion matrix:')
print(results[0]['naive_bayes'][0]['test_confusion_matrix'])
