from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import pandas as pd
import ast
from tqdm import tqdm
import numpy as np
import random
import pickle

def save_object(obj, filename):
    with open(filename, 'wb') as outp:  # Overwrites any existing file.
        pickle.dump(obj, outp, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    with open(filename, 'rb') as handle:
        obj = pickle.load(handle)
        return obj
# sample usage


class SVMMEMMTagger:
    def __init__(self):
        self.model = None
        self.feature_encoder = None
        self.label_encoder = None

    def extract_features(self, sentence, i, prev_tag, prev2_tag):
        # Extract features for a given word in a sentence
        word = sentence[i]
        features = {
            'word': word,
            'prev_tag': prev_tag,
            'prev2_tag': prev2_tag,
            'is_first': i == 0,
            'is_last': i == len(sentence) - 1,
        }
        return features

    def data_generator(self, tagged_corpus, chunk_size=1000):
        X, y = [], []

        for tagged_sentence in tagged_corpus:
            sentence, tags = zip(*tagged_sentence)
            for i in range(len(sentence)):
                features = self.extract_features(sentence, i, tags[i-1] if i > 0 else '<s>', tags[i-2] if i > 1 else '<s>')
                X.append(features)
                y.append(tags[i])

                if len(X) == chunk_size:
                    yield X, y
                    X, y = [], []
                    
    def prepare_data(self, tagged_corpus):
        X, y = [], []

        for tagged_sentence in tagged_corpus:
            sentence, tags = zip(*tagged_sentence)
            for i in range(len(sentence)):
                features = self.extract_features(sentence, i, tags[i-1] if i > 0 else '<s>', tags[i-2] if i > 1 else '<s>')
                X.append(features)
                y.append(tags[i])

        return X, y
    
    def train(self, tagged_corpus, chunk_size=10000, epochs=1):
        # Encode features and labels
        self.feature_encoder = DictVectorizer(sparse=False)
        self.label_encoder = LabelEncoder()
        X, y = self.prepare_data(tagged_corpus)
        self.feature_encoder.fit(X)
        y_encoded = self.label_encoder.fit_transform(y)
        classes= np.unique(y_encoded)
        self.model = SGDClassifier(loss='hinge', max_iter=epochs, random_state=42, verbose=1, n_jobs=-1)
        cnt = 0
        for X_chunk, y_chunk in self.data_generator(tagged_corpus, chunk_size):
            cnt+=1
            X_encoded = self.feature_encoder.transform(X_chunk)
            y_encoded = self.label_encoder.transform(y_chunk)
            self.model.partial_fit(X_encoded, y_encoded, classes=classes)
            if (cnt == 10):
                break

    def tag_sentence(self, sentence):
        if not self.model or not self.feature_encoder or not self.label_encoder:
            print("Error: Model not trained yet.")
            return []

        tagged_sentence = []
        prev_tag, prev2_tag = '<s>', '<s>'

        for i, word in enumerate(sentence):
            features = self.extract_features(sentence, i, prev_tag, prev2_tag)
            features_encoded = self.feature_encoder.transform([features])
            predicted_label = self.label_encoder.inverse_transform(self.model.predict(features_encoded))[0]
            tagged_sentence.append((word, predicted_label))
            prev2_tag = prev_tag
            prev_tag = predicted_label

        return tagged_sentence

# Example usage:
# df = pd.read_csv('../data/train.csv')
# tagged_corpus = [ast.literal_eval(row['tagged_sentence']) for _, row in tqdm(df.iterrows())]
# random.shuffle(tagged_corpus)
# tagger = SVMMEMMTagger()
# tagger.train(tagged_corpus)
# save_object(tagger, 'tagger.pkl')
tagger = load_object('tagger.pkl')
new_sentence = ['The', 'jury', 'further', 'said', 'in', 'term-end', 'presentments', 'that', 'the', 'City', 'Executive']
tagged_sentence = tagger.tag_sentence(new_sentence)
print(tagged_sentence)


df = pd.read_csv('../data/test_small.csv') # loading test data
test_data = {}
for index, row in tqdm(df.iterrows()):
    test_data[row['id']] = ast.literal_eval(row['untagged_sentence']) # changing data-type of entries from 'str' to 'list'
    
submission = {'id': [], 'tagged_sentence' : []} # dictionary to store tag predictions
# NOTE ---> ensure that tagged_sentence's corresponing 'id' is same as 'id' of corresponding 'untagged_sentence' in training data
def store_submission(sent_id, tagged_sentence):
    global submission
    if(sent_id in list(submission['id'])):
        return
    submission['id'].append(sent_id)
    submission['tagged_sentence'].append(tagged_sentence)
    
    
for sent_id in tqdm(list(test_data.keys())):
    sent = test_data[sent_id]
    tagged_sentence = tagger.tag_sentence(sent)
    store_submission(sent_id, tagged_sentence)
    
import os
path = '../data/submission_memm.csv'
if (os.path.exists(path)):
    os.remove(path)
pd.DataFrame(submission).to_csv(path, index = False)