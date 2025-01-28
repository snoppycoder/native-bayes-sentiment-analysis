from collections import Counter, defaultdict
from .preprocess import preprocess
import os
import numpy as np
import pandas as pd
target_dir = os.path.join(os.getcwd(), "data", "data_set.csv")
data = pd.read_csv(target_dir)
dataset = preprocess(data)
def build_vocabulary(dataset):
    vocabulary = set()
    for text in dataset['Text']:
        vocabulary.update(text.split())
    return list(vocabulary)
def text_to_vector(text, vocabulary):
    word_count = Counter(text.split())
    return [word_count.get(word, 0) for word in vocabulary]
vocabulary = build_vocabulary(dataset)
x = np.array([text_to_vector(text, vocabulary) for text in dataset['Text']])
y = dataset['Sentiment']


