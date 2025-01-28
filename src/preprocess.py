import pandas as pd
import os 
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

target_dir = os.path.join(os.getcwd(), "data", "data_set.csv")

dataset = pd.read_csv(target_dir)


nltk.download('stopwords')


def preprocess(dataset):
    stemmer = PorterStemmer()
    dataset['Text'] = dataset['Text'].str.lower()
    dataset['Text'] = dataset['Text'].str.replace(r'[^\w\s]', '', regex=True)
    stop_words = set(stopwords.words('english'))
    dataset['Text'] = dataset['Text'].apply(lambda x: ' '.join(word for word in x.split() if word not in stop_words))

    dataset['Text'] = dataset['Text'].apply(lambda x: ' '.join(stemmer.stem(word) for word in x.split()))
    
    return  dataset

preprocessed_dataset = preprocess(dataset)


