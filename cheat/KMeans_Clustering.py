import re

import nltk
import pandas as pd
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.data.path.append("nltk_data")
stemmer = SnowballStemmer('portuguese')
stopwords = nltk.corpus.stopwords.words('portuguese')

df = pd.read_csv('datasets/documents.csv')
docs = df['docs'].tolist()


def tokenize_and_stem(text):
    tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
    filtered_tokens = []
    for token in tokens:
        if re.search('[a-zA-Z]', token):
            filtered_tokens.append(token)
    stems = [stemmer.stem(t) for t in filtered_tokens]
    return stems


def cluster(documents, num_clusters):
    tfidf_vectorizer = TfidfVectorizer(max_df=1.0, min_df=0.0, stop_words=stopwords, use_idf=True,
                                       tokenizer=tokenize_and_stem, ngram_range=(1, 3))
    matrix = tfidf_vectorizer.fit_transform(documents)
    km = KMeans(n_clusters=num_clusters, init='random')
    km.fit_predict(matrix)
    clusters = km.labels_.tolist()

    result = {'question': documents, 'cluster_id': clusters}
    return pd.DataFrame(result, index=clusters).sort_values(by='cluster_id', ascending=True)


cluster(documents=docs, num_clusters=5)
