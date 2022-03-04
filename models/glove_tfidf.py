# GloVe vectorizer based off https://edumunozsala.github.io/BlogEms/jupyter/nlp/classification/embeddings/python/2020/08/15/Intro_NLP_WordEmbeddings_Classification.html#Applying-the-word-embedding-to-a-text-classification-task 

import re

from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
import nltk
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import numpy as np 

from utils import f1_precision_recall

nltk.download('punkt')
nltk.download('stopwords')

class SVMForTelemedicalQueryClassification():
    def __init__(self, train_set, test_set, path_to_glove=None, model_type = 'TFIDF'):

        self.train_set = train_set
        self.test_set = test_set

        self.stop_words = stopwords.words('english')
        self.stemmer = SnowballStemmer('english') 

        if model_type == 'GLOVE':
          self.load_glove(path_to_glove)
          self.vectorizer = GloveVectorizer(self.glove_embeddings)
        elif model_type == 'TFIDF':
          self.vectorizer = TfidfVectorizer(tokenizer=self.tokenize)

        self.X_train_question = self.vectorizer.fit_transform(self.train_set['query'])
        self.X_test_question = self.vectorizer.transform(self.test_set['query'])
        self.model = SVC() 

    def train(self):
      self.model.fit(self.X_train_question, self.train_set['label'])

    def evaluate(self):
      labels = self.test_set['label']
      test_preds = self.model.predict(self.X_test_question)
      f1, precision, recall = f1_precision_recall(labels, test_preds)


      return (f1, precision, recall), test_preds

    def load_glove(self, path):
        '''
        Load glove vectors 
        '''
        glove_embeddings = dict()
        with open(path, 'r') as f:
          for line in f:
            vals = line.split()
            word = vals[0]
            vec = np.asarray(vals[1:], 'float32')
            glove_embeddings[word] = vec
        self.glove_embeddings = glove_embeddings

    def tokenize(self, s):
      '''
      Tokenize samples 
      '''
      return [self.stemmer.stem(w) for w in nltk.word_tokenize(s.lower()) if w not in self.stop_words]

    def tokenize_raw_symptoms(self, s):
      return [w.strip() for w in re.split(':[A-Za-z]*,', s)[:-1]]

    def preprocess_symptoms(self, raw_symptoms):
      return list(map(lambda s: 'none' if not s else s.lower(), raw_symptoms))

class GloveVectorizer:
  def __init__(self, embeddings):
    self.embeddings = embeddings
    self.D = 300

  def fit(self, data):
    pass

  def transform(self, data):
    X = np.zeros((len(data), self.D))
    n = 0
    emptycount = 0
    for sentence in data:
      tokens = sentence.split()
      vecs = []
      m = 0
      for word in tokens:
        try:
          vec = self.embeddings[word]
          vecs.append(vec)
          m += 1
        except KeyError:
          pass

      if len(vecs) > 0:
        vecs = np.array(vecs)
        X[n] = vecs.mean(axis=0)
      else:
        emptycount += 1
      n += 1
    return X

  def fit_transform(self, data):
    self.fit(data)
    return self.transform(data)
