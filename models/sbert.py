### SBERT LOSS CLASSIFIER ### 
import torch.nn as nn 
import torch 

from sentence_transformers import SentenceTransformer, InputExample, LoggingHandler, losses, models, util
from torch.utils.data import DataLoader
from sentence_transformers.evaluation import TripletEvaluator
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import scipy.stats as st

from utils import f1_precision_recall

class SBERTForTelemedicalQueryClassification(nn.Module):
  def __init__(self, train_set, test_set, triplets_per = 50, batch_size = 8, num_epochs = 1):
    super(SBERTForTelemedicalQueryClassification, self).__init__()
    self.train_set = train_set
    self.test_set = test_set
    self.triplets_per = triplets_per
    self.train_triplets = self.get_triplets(self.train_set)
    self.batch_size = batch_size
    self.num_epochs = num_epochs

    # Build training example list for sentence-transformer library using InputExamples 
    self.train_examples = []
    for sample in self.train_triplets:
      self.train_examples.append(InputExample(texts=[sample[0][0], sample[1][0], sample[2][0]]))

    # Initialize embedding model 
    self.word_embedding_model = models.Transformer('all-MiniLM-L6-v2')
    self.pooling_model = models.Pooling(self.word_embedding_model.get_word_embedding_dimension(),
                                  pooling_mode_mean_tokens=True,
                                  pooling_mode_cls_token=False,
                                  pooling_mode_max_tokens=False)
    self.model = SentenceTransformer(modules=[self.word_embedding_model, self.pooling_model])
    self.model.max_seq_length = 200
  def train(self):
    '''
    Fine-Tune SBERT using Triplet loss. 
    '''
    train_dataloader = DataLoader(self.train_examples, shuffle=True, batch_size=self.batch_size)
    train_loss = losses.TripletLoss(model=self.model)
    warmup_steps = int(len(train_dataloader) * self.num_epochs * 0.1)

    self.model.fit(train_objectives=[(train_dataloader, train_loss)],
            epochs=self.num_epochs,
            warmup_steps=warmup_steps,
            output_path='/content/')
    
  def evaluate(self, n=10):
    
    # Get embeddings to train/test KNN 
    train_embeddings = [self.model.encode(x) for x in self.train_set['query'].tolist()]
    test_embeddings = [self.model.encode(x) for x in self.test_set['query'].tolist()]
    
    # Fit KNN
    knn=KNeighborsClassifier(n)
    knn.fit(train_embeddings, self.train_set.label.tolist())
    # Get Preds 
    preds = knn.predict(test_embeddings)

    f1, precision, recall = f1_precision_recall(self.test_set.label.tolist(), preds)
    return (f1, precision, recall), preds 

  def get_triplets(self, dataset):

    # Separate severe from non severe samples 
    non_severes = dataset[dataset.label==0]
    severes = dataset[dataset.label==1]
    class_dfs = {"non_severes": non_severes,
                 "severes": severes}
    anchor_triplets = []

    # for each df in [non_severe, severe] 
    for key, df in class_dfs.items():
      for s in range(len(df)):
        # Get sample S
        text, label = df.iloc[s]
        # Create DF with every sample which is not S
        all_others = df[df['query'] != text]
        # Randomly generate triplets 
        for i in range(self.triplets_per):
          pos = all_others.sample(n=1)['query'].item()
          if key == "severes":
            neg = non_severes.sample(n=1)['query'].item()
            anchor_triplets.append([(text, 1), (pos, 1),(neg, 0)])
          elif key == "non_severes":
            neg = severes.sample(n=1)['query'].item()
            anchor_triplets.append([(text, 0), (pos, 0),(neg, 1)])
          else:
            print(key)
            raise NotImplementedError 
    
    return anchor_triplets
