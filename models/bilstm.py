### Bi-LSTM Model ### 

import torchtext as tt
from torchtext.legacy import data
from torchtext.vocab import Vectors
import spacy 
import torch.optim as optim
import torch
import torch.nn as nn 
from sklearn.metrics import * 
from tqdm import tqdm_notebook
from utils import f1_precision_recall

class Config(object):
    embed_size = 300
    hidden_layers = 2
    hidden_size = 128
    bidirectional = True                 
    output_size = 2
    max_epochs = 30 #20 
    lr = 2e-5 
    weight_decay =  1e-4
    batch_size = 8
    max_sen_len = 256 
    dropout_keep = 0.5


class BiLSTMForTelemedicalQueryClassification(nn.Module):
    def __init__(self, train_set, test_set, path_to_glove = '/content/drive/MyDrive/glove_vecs/glove.6B.300d.txt'):
        super().__init__()

        self.config = Config()

        self.w2v_file = path_to_glove
        self.dataset = Dataset()
        self.train_set, self.test_set = train_set, test_set
        self.dataset.load_data(self.w2v_file, self.train_set, self.test_set)

        self.model = TextRNN(self.config, len(self.dataset.vocab), self.dataset.word_embeddings)
        self.model.cuda() 
        self.model.train() 
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr, weight_decay = self.config.weight_decay)
        self.loss_fn = nn.CrossEntropyLoss()
        self.model.add_optimizer(self.optimizer)
        self.model.add_loss_op(self.loss_fn)

    def train(self):
      train_losses = []
      val_accuracies = []

      print("Training BiLSTM")
      for i in tqdm_notebook(range(self.config.max_epochs)):
          train_loss, val_accs = self.model.run_epoch(self.dataset.train_iterator, self.dataset.test_iterator, i)
          train_losses.append(train_loss)

      print("Training complete!")
    def evaluate(self):
      (f1, precision, recall), preds = evaluate_model(self.model, self.dataset.test_iterator)
      return (f1, precision, recall), preds

class Dataset(object):
    def __init__(self,):
   
        self.train_iterator = None
        self.test_iterator = None
        self.val_iterator = None
        self.vocab = []
        self.word_embeddings = {}

    
    def load_data(self, w2v_file, train, test):
 
        NLP = spacy.load('en')
        tokenizer = lambda sent: [x.text for x in NLP.tokenizer(sent) if x.text != " "]
        
        # Creating Field for data
        TEXT = data.Field(sequential=True, tokenize=tokenizer, lower=True, fix_length=256)
        LABEL = data.Field(sequential=False, use_vocab=False)
        datafields = [("query",TEXT),("label",LABEL)]
        
        train_examples = [data.Example.fromlist(i, datafields) for i in train[['query', 'label']].values.tolist()]
        train_data = data.Dataset(train_examples, datafields)

        test_examples = [data.Example.fromlist(i, datafields) for i in test[['query', 'label']].values.tolist()]
        test_data = data.Dataset(test_examples, datafields)
 
        
        TEXT.build_vocab(train_data, vectors=Vectors(w2v_file))
        self.word_embeddings = TEXT.vocab.vectors
        self.vocab = TEXT.vocab
        
        self.train_iterator = data.BucketIterator(
            (train_data),
            batch_size=8,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=True)
        
        self.test_iterator = data.BucketIterator(
            (test_data),
            batch_size=8,
            sort_key=lambda x: len(x.text),
            repeat=False,
            shuffle=False)
        
        
        print ("Loaded {} training examples".format(len(train_data)))
        print ("Loaded {} test examples".format(len(test_data)))


def evaluate_model(model, iterator):
    all_preds = []
    all_y = []
    for idx,batch in enumerate(iterator):
        x = batch.query.cuda() 
        y_pred = model(x)
        predicted = torch.max(y_pred.cpu().data, 1)[1]

        all_preds.extend(predicted.numpy())
        all_y.extend(batch.label.numpy())
    
    f1, precision, recall = f1_precision_recall(all_y, np.array(all_preds).flatten()) 

    return (f1, precision, recall), np.array(all_preds).flatten() #


class TextRNN(nn.Module):
    def __init__(self, config, vocab_size, word_embeddings):
        super(TextRNN, self).__init__()
        self.config = config
        
        # Embedding Layer
        self.embeddings = nn.Embedding(vocab_size, self.config.embed_size)
        self.embeddings.weight = nn.Parameter(word_embeddings, requires_grad=False)
        
        self.lstm = nn.LSTM(input_size = self.config.embed_size,
                            hidden_size = self.config.hidden_size,
                            num_layers = self.config.hidden_layers,
                            dropout = self.config.dropout_keep,
                            bidirectional = self.config.bidirectional)
        
        self.dropout = nn.Dropout(self.config.dropout_keep)
  
        self.bn = nn.LayerNorm(self.config.hidden_size * (1+self.config.bidirectional))
        self.fc = nn.Linear(
            self.config.hidden_size * (1+self.config.bidirectional),
            self.config.output_size
        )

        self.softmax = nn.Softmax()

    def forward(self, x):
        embedded_sent = self.embeddings(x)
        lstm_out, (h_n,c_n) = self.lstm(embedded_sent)
        lstm_out = self.dropout(lstm_out.mean(axis=0)) 
        lstm_out = self.bn(lstm_out)
        final_out = self.fc(lstm_out)
        return self.softmax(final_out)
    
    def add_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def add_loss_op(self, loss_op):
        self.loss_op = loss_op
                
    def run_epoch(self, train_iterator, val_iterator, epoch):
        train_losses = []
        val_accuracies = []
        losses = []

        for i, batch in enumerate(train_iterator):
            self.optimizer.zero_grad()
            if torch.cuda.is_available():
                x = batch.query.cuda()
  
                y = (batch.label).type(torch.cuda.LongTensor)
            else:
                x = batch.query
                y = (batch.label).type(torch.LongTensor)
            y_pred = self.__call__(x)
            loss = self.loss_op(y_pred, y)
            loss.backward()
            losses.append(loss.data.cpu().numpy())
            self.optimizer.step()
    
                
        return train_losses, val_accuracies