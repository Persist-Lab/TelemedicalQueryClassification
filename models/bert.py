from transformers import  BertForSequenceClassification, Trainer, TrainingArguments, AutoTokenizer
import torch.nn as nn 
import numpy as np 
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score 
from datasets import Dataset, DatasetDict 

from utils import f1_precision_recall

class BERTForTelemedicalQueryClassification(nn.Module):
    def __init__(self, train_set, test_set, model_name, num_labels = 2, lr = 5e-5, num_epochs = 5, batch_size = 8, weight_decay = 0.01):
        super().__init__()

        ### Prep Data ### 
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.MAX_LEN = 200


        train_set, test_set = Dataset.from_pandas(train_set), Dataset.from_pandas(test_set)
        self.train_enc_data = train_set.map(self.preprocess_data, batched=False)
        self.test_enc_data = test_set.map(self.preprocess_data, batched=False)

        ### Prep Model ###
        self.num_labels = num_labels
        self.model = BertForSequenceClassification.from_pretrained(model_name, num_labels = num_labels)

        ### Prep Trainer ###
        self.training_args = TrainingArguments(
                  do_train=True,
                  do_eval=False,
                  output_dir="/content",
                  learning_rate=lr,
                  num_train_epochs=num_epochs,
                  overwrite_output_dir=True,
                  per_device_eval_batch_size=batch_size,
                  per_device_train_batch_size=batch_size,
                  logging_steps = 100,
                  weight_decay = weight_decay,
                  load_best_model_at_end = False
              )
        self.trainer = Trainer(
                model=self.model,
                args=self.training_args,
                train_dataset=self.train_enc_data,
                eval_dataset=self.test_enc_data,
                compute_metrics=self.compute_metrics,
            )

    def compute_metrics(self, p):
        preds = p.predictions.argmax(-1)
        labels = p.label_ids
        f1, precision, recall = f1_precision_recall(labels, preds)
        return {
                'f1':f1,
                'precision': precision,
                'recall': recall
                }

    def train(self):
      self.trainer.train()

    def evaluate(self):
      results = self.trainer.evaluate() 
      f1, precision, recall = results['eval_f1'], results['eval_precision'], results['eval_recall']

      preds = self.trainer.predict(self.test_enc_data)
      preds = preds.predictions.argmax(-1)

      return (f1, precision, recall), preds 

    def preprocess_data(self, x):
      data = self.tokenizer(x['query'], max_length = self.MAX_LEN, truncation=True, padding = 'max_length')
      data['labels'] = [x['label']] 
      return data