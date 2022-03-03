### General Functions ### 
def data_integrity_checker(train, test):
  train_texts = train.original_text.tolist()
  test_texts = test.original_text.tolist()

  for text in train_texts:
    assert text not in test_texts

def f1_precision_recall(labels, preds):
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted') 
    print(classification_report(labels, preds)) 
    return f1, precision, recall 