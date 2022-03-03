from sklearn.metrics import f1_score, precision_score, recall_score, classification_report
def f1_precision_recall(labels, preds):
    f1 = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds, average='weighted')
    recall = recall_score(labels, preds, average='weighted') 
    print(classification_report(labels, preds)) 
    return f1, precision, recall 