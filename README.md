# TelemedicalQueryClassification
Code for the paper: Identifying the Perceived Severity of Patient-generated Telemedicine Queries Regarding COVID:  Developing and Evaluating a Transfer Learning Based Solution


Telemedical query dataset is located in found in **medical_query_dataset.csv**

Full example of how to run each model can be found on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Inggl-ILWpyNqFqNZmoS4mRC3qOu2fK4?usp=sharing)

Please note that, in order to run glove-based models, one must download them [here](https://nlp.stanford.edu/projects/glove/). We use glove.6b.300d for this study. Add that path to the config found in the Jupyter notebook to run glove-based models (we suggest doing so through mounting your google drive). Transformer/TFIDF models will all run without further intervention. 


Additionally, please note that non-lexical models are not fully deterministic as the linear classifiers are randomly initialized. Small-data problems are particularly sensitive to initialization, thus you might not produce the exact results found in the paper, however they will be close. Feel free to request pre-trained weights for any of the models as needed. 



### TODO

- Add HAN to codebase
