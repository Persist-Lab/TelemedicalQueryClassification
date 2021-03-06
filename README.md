# TelemedicalQueryClassification
Code for the paper: Identifying the Perceived Severity of Patient-generated Telemedicine Queries Regarding COVID:  Developing and Evaluating a Transfer Learning Based Solution

## Instructions 
Telemedical query dataset is found in  **medical_query_dataset.csv**

Full example of how to run each model can be found on Google Colab [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1Inggl-ILWpyNqFqNZmoS4mRC3qOu2fK4?usp=sharing)

Please note that, in order to run glove-based models, one must download them [here](https://nlp.stanford.edu/projects/glove/). We use glove.6b.300d for this study. Add that path to the config found in the Jupyter notebook to run glove-based models (we suggest doing so through mounting your google drive). Transformer/TFIDF models will all run without further intervention. 

Additionally, please note that outputs are not fully deterministic as 1) the linear classifiers are randomly initialized and 2) performance can vary slightly depending on Cross Validation split. Small-data problems are particularly sensitive to initializations, thus you might not produce the exact results found in the paper, however they will be close. Feel free to request pre-trained weights for any of the models as needed. 

## References 

The queries in this dataset were collected by those managing [this repository](https://github.com/UCSD-AI4H/COVID-Dialogue/) and they should be cited accordingly. Additionally, we note that samples are from [icliniq.com](https://www.icliniq.com/), [healthcaremagic.com](https://www.healthcaremagic.com/), [healthtap.com](https://www.healthtap.com/) and all copyrights of the data belong to these websites.


### TODO

- Add HAN to codebase
