'''
Author: Yaleesa Borgman
Date: 8-8-2019
predicter.py - handles predictions from imported models
'''

from elasticsearch import Elasticsearch

import pandas as pd
from joblib import dump, load
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from elasticer import Elasticer
from preprocessor import DataPreProcessor, DataCleaner
from reporter import ClassificationReports

'''
Author: Yaleesa Borgman
Date: 8-8-2019
predicter.py - predicts class for input
'''

import glob
import os
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%d%m%Y-%H:%M")

'''
predictions,
default = latest model 
- predict = returns label
- predict_probs = returns label + probability 
'''
class Predictions:
    def import_model(model='bow_latest'):
        if model == 'fasttext_latest':
            model_map = 'fasttext_models'
        
        elif model == 'bow_latest':
            model_map = 'bow_models'
        
        else: 
            return load(model)

        list_of_files = glob.glob(f'../data/trained_models/{model_map}/*')
        latest_file = max(list_of_files, key=os.path.getctime)
        model = load(latest_file)
        return model

    def predict(title,  dataframe, model='bow_latest'):
        model = import_model(model)
        reports = ClassificationReports()
        y_pred = model.predict(model, X)
        reports.scoring_report(title, y_pred, dataframe['text'], dataframe['label'])

    def predict_probs(title,  dataframe, model='bow_latest')):
        model = import_model(model)
        reports = ClassificationReports()
        proba = model.predict_proba(dataframe['text'])
        best_n = np.argsort(proba)
        zipped = dict(zip(model.classes_, model.predict_proba(dataframe['text'])[0]))
        print(zipped)

  


