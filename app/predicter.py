import pandas as pd
from elasticsearch import Elasticsearch
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score
from joblib import dump, load
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from elasticer import Elasticer

from preprocessor import DataPreProcessor, DataCleaner

now = datetime.now()
timestamp = now.strftime("%d%m%Y-%H:%M")

'''
Importeer modellen, importeer data, gooi ze tegen elkaar aan. oke?
'''
NB = 'model-NB-28062019-12:14.joblib'
SGD = 'SGD-all-True-05082019-11:22.joblib'
SVM = 'SVM-all-True-05082019-11:22.joblib'

modelNB = load(f'../data/trained_models/bow_models/{NB}')
modelSGD = load(f'../data/trained_models/bow_models/{SGD}')
modelSVM = load(f'../data/trained_models/bow_models/{SVM}')



class xmlRemapper:
    def __init__(self):
        self.es = Elasticer()
    
        self.cleaner = DataCleaner()
        self.dataset_xml = 'sj-uk-vacancies-cleaned-4'
        self.exclude_xml = ['@language', 'DatePlaced', 'Id', 'companyLogo', 'country', 'topjob', 'HoursWeek', 'JobUrl', 'JobMinDaysPerWeek', 'JobParttime', 'JobCompanyBranch', 'JobCompanyProfile', 'JobRequirements.MinAge']
        self.include_xml = ['JobBranch', 'JobCategory', 'JobCompany', 'JobDescription','JobLocation.LocationRegion', 'JobProfession', 'Title','TitleDescription', 'functionTitle', 'postalCode', 'profession']
            
    def import_data(self):
        dataset = self.es.import_dataset(self.dataset_xml, self.include_xml)
        dataframe = self.prepro.to_dataframe(dataset)
        return dataframe

    def remap(self, dataframe):
        rename_dict = {'Title': 'vacancy_title',
                        'functionTitle': 'vacancy_title',
                        'TitleDescription': 'introduction',
                        'JobCategory': 'contract_type',
                        'JobBranch': 'job_category',
                        'JobDescription': 'description',
                        'profession': 'job_category',
                        'JobLocation.LocationRegion': 'location',
                        'postalCode': 'location',
                        'JobCompany': 'company_name',
                        'JobProfession': 'job_category'}
        dataframe.rename(columns=rename_dict, inplace=True)
        return dataframe

    def get_dataframe(self):
        self.prepro = DataPreProcessor()
        dataframe = self.import_data()
        dataframe = self.remap(dataframe)
        dataframe = self.prepro.transform_dataframe(dataframe)
        dataframe = self.cleaner.remove_values(dataframe, 'None')
        dataframe = self.cleaner.lowercase(dataframe)
        dataframe = self.cleaner.remove_stopwords(dataframe)
        return dataframe

def predict(title, model, dataframe):
    reports = ClassificationReports()
 
    reports.scoring_report(title, model, dataframe['text'], dataframe['label'])
   
    #proba = model.predict_proba(dataframe['text'])
    #best_n = np.argsort(proba)
    #print(best_n)

    #zipped = dict(zip(model.classes_, model.predict_proba(dataframe['text'])[0]))
    #print(zipped)

  
if __name__ == '__main__':
    xml_data = xmlRemapper().get_dataframe()
    predict('SGD_unseen_data', modelSGD, xml_data)
    predict('SVM_unseen_data', modelSVM, xml_data)


