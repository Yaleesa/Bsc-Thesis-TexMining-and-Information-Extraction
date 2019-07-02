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

now = datetime.now()
timestamp = now.strftime("%d%m%Y-%H:%M")

'''
Importeer modellen, importeer data, gooi ze tegen elkaar aan. oke?
'''
NB = 'model-NB-28062019-12:14.joblib'
SGD = 'model-SGD-all-False-29062019-23:24.joblib'

modelNB = load(f'trained_models/{NB}')
modelSGD = load(f'trained_models/{SGD}')


dataframe = pd.read_json("xml_data.json", orient="records")

def visualisations(title, y_test, y_pred):
    conf_mat = confusion_matrix(y_test, y_pred)
    #print(conf_mat)
    columns = ['company_name','contract_type','description','introduction','job_category','location','vacancy_title']
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(conf_mat, annot=True, fmt='d',
        xticklabels=columns, yticklabels=columns)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title(title, fontdict=None, loc='center', pad=None)
    plt.savefig(f'figure-{timestamp}.png')
    plt.show()
    plt.close()

def scoring(y_pred, y_test):
    accuracy = 'accuracy %s' % accuracy_score(y_pred, y_test)
    columns = ['company_name','contract_type','description','introduction','job_category','location','vacancy_title']
    report = classification_report(y_test, y_pred,target_names=columns)
    return accuracy, report

def predict(title, model, dataframe):
    #print(dataframe.head())
    tfidf_transformer = TfidfTransformer()
    count_vect = CountVectorizer()
    X = dataframe['text']
    #X_lower = dataframe['text'].str.lower()
    #X_new_counts = count_vect.transform(X)
    y_pred = model.predict(["senior developer in retail bizz part time in London"])
    print(model.__dict__)
    #print(predicted)
    # d = {'Text': X_lower, 'Label': dataframe['label'], 'Predicted': y_pred}
    # df = pd.DataFrame(d)
    # print(dataframe['label'].value_counts())
    # good_df = df[df.Label == df.Predicted]
    # bad_df = df[df.Label != df.Predicted]
    # print(good_df[good_df.Label == 'company_name'].sample())
    # print(bad_df[bad_df.Label == 'company_name'].sample(n=25))
    #print(good_df[good_df.Label == 'vacancy_title'])
    #proba = model.predict_proba(dataframe['text'])
    #best_n = np.argsort(proba)
    #print(best_n)

    #zipped = dict(zip(model.classes_, model.predict_proba(dataframe['text'])[0]))
    #print(zipped)

    # score, report = scoring(dataframe['label'], y_pred)
    # print(f'\n{title}\n{score} \n {report}')
    # visualisations(title, dataframe['label'], y_pred)
    # dflientje = pd.DataFrame(alles)
    # print(dflientje[['text','item','predicted','company_name','contract_type','description','introduction','job_category','location','vacancy_title']])

predict(SGD, modelSGD, dataframe)
