'''
Author: Yaleesa Borgman
Date: 8-8-2019
trainer.py - handles the text classification, can be used stand alone or with pipeline class 
'''

'''
Sklearn
'''
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from joblib import dump, load

'''
Pandas & Numpy & vis
'''
import pandas as pd
import numpy as np
np.set_printoptions(threshold=np.inf)
pd.set_option("display.max_rows", 25)
pd.set_option('max_columns', 7)
pd.reset_option('max_columns')

import matplotlib.pyplot as plt
import seaborn as sns
'''
globals
'''
import os
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%d%m%Y-%H:%M")

'''
Modules
'''
from elasticer import Elasticer
from explorer import DataExploration
from preprocessor import DataPreProcessor, DataCleaner
from reporter import ClassificationReports

class TextClassification:
    '''
    Text Classification with sklearn, splitting, training and saving a model.
    following classifiers can be chosen: ['NB', 'SVM', 'LR', 'SGD', 'DT', 'RF']
    '''
    def splitting_dataset(self, data_size, dataframe):
        '''
        Splitting the test and train dataset
        '''
        text = dataframe['text'].values
        y = dataframe['label'].values

        if data_size == 'all':
            return text, y

        else:
            text_train, text_test, y_train, y_test = train_test_split(
                    text, y, test_size=0.25, random_state=2, stratify=True)

            return text_train, text_test, y_train, y_test, text, y

    def model_pipeline(self, wanted_clf, X, y):
        '''
        training pipeline
        '''
        classifiers = {
                  'NB': MultinomialNB(),
                  'SVM': LinearSVC(),
                  'SGD': SGDClassifier(),
                  'LR': LogisticRegression(multi_class='auto'),
                  'DT': DecisionTreeClassifier(),
                  'RF': RandomForestClassifier()
                  }


        model = Pipeline([('vect', CountVectorizer(stop_words='english')),
                       ('tfidf', TfidfTransformer()),
                       ('clf', classifiers[wanted_clf]),
                      ])

        model.fit(X, y)
        return model

    def grid_search(self, wanted_clf, X, y):
        clf = self.model_pipeline(wanted_clf)
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            #'clf__alpha': (1e-2, 1e-3),
        }
  
        grid_clf = GridSearchCV(clf, parameters, cv=10, iid=False, n_jobs=-1)
        grid_clf.fit(X, y)

        #print(grid_clf.cv_results_)
        print(grid_clf.best_estimator_.get_params())
        return {'model':grid_clf.best_estimator_, 'best_params': grid_clf.best_estimator_.get_params()}

    def train_model(self, X, y, wanted_clf):
        #best_model = self.grid_search(wanted_clf, X, y)
        clf = self.model_pipeline(wanted_clf, X, y)
        #clf.fit(X, y)
        return clf

    def model_save(self, clf, filename):
        dump(clf, f'trained_models/{filename}.joblib')

class CrossValidation:
    def __init__(self):
        self.TextClassifier = TextClassification()

    def cross_validation(self, X, y, wanted_clf):
        skf = StratifiedKFold(n_splits=10)
        clf = self.TextClassifier.train_model(X, y, wanted_clf)
        scores = cross_val_score(clf, X, y, cv=skf)
        y_pred = cross_val_predict(clf, X, y, cv=skf)
        return scores, y_pred
    
    def cross_validation_report(self, X, y, wanted_clf, kijkdoos=False, vis=True):
        reports = ClassificationReports()
        scores, y_pred = self.cross_validation(X, y, wanted_clf)
        report = reports.cv_report(wanted_clf, scores, y, y_pred, vis=True)
        return report, y_pred 

class FeatureSelection: 
    def __init__(self):
        self.TextClassifier = TextClassification()

    def tfidf_vectorizer(self):
        ngram_range = (1,2)
        min_df = 10
        max_df = 0.85
        max_features = 300000

        tfidf = TfidfVectorizer(encoding='utf-8',
                                ngram_range=ngram_range,
                                stop_words='english',
                                lowercase=True,
                                max_df=max_df,
                                min_df=min_df,
                                max_features=max_features,
                                norm='l2',
                                sublinear_tf=True)

        return tfidf

    def count_vectorizer(self):
        ngram_range = (1,2)
        min_df = 1
        max_df = 0.9
        max_features = 3000

        count = CountVectorizer(encoding='utf-8',
                            ngram_range=ngram_range,
                            stop_words='english',
                            lowercase=True,
                            max_df=max_df,
                            min_df=min_df,
                            max_features=max_features
                            )

        return count

    def bag_of_words(self, X, y):
        tfidf = self.tfidf_vectorizer()
        features_train = tfidf.fit_transform(X).toarray()
        labels_train = y
        categories = np.unique(y)
        grams_list = []
        for category in categories:
            #source: https://towardsdatascience.com/multi-class-text-classification-with-scikit-learn-12f1e60e0a9f
            features_chi2 = chi2(features_train, labels_train == category)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(tfidf.get_feature_names())[indices]
            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
            bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

            grams_list.update({category: {'unigrams':unigrams, 'bigrams':bigrams}})
        return grams_list
   

    def linear_reg_report(self, X, y):
        clf = self.TextClassifier.train_model(X, y, 'LR')
        names = clf.named_steps['vect'].get_feature_names()
        data = {}
        for i, label in enumerate(np.unique(y)):
            coefficent = np.argsort(clf.named_steps['clf'].coef_[i])[-10:]
            names = clf.named_steps['vect'].get_feature_names()
            data[label] = [names[j] for j in coefficent]

        feature_coeff = pd.DataFrame(data)
        print(feature_coeff)

    def decision_tree_report(self, X, y):
        clf = self.TextClassifier.train_model(X, y, 'DT')

        names = clf.named_steps['vect'].get_feature_names()
        importance = clf.named_steps['clf'].feature_importances_

        # for index in np.argsort(-clf.named_steps['clf'].feature_importances_)[:10]:
        #     print(clf.named_steps['vect'].get_feature_names()[index])
        #     print(clf.named_steps['clf'].feature_importances_[index])

        feature_importances = pd.DataFrame(importance,
                                   index = names,
                                    columns=['importance']).sort_values('importance', ascending=False)

        print(feature_importances.head(15))

    def dimenion_reduction(self, model, features, labels, n_components=2, save_graph=False):

        title = "PCA decomposition"
        mod = PCA(n_components=n_components)
 
        tfidf = self.tfidf_vectorizer()
        features = tfidf.fit_transform(features).toarray()
        principal_components = mod.fit_transform(features)

        df_features = pd.DataFrame(data=principal_components,
                         columns=['PC1', 'PC2'])

        df_labels = pd.DataFrame(data=labels,
                                 columns=['label'])

        df_full = pd.concat([df_features, df_labels], axis=1)
        df_full['label'] = df_full['label'].astype(str)

        plt.figure(figsize=(10,10))
        sns.scatterplot(x='PC1',
                        y='PC2',
                        hue=df_full['label'],
                        data=df_full,
                        alpha=.7).set_title(title)
        if save_graph:
            plt.savefig(f'images/decomp-{timestamp}')
        plt.show()

class HyperParameterTuning:
    def __init__(self):
        self.featureSelect = FeatureSelection()
        self.TextClassifier = TextClassification()

    def select_param_tree(self, X, y):
        tfidf = self.featureSelect.tfidf_vectorizer()
        features = tfidf.fit_transform(X, y)
        treeCL = DecisionTreeClassifier(criterion="entropy")
        treeCL.fit(features, y)
        sel = SelectFromModel(treeCL,prefit=True)
        # selected_feat= features.columns[(sel.get_support())]
        # print(len(selected_feat))
        # print(selected_feat)

        print(list(tfidf.vocabulary_.keys())[:10])

    def grid_search(self, wanted_clf, X, y):
        clf = self.TextClassifier.model_pipeline(wanted_clf)
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-2, 1e-3),
        }
        grid_clf = GridSearchCV(clf, parameters, cv=10, iid=False, n_jobs=-1)
        grid_clf.fit(X, y)

        #print(grid_clf.cv_results_)
        print(grid_clf.best_estimator_.get_params())
        return grid_clf.best_estimator_



class ClassificationPipeline:
    '''
    I know, this is not how OO programming works. pls spare me
    '''
    def __init__(self, clf=None, dataset=None, data_size='all', data_unique=True, filepath_prefix=''):
        self.clf = clf
        self.dataset = dataset
        self.data_size = data_size
        self.data_unique = data_unique
        self.filepath = filepath_prefix
        self.DataCleaner = DataCleaner()
        self.DataExploration = DataExploration(filepath=self.filepath)
        self.Features = FeatureSelection()
        self.TextClassifier = TextClassification()
        self.Reports = ClassificationReports(self.data_size, self.data_unique)
        self.Tuning = HyperParameterTuning()

    def exploration(self):
        DataPreProc = DataPreProcessor(self.dataset)
        dataframe = DataPreProc.transformed_df
        dataframe = self.DataCleaner.remove_values(dataframe, 'Unknown')
        if self.data_unique:
            dataframe = self.DataCleaner.unique(dataframe)

        X, y = self.TextClassifier.splitting_dataset(self.data_size, dataframe)
        self.Tuning.grid_search(self.clf, X, y)
        #self.Features.decision_tree_report(X, y)
        #self.Features.select_tree(X, y)
        #self.Features.linear_reg_report(X, y)
        #self.Features.bag_of_words(X, y)
        self.DataExploration.metrics(dataframe, save_graph=True)
        return dataframe

    def cross_validation(self):
        DataPreProc = DataPreProcessor(self.dataset)
        dataframe = DataPreProc.transformed_df
        dataframe = self.DataCleaner.remove_values(dataframe, 'Unknown')
        if self.data_unique:
            dataframe = self.DataCleaner.unique(dataframe)

        if self.data_size == 'all':
            X, y = self.TextClassifier.splitting_dataset(self.data_size, dataframe)
            crossval = CrossValidation()
            report, y_pred = crossval.cross_validation_report(X, y, self.clf, vis=True)
            return y, y_pred

        else:
            text_train, text_test, y_train, y_test, X, y = self.TextClassifier.splitting_dataset(self.data_size, dataframe)
            self.Reports.cross_validation_report(self.clf, text_train, y_train, vis=True)

    def training(self, save_model=False):
        DataPreProc = DataPreProcessor(self.dataset)
        dataframe = DataPreProc.transformed_df
        dataframe = self.DataCleaner.remove_values(dataframe, 'Unknown')
        if self.data_unique:
            dataframe = self.DataCleaner.unique(dataframe)

        if self.data_size == 'all':
            text, y = self.TextClassifier.splitting_dataset(self.data_size, dataframe)
            model = self.TextClassifier.train_model(text, y, self.clf)
            self.Reports.scoring_report(self.clf, model, text, y)

        else:
            text_train, text_test, y_train, y_test, text, y = self.TextClassifier.splitting_dataset(self.data_size, dataframe)
            model = self.TextClassifier.train_model(text_train, y_train, self.clf)
            self.Reports.scoring_report(self.clf, model, text_test, y_test)

        if save_model == True:
            self.TextClassifier.model_save(model, f'{self.clf}-{self.data_size}-{str(self.data_unique)}-{timestamp}')

