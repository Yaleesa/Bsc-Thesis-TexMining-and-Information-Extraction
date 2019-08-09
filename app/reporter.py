from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from explorer import DataExploration
from preprocessor import DataPreProcessor, DataCleaner
from trainer import CrossValidation

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

class ClassificationReports:
    def __init__(self, data_size='all', data_unique=True, filepath=''):
        self.data_size = data_size
        self.data_unique = data_unique
        self.filepath = filepath
        self.DataExploration = DataExploration(filepath=self.filepath)
        self.CrossValidation = CrossValidation()
        
    def predictions(self, model, text_test):
        predictions = model.predict(text_test)
        return predictions

    def scoring(self, y_test, y_pred):
        accuracy = 'accuracy %s' % accuracy_score(y_pred, y_test)
        columns = np.unique(y_test)
        report = classification_report(y_test, y_pred,target_names=columns)
        return accuracy, report

    def cross_validation_report(self, clf, X, y, folds=10, kijkdoos=False, vis=False):
        scores, y_pred = self.CrossValidation.cross_validation(X, y, clf)
        mean = scores.mean()
        accuracy, report = self.scoring(y, y_pred)
        print(f'''
        Cross Validation - {str(folds)} folds
        \n
        classifier: {clf}
        Mean score: {mean}
        {accuracy}
        Data: {self.data_size}
        Unique: {str(self.data_unique)}\n
        \n{report}
        ''')
        title = f'Cross Validation: {clf} - Data: {self.data_size} Unique: {str(self.data_unique)}'
        if vis:
           self.confusion_matrix_vis(title, y, y_pred)
        if kijkdoos:
            self.DataExploration.kijkdoos(X, y, y_pred, 'Title')

    def scoring_report(self, clf, model, X, y):
        y_pred = self.predictions(model, X)
        accuracy, report = self.scoring(y, y_pred)
        print(f'''
        Trained model tested on test data:\n\n
        \tclassifier: {clf} \n
        \t{accuracy}\n
        \n{report}
        ''')
        self.confusion_matrix_vis(f'{clf}', y, y_pred)
        self.DataExploration.kijkdoos(X, y, y_pred, 'location')
        return accuracy

    def confusion_matrix_vis_old(self, title, y_test, y_pred, save_graph=True):
        conf_mat = confusion_matrix(y_test, y_pred)
        columns = np.unique(y_test)
        plt.subplots(figsize=(10,6))
        sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=columns, yticklabels=columns, linewidths=.2, cmap=sns.cm.rocket_r)
        plt.title(title, fontdict=None, loc='center', pad=None)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        if save_graph:
            plt.savefig(f'matrix-{title}-{timestamp}.png', format='png')
        plt.close()

    def confusion_matrix_vis(self, y, y_pred, filepath, title, figsize=(12,13)):
        cm = confusion_matrix(y, y_pred, labels=np.unique(y))
        cm_sum = np.sum(cm, axis=1, keepdims=True)
        cm_perc = cm / cm_sum.astype(float) * 100
        annot = np.empty_like(cm).astype(str)
        nrows, ncols = cm.shape
        for i in range(nrows):
            for j in range(ncols):
                c = cm[i, j]
                p = cm_perc[i, j]
                if i == j:
                    s = cm_sum[i]
                    annot[i, j] = '%.1f%%\n%d/%d' % (p, c, s)
                elif c == 0:
                    annot[i, j] = ''
                else:
                    annot[i, j] = '%.1f%%\n%d' % (p, c)
        cm = pd.DataFrame(cm_perc, index=np.unique(y_true), columns=np.unique(y_true))
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        fig, ax = plt.subplots(figsize=figsize)
        plt.title(title, fontdict=None, loc='center', pad=None)
        sns.heatmap(cm, cmap= sns.cubehelix_palette(8, start=.5, rot=-.75), annot=annot, fmt='', ax=ax, vmax=100, annot_kws={"fontsize":12})
        plt.savefig(f'matrix-{title}-{timestamp}.png', format='png')