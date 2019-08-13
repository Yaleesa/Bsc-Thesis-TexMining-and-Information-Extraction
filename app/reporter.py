'''
Author: Yaleesa Borgman
Date: 8-8-2019
reporter.py - handles output reports and visualisations
'''
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from explorer import DataExploration
from preprocessor import DataPreProcessor, DataCleaner

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
    def __init__(self, title='', data_size='all', data_unique=True, filepath='../data/pipeline-reports/'):
        self.data_size = data_size
        self.data_unique = data_unique
        self.filepath = filepath
        self.title = title
        self.DataExploration = DataExploration(filepath=self.filepath)

    def scoring(self, y_test, y_pred):
        accuracy = 'accuracy %s' % accuracy_score(y_pred, y_test)
        columns = np.unique(y_test)
        report = classification_report(y_test, y_pred,target_names=columns)
        return accuracy, report

    def cv_report(self, clf, scores, y, y_pred, folds=10, kijkdoos=False, vis=True):
        mean = scores.mean()
        accuracy, report = self.scoring(y, y_pred)
        report = f'''
        Cross Validation - {str(folds)} folds
        \n
        classifier: {clf}
        Mean score: {mean}
        {accuracy}
        Data: {self.data_size}
        Unique: {str(self.data_unique)}\n
        \n{report}
        '''
        title = f'Cross Validation: {clf} - Data: {self.data_size} Unique: {str(self.data_unique)}'
        if vis:
           self.confusion_matrix_vis(y, y_pred, '', title)
        if kijkdoos:
            self.DataExploration.kijkdoos(X, y, y_pred, 'Title')
        return report 
    
    def scoring_report(self, clf, y_pred, X, y):    
        accuracy, report = self.scoring(y, y_pred)
        print(f'''
        Trained model tested on test data:\n\n
        \tclassifier: {clf} \n
        \t{accuracy}\n
        \n{report}
        ''')
        self.confusion_matrix_vis(y, y_pred, f'{clf}')
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
        cm = pd.DataFrame(cm_perc, index=np.unique(y), columns=np.unique(y))
        cm.index = cm.index.map(lambda x: str(x)[9:])
        cm.columns = cm.columns.map(lambda x: str(x)[9:])
        cm.index.name = 'Actual'
        cm.columns.name = 'Predicted'
        fig, ax = plt.subplots(figsize=figsize)
        plt.title(title, fontdict=None, loc='center', pad=None)
        sns.heatmap(cm, cmap= sns.cubehelix_palette(8, start=.5, rot=-.75), annot=annot, fmt='', ax=ax, vmax=100, annot_kws={"fontsize":12})
        plt.savefig(f'{filepath}/matrix-{title}-{timestamp}.png', format='png')