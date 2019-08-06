from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Pandas & Numpy & vis
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

np.set_printoptions(threshold=np.inf)
pd.set_option("display.max_rows", 25)
pd.set_option('max_columns', 7)
pd.reset_option('max_columns')

# globals
import os
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%d%m%Y-%H:%M")


class DataExploration:
    def __init__(self, filepath):
        self.filepath = filepath

    def metrics(self, dataframe, save_graph=False):
        '''
        Some metrics:
        - How many observations per label?
        - how many observations in total
        '''
        metrics = dataframe['label'].value_counts()
        plt.figure(figsize=(15,10))
        sns.barplot(x=metrics.index, y=metrics, palette="ch:.25")
        if save_graph:
            plt.savefig(f'{self.filepath}/metrics-{timestamp}.png', format='png', dpi=1200)
        plt.close()

        print(f"Observations per label: \n{metrics}")
        print(f"Total Observations: {dataframe.shape[0]}")

    def length_distribution(self, dataframe, save_graph=False):
        dataframe['text_length'] = dataframe['text'].str.len()

        plt.figure(figsize=(12.8,6))
        sns.boxplot(data=dataframe, x='label', y='text_length', width=.5)
        if save_graph:
            plt.savefig(f'{self.filepath}/length_box-{timestamp}.png', format='png', dpi=1200)
        plt.show()


    def kijkdoos(self, text_test, y_test, y_pred, label):
        d = {'Text': text_test, 'Label': y_test, 'Predicted': y_pred}
        df = pd.DataFrame(d)
        df_error = df[df.Label != df.Predicted]
        df_good = df[df.Label == df.Predicted]
        print(df_error.sample(n=20))
        print(df_error[df_error.Label == label])
        print(df_good.sample(n=20))
        print(df_good[df_good.Label == label])

    def confusion_matrix_vis(self, title, y_test, y_pred, save_graph=True):
        conf_mat = confusion_matrix(y_test, y_pred)
        columns = np.unique(y_test)
        plt.subplots(figsize=(10,6))
        sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=columns, yticklabels=columns, linewidths=.2, cmap=sns.cm.rocket_r)
        plt.title(title, fontdict=None, loc='center', pad=None)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        if save_graph:
            plt.savefig(f'matrix-{title}-{timestamp}.png', format='png', dpi=1200)
        plt.close()

    def plot_cm(self, title, y_true, y_pred, figsize=(12,13)):
        cm = confusion_matrix(y_true, y_pred, labels=np.unique(y_true))
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
        plt.savefig(f'matrix_perc-{title}-{timestamp}.png', format='png', dpi=1200)
