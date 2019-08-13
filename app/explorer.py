'''
Author: Yaleesa Borgman
Date: 8-8-2019
explorer.py - exploration of data
'''
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
    def __init__(self, filepath=''):
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


