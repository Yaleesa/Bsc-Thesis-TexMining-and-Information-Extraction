import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from nltk import word_tokenize
from nltk.corpus import stopwords

stop_words = set(stopwords.words('english'))

class DataPreProcessor:
    def __init__(self, data):
        self.data = data
        self.dataframe = self.to_dataframe(self.data)
        self.transformed_df = self.transform_dataframe(self.dataframe)

    def to_dataframe(self, documents):
        '''
        make a dataframe from the list of dicts
        '''
        dataframe = pd.io.json.json_normalize(documents, sep='.')
        return dataframe

    def transform_dataframe(self, dataframe):
        '''
        transform the data from a multidimensional array to a 2d array, with ['label', 'text'] as columns
        '''

        dataframe = dataframe.astype(str)

        self.column_names = dataframe.columns.values
        return pd.melt(dataframe, value_vars=self.column_names, var_name='label', value_name='text',)

    def remove_categories(self, dataframe, list_of_cats):
        dataframe = dataframe.drop(list_of_cats ,axis=1)
        return dataframe


class DataCleaner:
    def remove_values(self, dataframe, value):
        '''
        The data has some 'unknown' values, as where the scrapetool didnt find any data for that category
        3997 rows contained an unknown value. introduction(2898), contract_type(1066), location(33)
        '''
        print(
        f'''---> removing missing values from the dataset
        ''')
        cleaned_df = dataframe[~dataframe.text.str.contains(value)]
        if cleaned_df[cleaned_df.text=='Unknown'].shape[0] == 0: print(f"   ---> all '{value}' removed")
        print(f'   ---> Removed {dataframe.shape[0] - cleaned_df.shape[0]} {value}\n')
        return cleaned_df

    def remove_not_null(self, dataframe):
        dataframe = dataframe[dataframe.notnull()]

    def unique(self, dataframe):
        dropped_df = dataframe.drop_duplicates()
        return dropped_df

    def lowercase(self, dataframe):
        dataframe['text'] = dataframe['text'].apply(lambda row:' '.join([w.lower() for w in word_tokenize(row)]) )
        return dataframe

    def remove_stopwords(self, dataframe):
        dataframe['text'] = dataframe['text'].apply(lambda row:' '.join([w for w in word_tokenize(row) if w not in stop_words]) )
        return dataframe
