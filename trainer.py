import pandas as pd
from elasticsearch import Elasticsearch
from sklearn.model_selection import train_test_split, cross_val_predict, StratifiedKFold
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
np.set_printoptions(threshold=np.inf)
'''
options for pandas
'''
now = datetime.now()
timestamp = now.strftime("%d%m%Y-%H:%M")

pd.set_option("display.max_rows", 25)
pd.set_option('max_columns', 7)
pd.reset_option('max_columns')

def import_elastic():
    es = Elasticsearch(host="127.0.0.1")

    '''
    following code gets data from elasticsearch, removes the keys not needed, retrieves the total_hits, uses the total hits and returns a list of dicts.
    '''
    exclude = ['id', 'root_url', 'country', 'config_name']

    totalhits = es.search(index='scrapy_test-early_mornin_4',_source='false', body={})['hits']['total']['value']
    all_documents = es.search(index='scrapy_test-early_mornin_4',body={})['hits']['hits']
    documents_exclude = es.search(index='scrapy_test-early_mornin_4',body={}, _source_excludes=exclude, size=totalhits)['hits']['hits']

    documents_exclude = [source['_source'] for source in documents_exclude]
    return documents_exclude


class TextClassification:


    def to_dataframe(self, documents):
        '''
        make a dataframe from the list of dicts
        '''
        return pd.DataFrame(documents)

    def transform_dataframe(self, dataframe):
        '''
        transform the data from a multidimensional array to a 2d array, with ['label', 'text'] as columns
        '''
        print()
        self.column_names = dataframe.columns.values
        return pd.melt(dataframe, value_vars=self.column_names, var_name='label', value_name='text',)

    def remove_values(self, dataframe, value):
        '''
        The data has some 'unknown' values, as where the scrapetool didnt find any data for that category
        3997 rows contained an unknown value. introduction(2898), contract_type(1066), location(33)
        '''
        unknowns_df = dataframe[dataframe.text == value]
        #self.metrics(dataframe)
        print(
        f'''
        removing missing values from the dataset
         ''')
        cleaned_df = dataframe[~dataframe.text.str.contains(value)]
        if cleaned_df[cleaned_df.text=='Unknown'].shape[0] == 0: print(f"all '{value}' removed")
        print(f'Removed {dataframe.shape[0] - cleaned_df.shape[0]} {value}')
        return cleaned_df

    def unique(self, dataframe):
        #print(dataframe['text'].shape[0])
        #print(dataframe['text'].unique().shape[0])
        dropped_df = dataframe.drop_duplicates()
        #print(dataframe.groupby('label').nunique())
        #print(dataframe['text'].duplicated(keep='first').sum())
        #print(dataframe.loc[dataframe.duplicated(keep='first'), :])
        return dropped_df

    def add_values(self, dataframe):
        dataframe.append({'label': 'location', 'text':'1234 BV'}, ignore_index=True)
        #print(dataframe.tail(5))
        return dataframe

    def metrics(self, dataframe):
        '''
        Some metrics:
        - How many observations per label?
        - how many observations in total
        '''
        print(f"Observations per label: \n{dataframe['label'].value_counts()}")
        print(f"Total Observations: {dataframe.shape[0]}")


    def splitting_dataset(self, dataframe):
        '''
        Splitting the test and train dataset
        '''
        text = dataframe['text'].values
        y = dataframe['label'].values

        text_train, text_test, y_train, y_test = train_test_split(
                text, y, test_size=0.25, random_state=2)

        return text_train, text_test, y_train, y_test, text, y




    def train_model(self, text_train, y_train, wanted_clf):
        '''
        training pipeline
        '''
        classifiers = {
                  'NB': MultinomialNB(),
                  'SVM': SVC(C=1.0, kernel='linear', degree=3, gamma='auto'),
                  'SGD': SGDClassifier(),
                  'LR': LogisticRegression()
                  }


        model = Pipeline([('vect', CountVectorizer()),
                       ('tfidf', TfidfTransformer()),
                       ('clf', classifiers[wanted_clf]),
                      ])

        model.fit(text_train, y_train)
        #print(model.get_support())
        #y_pred = model.predict(text_test)
        return model

    def features(self, X, y):
        SGD = SGDClassifier().fit(X, y)
        model = sk.SelectFromModel(SGD, prefit=True)
        X_new = model.transform(X)
        print(X.columns[model.get_support()])

    def cross_validation(self, X, y, wanted_clf):
        #self.features(X, y)
        skf = StratifiedKFold(n_splits=10)
        clf = self.train_model(X, y, wanted_clf)
        scores = cross_val_score(clf, X, y, cv=skf)
        y_pred = cross_val_predict(clf, X, y, cv=skf)
        return scores, y_pred

    def predictions(self, model, text_test):
        predictions = model.predict(text_test)
        return predictions

    def scoring(self, y_test, y_pred):
        accuracy = 'accuracy %s' % accuracy_score(y_pred, y_test)
        columns = self.column_names
        report = classification_report(y_test, y_pred,target_names=columns)
        return accuracy, report


    def kijkdoos(self, text_test, y_test, y_pred):
        d = {'Text': text_test, 'Label': y_test, 'Predicted': y_pred}
        df = pd.DataFrame(d)
        df_error = df[df.Label != df.Predicted]
        print(df_error.sample(n=20))
        print(df_error[df_error.Label == 'contract_type'])

    def visualisations(self, title, y_test, y_pred):
        conf_mat = confusion_matrix(y_test, y_pred)
        #print(conf_mat)
        columns = self.column_names
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=columns, yticklabels=columns)
        plt.title(title, fontdict=None, loc='center', pad=None)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        #plt.savefig(f'figure-{timestamp}.png')
        plt.show()
        plt.close()

    def model_save(self, clf, filename):
        dump(clf, f'trained_models/{filename}.joblib')



class ClassificationPipeline:
    def __init__(self, clf, dataset):
        self.clf = clf
        self.dataset = dataset
        self.textclassifier = TextClassification()

    def data_collection(self, dataset):
        #documents = import_elastic()
        dataframe = self.textclassifier.to_dataframe(dataset)
        return dataframe
        #print(dataframe[dataframe.duplicated()])

    def data_cleaning(self, dataframe, data_unique):
        dataframe = self.textclassifier.transform_dataframe(dataframe)
        dataframe = self.textclassifier.add_values(dataframe)
        #print(dataframe[dataframe.text == '1234 BV'])
        dataframe = self.textclassifier.remove_values(dataframe, 'Unknown')
        if data_unique:
            dataframe = self.textclassifier.unique(dataframe)

        self.textclassifier.metrics(dataframe)
        return dataframe

    def cross_validation_report(self, clf, X, y, data_size, data_unique):
        scores, y_pred = self.textclassifier.cross_validation(X, y, clf)
        mean = scores.mean()
        accuracy, report = self.textclassifier.scoring(y, y_pred)
        print(f'''
        Cross Validation on trained model\n
        classifier: {clf}
        Mean score: {mean}
        {accuracy}
        Data: {data_size}
        Unique: {str(data_unique)}\n
        \n{report}
        ''')
        title = f'Cross Validation: {clf} - Data: {data_size} Unique: {str(data_unique)}'
        self.textclassifier.visualisations(title, y, y_pred)
        self.textclassifier.kijkdoos(X,y,y_pred)


    def scoring_report(self, clf, model, text_test, y_test):
        predictions_list = self.textclassifier.predictions(model, text_test)
        accuracy, report = self.textclassifier.scoring(y_test, predictions_list)
        print(f'''
        Trained model tested on test data:\n\n
        \tclassifier: {clf} \n
        \t{accuracy}\n
        \n{report}
        ''')
        #self.textclassifier.kijkdoos(text_test,y_test,predictions_list)

    def training(self, save_model=False, data_size='all', data_unique=True):
        dataframe = self.data_collection(dataset)
        #dataframe.to_json('vacancy_data.json', orient='records')
        cleaned_dataframe = self.data_cleaning(dataframe, data_unique)
        text_train, text_test, y_train, y_test, text, y = self.textclassifier.splitting_dataset(cleaned_dataframe)
        if data_size == 'all':
            self.cross_validation_report(self.clf, text, y, data_size, data_unique)
            model = self.textclassifier.train_model(text, y, self.clf)

        else:
            self.cross_validation_report(self.clf, text_train, y_train, data_size, data_unique)
            model = self.textclassifier.train_model(text_train, y_train, self.clf)
            self.scoring_report(self.clf, model, text_test, y_test, data_size, data_unique)

        if save_model == True:
            self.textclassifier.model_save(model, f'model-{self.clf}-{data_size}-{str(data_unique)}-{timestamp}')





dataset = import_elastic()
pipeline = ClassificationPipeline("SGD", dataset)
pipeline.training(data_size='all', data_unique=True)
#pipeline.training(data_unique=False, save_model=True)


data_size = ['all', 'train_test']
unique = ['True', 'False']

#data = pd.read_csv("./vacancy_items_4.csv", encoding='latin-1').sample(frac=1).drop_duplicates() ### Hmm?


# data['label'] = '__label__' + data['label'].astype(str)
# data.iloc[0:int(len(data)*0.8)].to_csv('vacancy_items_4_train.csv', sep='\t', index = False, header = False)
# data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('vacancy_items_4_dev.csv', sep='\t', index = False, header = False)
# data.iloc[int(len(data)*0.9):].to_csv('vacancy_items_4_test.csv', sep='\t', index = False, header = False);



'''
TODO:

- zon leuk modelletje en grafiekje
- flair, fasttext

Using error to see if i need to leave out features

testing on new websites,
scores etc
'''
