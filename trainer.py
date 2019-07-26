import pandas as pd
from elasticsearch import Elasticsearch
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.feature_selection import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
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
    #all_documents = es.search(index='scrapy_test-early_mornin_4',body={})['hits']['hits']
    documents_exclude = es.search(index='scrapy_test-early_mornin_4',body={}, _source_excludes=exclude, size=totalhits)['hits']['hits']

    documents_exclude = [source['_source'] for source in documents_exclude]
    return documents_exclude

class DataPreProcessor:
    def to_dataframe(self, documents):
       '''
       make a dataframe from the list of dicts
       '''
       return pd.DataFrame(documents)

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
    # def __init__(dataframe, values_to_remove=[], cats_to_remove=[], unique=True):
    #     self.dataframe = dataframe
    #     self.values_to_remove = values_to_remove
    #     self.cats_to_remove = cats_to_remove
    #     self.unique = unique

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

    def add_values(self, dataframe):
        dataframe.append({'label': 'location', 'text':'1234 BV'}, ignore_index=True)
        #print(dataframe.tail(5))
        return dataframe


class DataExploration:
    def metrics(self, dataframe, save_graph=False):
        '''
        Some metrics:
        - How many observations per label?
        - how many observations in total
        '''
        metrics = dataframe['label'].value_counts()
        plt.figure(figsize=(10,10))
        sns.barplot(x=metrics.index, y=metrics, palette="ch:.25")
        if save_graph:
            plt.savefig(f'images/metrics-{timestamp}.png', format='png', dpi=1200)

        print(f"Observations per label: \n{metrics}")
        print(f"Total Observations: {dataframe.shape[0]}")

    def length_distribution(self, dataframe):
        dataframe['text_length'] = dataframe['text'].str.len()
        locations = dataframe[dataframe['label'] == 'location']
        print(locations[locations['text_length'] > 30])

        quantile_95 = dataframe['text_length'].quantile(0.95)
        df_95 = dataframe[dataframe['text_length'] < quantile_95]
        plt.figure(figsize=(12.8,6))
        #sns.distplot(df_95['text_length']).set_title('vacancy length distribution');
        sns.boxplot(data=dataframe, x='label', y='text_length', width=.5)
        if save_graph:
            plt.savefig(f'images/length_box-{timestamp}.png', format='png', dpi=1200)
        plt.show()


    def kijkdoos(self, text_test, y_test, y_pred, label):
        d = {'Text': text_test, 'Label': y_test, 'Predicted': y_pred}
        df = pd.DataFrame(d)
        df_error = df[df.Label != df.Predicted]
        print(df_error.sample(n=20))
        print(df_error[df_error.Label == label])

    def confusion_matrix_vis(self, title, y_test, y_pred):
        conf_mat = confusion_matrix(y_test, y_pred)
        #print(conf_mat)
        columns = self.column_names
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=columns, yticklabels=columns)
        plt.title(title, fontdict=None, loc='center', pad=None)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        if save_graph: 
            plt.savefig(f'images/matrix-{timestamp}.png')
        plt.show()
        plt.close()


class TextClassification:

    def splitting_dataset(self, dataframe):
        '''
        Splitting the test and train dataset
        '''
        text = dataframe['text'].values
        y = dataframe['label'].values

        text_train, text_test, y_train, y_test = train_test_split(
                text, y, test_size=0.25, random_state=2)

        return text_train, text_test, y_train, y_test, text, y


    def model_pipeline(self, wanted_clf):
        '''
        training pipeline
        '''
        classifiers = {
                  'NB': MultinomialNB(),
                  'SVM': SVC(C=1.0, kernel='linear', degree=3, gamma='auto'),
                  'SGD': SGDClassifier(),
                  'LR': LogisticRegression(multi_class='auto'),
                  'DT': DecisionTreeClassifier(),
                  'RF': RandomForestClassifier()
                  }


        model = Pipeline([('vect', CountVectorizer(stop_words='english')),
                       ('tfidf', TfidfTransformer()),
                       ('clf', classifiers[wanted_clf]),
                      ])

        return model

    def train_model(self, text_train, y_train, wanted_clf):
        model = self.model_pipeline(wanted_clf)
        model.fit(text_train, y_train)
        return model

    def grid_search(self, X, y, wanted_clf):
        clf = self.train_model(X, y, wanted_clf)
        parameters = {
            'vect__ngram_range': [(1, 1), (1, 2)],
            'tfidf__use_idf': (True, False),
            'clf__alpha': (1e-2, 1e-3),
        }
        grid_clf = GridSearchCV(clf, parameters, cv=10, iid=False, n_jobs=-1)
        grid_clf.fit(X, y)

        print(grid_clf.best_score_)
        print(grid_clf.cv_results_)
        print(grid_clf.best_estimator_.get_params())

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
    

class Features: ### clean this shit up
    def __init__(self):
        self.TextClassifier = TextClassification()

    def tfidf_vectorizer(self):
        ngram_range = (1,2)
        min_df = 1
        max_df = 1.
        max_features = 300

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
        max_df = 1.
        max_features = 300

        count = CountVectorizer(encoding='utf-8',
                            ngram_range=ngram_range,
                            stop_words=None,
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
        print(features_train.shape)
        category = y[0]

        features_chi2 = chi2(features_train, labels_train == category)
        indices = np.argsort(features_chi2[0])
        feature_names = np.array(tfidf.get_feature_names())[indices]
        unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
        bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
        print(f"'{category}'\n")
        print("unigrams:\n. {}".format('\n. '.join(unigrams[-5:])))
        print("bigrams:\n. {}".format('\n. '.join(bigrams[-3:])))
        print("")

    def linear_reg_report(self, X, y):
        clf = self.TextClassifier.train_model(X, y, 'LR')
        names = clf.named_steps['vect'].get_feature_names()
        data = {}
        for i, label in enumerate(np.unique(y)):
            coefficent = np.argsort(clf.named_steps['clf'].coef_[i])[-10:]
            names = clf.named_steps['vect'].get_feature_names()
            #print(label, names,  coefficent)

            data[label] = [names[j] for j in coefficent]

        feature_coeff = pd.DataFrame(data)
        print(feature_coeff)


    def decision_tree_report(self, X, y):
        clf = self.TextClassifier.train_model(X, y, 'DT')

        names = clf.named_steps['vect'].get_feature_names()
        #print(clf.named_steps['vect'].vocabulary_.keys())
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
            plt.save_fig(f'images/decomp-{timestamp}')
        plt.show()

class ClassificationReports:
    def __init__(self, data_size='all', data_unique=True):
        self.data_size = data_size
        self.data_unique = data_unique
        self.TextClassifier = TextClassification()
        self.DataExploration = DataExploration()
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
            self.DataExploration.confusion_matrix_vis(title, y, y_pred)
        if kijkdoos:
            self.DataExploration.kijkdoos(X, y, y_pred, 'Title')

    def scoring_report(self, clf, model, text_test, y_test):
        predictions_list = self.predictions(model, text_test)
        accuracy, report = self.scoring(y_test, predictions_list)
        print(f'''
        Trained model tested on test data:\n\n
        \tclassifier: {clf} \n
        \t{accuracy}\n
        \n{report}
        ''')

class ClassificationPipeline:
    def __init__(self, clf=None, dataset=None, data_size='all', data_unique=True):
        self.clf = clf
        self.dataset = dataset
        self.data_size = data_size
        self.data_unique = data_unique
        self.DataPreProcessor = DataPreProcessor()
        self.DataCleaner = DataCleaner()
        self.DataExploration = DataExploration()
        self.Features = Features()
        self.TextClassifier = TextClassification()
        self.Reports = ClassificationReports(self.data_size, self.data_unique)

    def exploration(self):
        dataframe = self.DataPreProcessor.to_dataframe(self.dataset)
        #dataframe = self.DataPreProcessor.remove_categories(dataframe,['location', 'contract_type', 'job_category', 'company_name'])
        #dataframe = self.DataPreProcessor.remove_categories(dataframe,['JobLocation', 'JobRequirements', 'JobCompany'])
        #dataframe = self.DataPreProcessor.remove_categories(dataframe, ['description', 'introduction', 'vacancy_title'])

        dataframe = self.DataPreProcessor.transform_dataframe(dataframe)
        dataframe = self.DataCleaner.remove_values(dataframe, 'Unknown')
        if self.data_unique:
            dataframe = self.DataCleaner.unique(dataframe)

        text_train, text_test, y_train, y_test, X, y = self.TextClassifier.splitting_dataset(dataframe)
        self.Features.decision_tree_report(X, y)
        self.Features.linear_reg_report(X, y)
        self.Features.bag_of_words(X, y)
        self.DataExploration.metrics(dataframe)
        return dataframe

    def cross_validation(self):
        dataframe = self.DataPreProcessor.to_dataframe(self.dataset)
        dataframe = self.DataPreProcessor.remove_categories(dataframe,['location', 'contract_type', 'job_category', 'company_name'])
        #dataframe = self.DataPreProcessor.remove_categories(dataframe,['JobLocation', 'JobRequirements', 'JobCompany'])
        #dataframe = self.DataPreProcessor.remove_categories(dataframe, ['description', 'introduction', 'vacancy_title'])

        dataframe = self.DataPreProcessor.transform_dataframe(dataframe)
        dataframe = self.DataCleaner.remove_values(dataframe, 'Unknown')
        if self.data_unique:
            dataframe = self.DataCleaner.unique(dataframe)

        text_train, text_test, y_train, y_test, X, y = self.TextClassifier.splitting_dataset(dataframe)

        if self.data_size == 'all':
            self.Reports.cross_validation_report(self.clf, X, y)

        else:
            self.Reports.cross_validation_report(self.clf, text_train, y_train)

    def training(self, save_model=False):
        dataframe = self.DataPreProcessor.to_dataframe(self.dataset)
        #dataframe = self.DataPreProcessor.remove_categories(dataframe,['location', 'contract_type', 'job_category', 'company_name'])
        #dataframe = self.DataPreProcessor.remove_categories(dataframe,['JobLocation', 'JobRequirements', 'JobCompany'])
        #dataframe = self.DataPreProcessor.remove_categories(dataframe, ['description', 'introduction', 'vacancy_title'])

        dataframe = self.DataPreProcessor.transform_dataframe(dataframe)
        dataframe = self.DataCleaner.remove_values(dataframe, 'Unknown')
        if self.data_unique:
            dataframe = self.DataCleaner.unique(dataframe)

        text_train, text_test, y_train, y_test, text, y = self.TextClassifier.splitting_dataset(dataframe)

        if self.data_size == 'all':
            model = self.TextClassifier.train_model(text, y, self.clf)
            self.Reports.scoring_report(self.clf, model, text_test, y_test)

        else:
            model = self.TextClassifier.train_model(text_train, y_train, self.clf)
            self.Reports.scoring_report(self.clf, model, text_test, y_test)

        if save_model == True:
            self.TextClassifier.model_save(model, f'model-{self.clf}-{self.data_size}-{str(self.data_unique)}-{timestamp}')

if __name__ == '__main__':
    elastic_dataset = import_elastic()
    xml_dataset = pd.read_json("data/columns_xml_data.json", orient="records")

    pipeline = ClassificationPipeline(clf="SGD", dataset=xml_dataset, data_size='meh')
    #pipeline.training(data_size='all', data_unique=True)
    pipeline.exploration()
    #pipeline.cross_validation()
    #pipeline.training()

#dataframe.to_json('vacancy_data.json', orient='records')

# data['label'] = '__label__' + data['label'].astype(str)
# data.iloc[0:int(len(data)*0.8)].to_csv('vacancy_items_4_train.csv', sep='\t', index = False, header = False)
# data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('vacancy_items_4_dev.csv', sep='\t', index = False, header = False)
# data.iloc[int(len(data)*0.9):].to_csv('vacancy_items_4_test.csv', sep='\t', index = False, header = False);
