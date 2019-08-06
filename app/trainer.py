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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
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
                    text, y, test_size=0.25, random_state=2)

            return text_train, text_test, y_train, y_test, text, y

    def model_pipeline(self, wanted_clf):
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

        return model

    def train_model(self, text_train, y_train, wanted_clf):
        model = self.model_pipeline(wanted_clf)
        model.fit(text_train, y_train)
        return model

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
        print(features_train)
        categories = np.unique(y)
        for category in categories:

            features_chi2 = chi2(features_train, labels_train == category)
            indices = np.argsort(features_chi2[0])
            feature_names = np.array(tfidf.get_feature_names())[indices]
            unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
            bigrams = [v for v in feature_names if len(v.split(' ')) == 2]

            #unit = {category: {'unigrams':unigrams, 'bigrams':bigrams}}
            #print(unit)
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

    def grid_search(self, X, y, wanted_clf):
        clf = self.TextClassifier.model_pipeline(wanted_clf)
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

class ClassificationReports:
    def __init__(self, data_size='all', data_unique=True, filepath=''):
        self.data_size = data_size
        self.data_unique = data_unique
        self.filepath = filepath
        self.TextClassifier = TextClassification()
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
            #self.DataExploration.confusion_matrix_vis(title, y, y_pred, save_graph=True)
            self.DataExploration.plot_cm(title, y, y_pred)
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
        self.DataExploration.plot_cm(f'wild - data - {clf}', y, y_pred)
        self.DataExploration.kijkdoos(X, y, y_pred, 'location')

class ClassificationPipeline:
    def __init__(self, clf=None, dataset=None, data_size='all', data_unique=True, filepath_prefix=''):
        self.clf = clf
        self.dataset = dataset
        self.data_size = data_size
        self.data_unique = data_unique
        self.filepath = filepath_prefix
        self.DataPreProcessor = DataPreProcessor()
        self.DataCleaner = DataCleaner()
        self.DataExploration = DataExploration(filepath=self.filepath)
        self.Features = FeatureSelection()
        self.TextClassifier = TextClassification()
        self.Reports = ClassificationReports(self.data_size, self.data_unique)

    def exploration(self):
        dataframe = self.DataPreProcessor.to_dataframe(self.dataset)
        #dataframe = self.DataPreProcessor.remove_categories(dataframe,['JobLocation', 'JobRequirements', 'JobCompany'])

        dataframe = self.DataPreProcessor.transform_dataframe(dataframe)
        dataframe = self.DataCleaner.remove_values(dataframe, 'Unknown')
        if self.data_unique:
            dataframe = self.DataCleaner.unique(dataframe)

        X, y = self.TextClassifier.splitting_dataset(self.data_size, dataframe)
        #self.Features.decision_tree_report(X, y)
        #self.Features.select_tree(X, y)
        #self.Features.linear_reg_report(X, y)
        #self.Features.bag_of_words(X, y)
        self.DataExploration.metrics(dataframe, save_graph=True)
        return dataframe

    def cross_validation(self):
        dataframe = self.DataPreProcessor.to_dataframe(self.dataset)
        dataframe = self.DataPreProcessor.transform_dataframe(dataframe)
        dataframe = self.DataCleaner.remove_values(dataframe, 'Unknown')
        if self.data_unique:
            dataframe = self.DataCleaner.unique(dataframe)

        if self.data_size == 'all':
            X, y = self.TextClassifier.splitting_dataset(self.data_size, dataframe)
            self.Reports.cross_validation_report(self.clf, X, y, vis=True)

        else:
            text_train, text_test, y_train, y_test, X, y = self.TextClassifier.splitting_dataset(self.data_size, dataframe)
            self.Reports.cross_validation_report(self.clf, text_train, y_train, vis=True)

    def training(self, save_model=False):
        dataframe = self.DataPreProcessor.to_dataframe(self.dataset)
        dataframe = self.DataPreProcessor.transform_dataframe(dataframe)
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

if __name__ == '__main__':
    es = Elasticer()
    
    dataset_scraped = 'scrapy_test-early_mornin_4'
    include_scraped = ['company_name', 'introduction', 'location', 'vacancy_title', 'description', 'job_category', 'contract_type']

    dataset_xml = 'sj-uk-vacancies-scraped-cleaned-3'
    exclude_xml = ['@language', 'DatePlaced', 'Id', 'companyLogo', 'country', 'topjob', 'HoursWeek', 'JobUrl', 'JobMinDaysPerWeek', 'JobParttime', 'JobCompanyBranch', 'JobCompanyProfile', 'JobRequirements.MinAge']
    include_xml = ['JobBranch', 'JobCategory', 'JobCompany', 'JobDescription','JobLocation.LocationRegion', 'JobProfession', 'Title','TitleDescription', 'functionTitle', 'postalCode', 'profession']
    
    dataset_name = dataset_scraped
    dataset = es.import_dataset(dataset_scraped, include_scraped)
    filepath_prefix = f'pipeline-reports/{dataset_name}/{timestamp}'
    os.makedirs(filepath_prefix, exist_ok=True) 
    list_of_clf = ['NB', 'SVM', 'LR', 'SGD']
    for clf in list_of_clf:
        pipeline = ClassificationPipeline(clf=clf, dataset=dataset, data_size='all', filepath_prefix=filepath_prefix)
        #pipeline.exploration()
        #pipeline.cross_validation()
        pipeline.training(save_model=True)



#SNIPPETS?

#dataframe.to_json('vacancy_data.json', orient='records')

# data['label'] = '__label__' + data['label'].astype(str)
# data.iloc[0:int(len(data)*0.8)].to_csv('vacancy_items_4_train.csv', sep='\t', index = False, header = False)
# data.iloc[int(len(data)*0.8):int(len(data)*0.9)].to_csv('vacancy_items_4_dev.csv', sep='\t', index = False, header = False)
# data.iloc[int(len(data)*0.9):].to_csv('vacancy_items_4_test.csv', sep='\t', index = False, header = False);
