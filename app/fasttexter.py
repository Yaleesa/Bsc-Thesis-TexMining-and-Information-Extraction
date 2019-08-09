from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

from elasticer import Elasticer
from explorer import DataExploration
from preprocessor import DataPreProcessor, DataCleaner
from predicter import xmlRemapper
from reporter import ClassificationReports
import pandas as pd
import numpy as np
import fasttext
import nltk
from nltk import word_tokenize

import os, csv
from datetime import datetime
now = datetime.now()
timestamp = now.strftime("%d%m%Y-%H:%M")



class FastTexter:
    def __init__(self, name):
        self.name = name
        self.filepath = '../data/trained_models/fasttext_models'
        self.model = None
        self.trainfile = ''
        self.testfile = ''

    def labalyzer(self, dataframe):
        def label(unit):
            return '__label__' + unit
        dataframe['label'] = dataframe['label'].apply(label)
        return dataframe

    def split_dataset(self, dataframe):
        text = dataframe['text'].values
        y = dataframe['label'].values

        text_train, text_test, y_train, y_test = train_test_split(
                    text, y, test_size=0.25, random_state=3)

        train_df = pd.DataFrame({'label':y_train, 'text': text_train}, columns=['label', 'text'])
        test_df = pd.DataFrame({'label':y_test, 'text': text_test}, columns=['label', 'text'])
        return train_df, test_df
    
    def write_to_txt(self, df, filepath):
        with open(f'{filepath}', 'w') as file:
            for index, row in df.iterrows():
                file.write(f"{row['label']} {row['text']}\n")

    def print_results(self, N, p, r):
        print("N\t" + str(N))
        print("P@{}\t{:.3f}".format(1, p))
        print("R@{}\t{:.3f}".format(1, r))


    def classification(self, trainfile, testfile, ngrams=1):
        hyper_params = {"lr": 0.01,
                        "epoch": 20,
                        "wordNgrams": ngrams,
                        "dim": 20}  

        self.model = fasttext.train_supervised(input=trainfile, **hyper_params)
        validation = self.model.test(testfile)
        score = {ngrams:{"N":int(validation[0]), "P@1": "{0:.3f}".format(validation[1]),"R@1": "{0:.3f}".format(validation[2])}}
        
        #self.print_results(*self.model.test(testfile))
        
        words, freq = self.model.get_words(include_freq=True)
        return score

    def compress_save(self):
        self.model.quantize(input=self.trainfile, retrain=True)
        modelname = f"{self.filepath}/model_fasttext_{self.name}-{timestamp}.ftz"
        self.model.save_model(modelname)
        return modelname

class FastTextPipeline:
    def __init__(self, name, lowercase=True, stopw=True, report='score'):
        self.name = name
        self.lowercase = lowercase
        self.stopw = stopw
        self.report = report
        self.modelname = ''
        self.fasttexter = FastTexter(name=self.name)
        self.explorer = DataExploration(filepath='../data/fasttext-report')
        self.reporter = ClassificationReports()
        

    def preprocessing(self, data, missing='Unknown'):
        Processor = DataPreProcessor(data)
        Cleaner = DataCleaner()
        dataframe = Processor.transformed_df
        dataframe = Cleaner.remove_values(dataframe, missing)
        if self.lowercase:
            dataframe = Cleaner.lowercase(dataframe)
        if self.stopw:
            dataframe = Cleaner.remove_stopwords(dataframe)
        return dataframe

    def prepare_files(self, data):
        dataframe = self.preprocessing(data)
        dataframe = self.fasttexter.labalyzer(dataframe)
        train_df, test_df = self.fasttexter.split_dataset(dataframe)
        self.trainfile = f'../data/fasttext_input/{self.name}.train.txt'
        self.testfile = f'../data/fasttext_input/{self.name}.test.txt'
        self.fasttexter.write_to_txt(train_df, self.trainfile)
        self.fasttexter.write_to_txt(test_df, self.testfile)

    def train_model(self, ngrams, save_model=False):
        score = self.fasttexter.classification(self.trainfile, self.testfile, ngrams)
        if save_model:
            self.modelname = self.fasttexter.compress_save()  
        return score

    def ngrams_performance(self):
        score_dict = {}
        pos_ngrams = [1, 2, 3, 4, 5]
        for ngram in pos_ngrams:
            score = self.train_model(ngram)
            score_dict.update(score)
        return score_dict

    def file_predictions(self):
        model = fasttext.load_model(self.modelname)


        label_scores = model.test_label(self.testfile)
        dataframe = pd.DataFrame(label_scores)

        if self.report == 'full':
            print(f'''
            Trained model tested on testfile.txt data:\n\n
            -> {self.name}\n
            \tclassifier: fasttext \n\n
            \t{self.fasttexter.print_results(*self.fasttexter.model.test(testfile))}\n\n
            \t{dataframe.T}\n
            ''')

    def scoring(self, y, y_pred):
        accuracy = 'accuracy %s' % accuracy_score(y_pred, y)
        columns = np.unique(y)
        report = classification_report(y, y_pred,target_names=columns)
        return accuracy, report

    def scoring_report(self, title, y, y_pred):
        accuracy, report = self.scoring(y, y_pred)
        print(f'''
        Trained model tested on test data:\n\n
        -> {title}\n
        \tclassifier: fasttext \n
        \t{accuracy}\n
        \n{report}
        ''')
        self.reporter.confusion_matrix_vis(y=y, y_pred=y_pred, filepath='../data/fasttext-reports',title=f'fasttext-{title}')
        #self.DataExploration.kijkdoos(X, y, y_pred, 'location')

    def dataframe_predictions(self, modelname, dataframe):
        model = fasttext.load_model(modelname)
        dataframe = self.fasttexter.labalyzer(dataframe)

        predictions_dict = {}  
        try:
            for index, row in dataframe.iterrows():
                text = str(row['text'].replace("\n", " "))
                label, prob = model.predict(text)
                predictions_dict.update({index: {'label':row['label'], 'text': row['text'], 'predict': label[0], 'proba': prob[0]}})
        except Exception as err:
            print(text)
            print(err)

        dataframe = pd.DataFrame(predictions_dict)
        transp = dataframe.T
        y_pred = transp['predict'].values
        y = transp['label'].values

        accuracy, report = self.scoring(y, y_pred)
        if self.report == 'full':
            self.scoring_report(self.name,y, y_pred)
            self.reporter.confusion_matrix_vis(y=y, y_pred=y_pred, filepath='../data/fasttext-reports',title=f'fasttext-{self.name}')

        return accuracy

if __name__ == '__main__':
        include = ['company_name', 'introduction', 'location', 'vacancy_title', 'description', 'job_category', 'contract_type']
        scrp_dataset = Elasticer().import_dataset('scrapy_test-early_mornin_4', include)
        #xml_data = xmlRemapper().get_dataframe()
        #print(xml_data[xml_data['label'] == 'introduction'])

        #modelname = 'model_fasttext_07082019-01:29.ftz'
        #modelname_low = 'model_fasttext_scrp_lowercased-07082019-06:17.ftz'
        testfile = '../data/unseen_test.txt.txt'

        # original = FastTextPipeline('original_dataset', lowercase=False, stopw=False)
        # original.prepare_files(scrp_dataset)
        # score = original.ngrams_performance()
        # df = pd.DataFrame(score)
        # print(df)

        # low_case = FastTextPipeline('scrp_lowercased', stopw=False)
        # low_case.prepare_files(scrp_dataset)
        # score = low_case.ngrams_performance()
        # df = pd.DataFrame(score)
        # print(df)

        low_case_minstop = FastTextPipeline('scrp_lowercased_minstop')
        low_case_minstop.prepare_files(scrp_dataset)
        low_case_minstop.train_model(1, save_model=True)
        # score = low_case_minstop.ngrams_performance()
        # df = pd.DataFrame(score)
        # print(df)
        low_case_minstop.file_predictions()
        #FastTextPipeline('scrp_to_xml').dataframe_predictions(modelname, xml_data)