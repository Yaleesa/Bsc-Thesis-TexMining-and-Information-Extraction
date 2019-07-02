import pandas as pd
from elasticsearch import Elasticsearch, helpers
import requests
from requests_html import HTMLSession
import json
from joblib import dump, load
import xml.etree.ElementTree as ET
NoneType = type(None)
'''
options for pandas
'''
pd.set_option("display.max_rows", 25)
pd.set_option('max_columns',10)
pd.set_option('display.width', 240)
pd.set_option('display.column_space', 18)

session = HTMLSession()

counter = 0
es = Elasticsearch(host="127.0.0.1")

def import_elastic(indexname):
    '''
    following code gets data from elasticsearch, removes the keys not needed, retrieves the total_hits, uses the total hits and returns a list of dicts.
    '''
    #exclude = ['id', 'root_url', 'country', 'config_name']

    totalhits = es.search(index=indexname,_source='false', body={})['hits']['total']['value']
    #all_documents = es.search(index='scrapy_test-early_mornin_4',body={})['hits']['hits']
    documents_exclude = es.search(index=indexname,body={}, size=totalhits)['hits']['hits']

    documents_exclude = [source['_source'] for source in documents_exclude]
    return documents_exclude

def to_elastic(indexname, df):
    jsonified = df.to_json(orient = "records")
    data = json.loads(jsonified)
    actions = [
        {
        "_index" : indexname,
        "_source" : record
        }
    for record in data
    ]
    helpers.bulk(es,actions)

class schemas:
    def schema_finder(url):
        schema = 'schema.org/JobPosting'
        global counter
        counter += 1
        print(counter)
        try:
            r = session.get(url)
            html = r.html.search(schema)
            if not isinstance(html, NoneType):
                return "True"
            else:
                return "False"
        except Exception as exc:
            return "Exception"

    def schema_parser(url):
        r = session.get(url)
        itemprops = r.html.find('[itemprop]')
        response = []
        for element in itemprops:
            item = {}
            attr = element.attrs
            item['item'] = attr['itemprop']
            text = element.text
            if text is not '':
                item['text'] = text
            if 'content' in attr:
                item['content'] = attr['content']
            response.append(item)
        return response

def json_ld(url):
    r = session.get(url)
    print(url)
    try:
        elements = r.html.xpath('//script[@type="application/ld+json"]')

    except Exception as err:
        print(elements)
        print(err)

    try:
        for unit in elements:
            text = unit.text
            text = text.replace('&#13;', '')
            LD = json.loads(text)

            for key in LD:

                print(key)
    except Exception as err:
        print(err)

    #print(LD['title'])

json_ld('https://www.whitbreadcareers.com/job-details/629467-2340/')
# docs = import_elastic('scrapy_test-early_mornin_4')
# for vac in docs:
#
#
#     json_ld(vac['root_url'])

def normal_parser(url):
    r = session.get(url)
    titles = r.html.xpath('//section//h1 | //section//h2 | //section//p | //section//ul', clean=True)
    for x in titles:
        print(x)
    # titles = r.html.find('h1', clean=True)
    # p = r.html.find('p', clean=True)
    # span = r.html.find('span', clean=True)
    # print([title.text for title in titles], [x.text for x in p], [sp.text for sp in span])
    # for unit, unit2, unit3 in titles, p, span:
    #     print(unit.text, unit2.text, unit3.text)

''''''

#units = schema_parser('https://jobs.nike.com/job/perry-barr/nike-part-time-athlete-sales-associate/824/12115196')


#normal_parser('https://www.studentjob.co.uk/vacancies/470091-mep-hospitality-staffing-part-time-and-full-time-london')
''''''

def xml_parser():
    r = requests.get('https://www.studentjob.co.uk/feed/jooble.xml?include_redirect_jobs=true')

    tree = ET.fromstring(r.text)
    count = 0
    unit = {}
    for job in tree.iter('JobDetails'):
        unit[count] = {}
        for job2 in job:

            unit[count].update({job2.tag: job2.text})

        count += 1
    print(count)
    return unit


def xml_to_frame():
    vacs = xml_parser()
    #print(vacs[2]['JobTitle'])
    drop_list = ['companyLogo', 'Id', 'JobCompanyBranch', 'JobCompanyProfile' , 'JobMinDaysPerWeek', 'JobUrl', 'topjob', 'HoursWeek', 'country', 'JobParttime', 'DatePlaced']
    df = pd.DataFrame.from_dict(vacs, orient='index')

    df.drop(drop_list, inplace=True, axis=1)
    #print(df['JobDescription'].str.strip())
    rename_dict = {'Title': 'vacancy_title',
                  'functionTitle': 'vacancy_title',
                  'TitleDescription': 'introduction',
                  'JobCategory': 'contract_type',
                  'JobBranch': 'job_category',
                  'JobDescription': 'description',
                  'profession': 'job_category',
                  'JobLocation': 'location',
                  'postalCode': 'location',
                  'JobCompany': 'company_name',
                  'JobProfession': 'job_category'}
    df.rename(columns=rename_dict, inplace=True)

    columns = df.columns.values
    melted_df = pd.melt(df, value_vars=columns, var_name='label', value_name='text',)

    dataframe = melted_df[melted_df['text'].notnull()]

    print(dataframe.sample(20))

    dataframe.to_json('xml_data_postalcode.json', orient='records')






            # documents = import_elastic('scrapy_test-early_mornin_4')
            # df = pd.DataFrame(documents)
            # urls = df['root_url']
            #grouped = df.groupby('company_name').first()
            #grouped['jobposting'] = grouped['root_url'].apply(schema_finder)
            #print(grouped.query("jobposting == 'True'"))
            #to_elastic(grouped)


        # title = r.html.find('[itemprop=title]', first=True)
        # print(f'title: {title.text}')
        # contract_type = r.html.find('[itemprop=employmentType]', first=True)
        # print(f'contract_type: {contract_type}')
        # location = r.html.find('[itemprop=jobLocation]', first=True)
        # print(f'location: {location.text}')
        # description = r.html.find('[itemprop=description]', first=True)
        # print(f'description: {description.text}')
        # base_salary = r.html.find('[itemprop=baseSalary]', first=True)
        # print(f'baseSalary: {base_salary.text}')
        ### missing introduction and category
