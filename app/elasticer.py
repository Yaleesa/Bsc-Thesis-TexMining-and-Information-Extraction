'''
Author: Yaleesa Borgman
Date: 8-8-2019
GET and POST for elasticsearch
'''
from elasticsearch import Elasticsearch, helpers
import json


class Elasticer:
    def __init__(self):
        self.es = Elasticsearch(host="127.0.0.1")
        #self.es = Elasticsearch(host="elasticsearch")

    def import_dataset(self, indexname, include_list):
        '''
        following code gets data from elasticsearch, removes the keys not needed, retrieves the total_hits, uses the total hits and returns a list of dicts.
        '''
        totalhits = self.es.search(index=indexname,_source='false', body={})['hits']['total']['value']
        documents = self.es.search(index=indexname,body={}, _source_includes=include_list, size=totalhits)['hits']['hits']

        documents = [source['_source'] for source in documents]
        return documents


    def dict_to_elastic(self, indexname, data):
        actions = [
            {
            "_index" : indexname,
            "_source" : record
            }
        for record in data.values()
        ]
        actions = json.dumps(actions)
        print(actions)
        helpers.bulk(self.es,actions)

    def list_to_elastic(self, indexname, data):
        actions = [
            {
            "_index" : indexname,
            "_source" : record
            }
        for record in data
        ]
        helpers.bulk(self.es,actions, index=indexname)