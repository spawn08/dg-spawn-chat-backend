from elasticsearch import Elasticsearch

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])

res = es.get(index="spawnai", doc_type='spawnai_file', id='bot_data')
print(res['_source'])
