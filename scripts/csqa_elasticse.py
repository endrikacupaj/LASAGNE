import time
from unidecode import unidecode
from elasticsearch import Elasticsearch
from knowledge_graph.knowledge_graph import KnowledgeGraph


entities = list(KnowledgeGraph().id_entity.items())
print(f'Num of wikidata entities: {len(entities)}')

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
# es.indices.delete(index='csqa_wikidata', ignore=[400, 404])

tic = time.perf_counter()
for i, ent in enumerate(entities):
    es.index(index='csqa_wikidata', doc_type='entities', id=i+1, body={'id': ent[0], 'label': unidecode(ent[1])})
    if (i+1) % 10000 == 0: print(f'==> Finished {((i+1)/len(entities))*100:.4f}% -- {time.perf_counter() - tic:0.2f}s')

# query = unidecode('Đồng Gia')
# res = es.search(index='csqa_wikidata', doc_type='entities', body={
#         'query': {
#             'match': {
#                 'label': {
#                     'query': query,
#                     'fuzziness': 'AUTO',
#                 }
#             }
#         }
#     })

# for hit in res['hits']['hits']:
#     print(f'{hit["_source"]["id"]} - {hit["_source"]["label"]} - {hit["_score"]}')
#     print('**********************')
