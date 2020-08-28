import time
from unidecode import unidecode
from elasticsearch import Elasticsearch
from knowledge_graph.knowledge_graph import KnowledgeGraph

kg = KnowledgeGraph()
kg_entities = list(kg.id_entity.items())
kg_types = kg.entity_type
print(f'Num of wikidata entities: {len(kg_entities)}')

es = Elasticsearch([{'host': 'localhost', 'port': 9200}])
# es.indices.delete(index='csqa_wikidata', ignore=[400, 404])

tic = time.perf_counter()
for i, (id, label) in enumerate(kg_entities):
    es.index(index='csqa_wikidata', doc_type='entities', id=i+1, body={'id': id, 'label': unidecode(label), 'type': kg_types[id] if id in kg_types else []})
    if (i+1) % 10000 == 0: print(f'==> Finished {((i+1)/len(kg_entities))*100:.4f}% -- {time.perf_counter() - tic:0.2f}s')

query = unidecode('Albania')
res = es.search(index='csqa_wikidata', doc_type='entities', body={
        'size': 50,
        'query': {
            'match': {
                'label': {
                    'query': query,
                    'fuzziness': 'AUTO',
                }
            }
        }
    })

for hit in res['hits']['hits']:
    print(f'{hit["_source"]["id"]} - {hit["_source"]["label"]} - {hit["_score"]}')
    print('**********************')
