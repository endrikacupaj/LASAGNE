import os
import ujson
import time
from pathlib import Path
ROOT_PATH = Path(os.path.dirname(__file__))

class KnowledgeGraph:
    def __init__(self, wikidata_path=f'{ROOT_PATH}'):
        tic = time.perf_counter()

        # id -> entity label
        self.id_entity = ujson.loads(open(f'{wikidata_path}/items_wikidata_n.json').read())
        print(f'Loaded id_entity {time.perf_counter()-tic:0.2f}s')

        # id -> relation label
        self.id_relation = ujson.loads(open(f'{wikidata_path}/filtered_property_wikidata4.json').read())
        print(f'Loaded id_relation {time.perf_counter()-tic:0.2f}s')

        # entity -> type
        self.entity_type = ujson.loads(open(f'{wikidata_path}/entity_type.json').read()) # dict[e] -> type
        print(f'Loaded entity_type {time.perf_counter()-tic:0.2f}s')

        # type -> relation -> type
        self.type_triples = ujson.loads(open(f'{wikidata_path}/wikidata_type_dict.json').read())
        print(f'Loaded type_triples {time.perf_counter()-tic:0.2f}s')

        # subject -> relation -> object
        self.subject_triples_1 = ujson.loads(open(f'{wikidata_path}/wikidata_short_1.json').read())
        self.subject_triples_2 = ujson.loads(open(f'{wikidata_path}/wikidata_short_2.json').read())
        self.subject_triples = {**self.subject_triples_1, **self.subject_triples_2}
        print(f'Loaded subject_triples {time.perf_counter()-tic:0.2f}s')

        # object -> relation -> subject
        self.object_triples = ujson.loads(open(f'{wikidata_path}/comp_wikidata_rev.json').read())
        print(f'Loaded object_triples {time.perf_counter()-tic:0.2f}s')

        # relation -> subject -> object | relation -> object -> subject
        self.relation_subject_object = ujson.loads(open(f'{wikidata_path}/relation_subject_object.json').read())
        self.relation_object_subject = ujson.loads(open(f'{wikidata_path}/relation_object_subject.json').read())
        print(f'Loaded relation_triples {time.perf_counter()-tic:0.2f}s')

        # labels
        self.labels = {
            'entity': self.id_entity, # dict[e] -> label
            'relation': self.id_relation # dict[r] -> label
        }

        # triples
        self.triples = {
            'subject': self.subject_triples, # dict[s][r] -> [o1, o2, o3]
            'object': self.object_triples, # dict[o][r] -> [s1, s2, s3]
            'relation': {
                'subject': self.relation_subject_object, # dict[r][s] -> [o1, o2, o3]
                'object': self.relation_object_subject # dict[r][o] -> [s1, s2, s3]
            },
            'type': self.type_triples # dict[t][r] -> [t1, t2, t3]
        }
