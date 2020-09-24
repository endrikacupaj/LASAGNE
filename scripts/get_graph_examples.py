# %%
import ujson
entity_type = ujson.loads(open('entity_type.json').read())
id_entity = ujson.loads(open('items_wikidata_n.json').read())
id_relation = ujson.loads(open('filtered_property_wikidata4.json').read())
type_triples = ujson.loads(open('wikidata_type_dict.json').read())
# %%
predicate = 'P6'
for k, v in type_triples.items():
    if predicate in v.keys():
        for pv in v[predicate]:
            print(f'{id_entity[k]} - > {id_relation[predicate]} -> {id_entity[pv]}')
# %%
sport = 'form of government'
for k, v in type_triples.items():
    if sport in id_entity[k]:
        for kv, vv in v.items():
            for vvv in vv:
                print(f'{id_entity[k]} -> {id_relation[kv]} -> {id_entity[vvv]}')
# %%
