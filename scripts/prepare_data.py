import os
import json
import argparse
from tqdm import tqdm

from constants import *

parser = argparse.ArgumentParser(description='Prepare wikidata files')
parser.add_argument('--wiki_path', default='knowledge_graph', help='Wikidata folder path')

def create_entity_type(type_ents, wiki_path):
    '''
    Build a dictionary
    key: ids of entity
    values: ids of type
    '''
    ent_type = {}

    for t in tqdm(type_ents.keys()):
        for e in type_ents[t]:
            if e in ent_type:
                ent_type[e].append(t)
            else:
                ent_type[e] = [t]

    json.dump(ent_type, open(str(ROOT_PATH.parent) + f'/{wiki_path}/entity_type.json', 'w'))

    return ent_type

def create_pred_sub_ob(triples, wiki_path):
    pred_sub_ob={}

    for e in tqdm(triples):
        for p in triples[e].keys():
            if p not in pred_sub_ob:
                pred_sub_ob[p] = {}

            pred_sub_ob[p][e] = []
            for o in triples[e][p]:
                pred_sub_ob[p][e].append(o)

    json.dump(pred_sub_ob, open(str(ROOT_PATH.parent) + f'/{wiki_path}/relation_subject_object.json', 'w'))


def create_pred_ob_sub(object_invTriples, wiki_path):
    pred_ob_sub={}

    for e in tqdm(object_invTriples):
        for p in object_invTriples[e].keys():
            if p not in pred_ob_sub:
                pred_ob_sub[p] = {}

            pred_ob_sub[p][e] = []
            for s in object_invTriples[e][p]:
                pred_ob_sub[p][e].append(s)

    json.dump(pred_ob_sub, open(str(ROOT_PATH.parent) + f'/{wiki_path}/relation_object_subject.json', 'w'))


if __name__ == '__main__':
    args = parser.parse_args()

    par_childs = json.load(open(str(ROOT_PATH.parent) + f'/{args.wiki_path}/par_child_dict.json', "r"))
    subject_triples = json.load(open(str(ROOT_PATH.parent) + f'/{args.wiki_path}/wikidata_short_1.json', "r"))
    subject_triples_2 = json.load(open(str(ROOT_PATH.parent) + f'/{args.wiki_path}/wikidata_short_2.json', "r"))
    object_invTriples = json.load(open(str(ROOT_PATH.parent) + f'/{args.wiki_path}/comp_wikidata_rev.json', "r"))

    triples_ = { **subject_triples, **subject_triples_2 } # dict[e][p] -> [o1, o2, o3]

    print("Preparing entity_type.json ...")
    create_entity_type(par_childs, args.wiki_path)

    print("Preparing relation_subject_object.json ...")
    create_pred_sub_ob(triples_, args.wiki_path)

    print("Preparing relation_object_subject.json ...")
    create_pred_ob_sub(object_invTriples, args.wiki_path)
