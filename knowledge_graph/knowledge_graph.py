import time

from tqdm import tqdm
import json
import sys
sys.path.append('/data/joanplepi/csqa')
#from utils.train_utils import tic
import os
import pickle

def tic(start, s=""):
    t = time.time() - start
    print(s + "took: %.2f sec" % t)


class CustomDict(dict):
    def __init__(self,*arg,**kw):
        super(CustomDict, self).__init__(*arg, **kw)
    
    def add_set(self, key, value):
        if key not in self:
            self[key] = set({value})
        else:
            self[key].add(value)
    
    def add_list(self, key, value):
        if key not in self:
            self[key] = [value]
        else:
            self[key].append(value)
            
class KnowledgeBase:
    def __init__(self, wikidata_path="data/wikidata"):
        start = time.time()
        self.wikidata_path = wikidata_path
        
        self.id_entity = json.load(open(f'{self.wikidata_path}items_wikidata_n.json'))  # id -> entity value and type values
        tic(start, "Id->entity loading ")
        
        start = time.time()
        self.id_predicates = json.load(open(f'{self.wikidata_path}filtered_property_wikidata4.json'))
        tic(start, "Id->predicates loading ")
        
        start = time.time()
        self.subject_triples = json.load(open(f'{self.wikidata_path}wikidata_short_1.json'))
        tic(start, "Subject_key->triples_1 loading ")
        
        start = time.time()
        self.subject_triples_2 = json.load(open(f'{self.wikidata_path}wikidata_short_2.json'))
        tic(start, "Subject_key->triples_2 loading ")
        
        start = time.time()
        self.object_invTriples = json.load(open(f'{self.wikidata_path}comp_wikidata_rev.json'))
        tic(start, "Object_key->invTriples loading ")
        
        self.triples_ = {**self.subject_triples, **self.subject_triples_2} # dic[e][p] -> [o1, o2, o3]
        self.triples = [self.triples_, self.object_invTriples]
        """
        key: entity TYPE
        value: dict
        dict: key: predicate, value: object entity types

        Which entity types as a subject are linked with a predicate to the entity types as an object
        """
        start = time.time()
        self.type_pred_type = json.load(open(f'{self.wikidata_path}wikidata_type_dict.json', "r")) 
        #########################################
        self.rev_type_pred = json.load(open(f'{self.wikidata_path}wikidata_rev_type_dict.json', "r"))
        tic(start, "Loading type-pred-type")
        """
        key: entity TYPE
        value: list of children entities
        """
        start = time.time()
        self.par_childs = json.load(open(f'{self.wikidata_path}par_child_dict.json', "r"))
        self.ent_type, self.multiple_type_e = self.create_entity_type(self.par_childs)
        self.pred_etype, self.errors, self.rev_errors, self.no_type_errors = self.create_pred_etype(self.ent_type, self.triples)
        tic(start, "Loading  type-entity")
        
        if os.path.exists(f'{self.wikidata_path}kb_objects/pred_ob_sub.pkl'):
            self.pred_ob_sub = pickle.load(open(f'{self.wikidata_path}kb_objects/pred_ob_sub.pkl', 'rb'))
        else:
            self.pred_ob_sub = self.create_pred_ob_sub()

        if os.path.exists(f'{self.wikidata_path}kb_objects/pred_sub_ob.pkl'):
            self.pred_sub_ob = pickle.load(open(f'{self.wikidata_path}kb_objects/pred_sub_ob.pkl', 'rb'))
        else:
            self.pred_sub_ob = self.create_pred_sub_ob()

        if os.path.exists(f'{self.wikidata_path}kb_objects/child_par_5_levels.json'):
            self.child_par_5_levels = pickle.load(open(f'{self.wikidata_path}kb_objects/child_par_5_levels.json', 'rb'))


    def create_pred_sub_ob(self):
        pred_sub_ob={}

        for e in self.triples_:
            for p in self.triples_[e].keys():
                if p not in pred_sub_ob:
                    pred_sub_ob[p] = {}
                
                pred_sub_ob[p][e] = []
                for o in self.triples_[e][p]:
                    pred_sub_ob[p][e].append(o)

        pickle.dump(pred_sub_ob, open(f'{self.wikidata_path}kb_objects/pred_sub_ob.pkl', 'wb'))

        return pred_sub_ob

    def create_pred_ob_sub(self):
        pred_ob_sub={}

        for e in self.object_invTriples:
            for p in self.object_invTriples[e].keys():
                if p not in pred_ob_sub:
                    pred_ob_sub[p] = {}
                
                pred_ob_sub[p][e] = []
                for s in self.object_invTriples[e][p]:
                    pred_ob_sub[p][e].append(s)
        
        pickle.dump(pred_ob_sub, open(f'{self.wikidata_path}kb_objects/pred_ob_sub.pkl', 'wb'))

        return pred_ob_sub

    def create_entity_type(self, type_ents):
        '''
        Build a dictionary
        key: ids of entity
        values: ids of type
        '''
        
        ent_type_path = f'{self.wikidata_path}kb_objects/entity_type.pkl'

        if os.path.exists(ent_type_path): 
            print("Loading entity->type dict...")
            ent_type = pickle.load(open(ent_type_path, 'rb'))
            return ent_type, None 

        ent_type = {}
        multiple_type_e = []

        for t in tqdm(type_ents.keys()):
            for e in type_ents[t]:
                if e in ent_type:
                    ent_type[e].append(t)
                    multiple_type_e.append(e)
                else:
                    ent_type[e] = [t]


        pickle.dump(ent_type,open(ent_type_path,'wb'))

        return ent_type, multiple_type_e

    
    def create_pred_etype(self, eType, triples):
        pred_etype = CustomDict()
        errors = []
        rev_errors = []
        no_type_errors = []
        path = f'{self.wikidata_path}kb_objects/predicate_types.pkl'

        if os.path.exists(path):
            print("Loading from " + path)
            pred_etype = pickle.load(open(path, 'rb'))
            return pred_etype, None, None, None

        for t in triples:
            for e, pred_e in tqdm(t.items()):
                for p in pred_e:
                    try:
                        for typ in self.ent_type[e]: # might have two types for one entity.
                            pred_etype.add_set(p, typ)
                    except KeyError:
                        no_type_errors.append(e)

                    for o in pred_e[p]:
                        try:
                            for typ in self.ent_type[o]: # might have two types for one entity.
                                pred_etype.add_set(p, typ)
                        except KeyError:
                            no_type_errors.append(o)

        pickle.dump(pred_etype, open(path, 'wb'))

        return pred_etype, errors, rev_errors, no_type_errors

