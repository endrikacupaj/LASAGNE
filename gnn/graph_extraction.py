from utils.csqa_dataset import CSQADataset
from utils.data_holders import Dialog
from utils.vocab import Vocab

from kb.knowledge_base import *
from kb.sampling import *
from kb.operations import OPs
from kb.actions import Actions
from kb import agent

import json
import os
import sys
from kb.utils import *
import pickle
from tqdm import tqdm


def add_e_p_triple(e, p, graph, kb):
    if p in kb.triples_[e]:
        for o in kb.triples_[e][p]:
            graph.add((e, p, o))
    if e in kb.object_invTriples and p in kb.object_invTriples[e]:
        for s in kb.object_invTriples[e][p]:
            graph.add((s, p, e))


if __name__ == "__main__":
    print("Started loading kb...")
    entity2id, relation2id, entity2vec, pred2vec = read_kb_embeddings()
    kb = KnowledgeBase(wikidata_path="data/kb/")
    sampler = KBSampler(kb, relation2id, entity2id)
    dataset = CSQADataset(sampler)
    vocab = Vocab(dataset)
    dataset.add_vocab(vocab)
    print("Finished constructing dataset...")
    
    
    graph_lengths = []
    types = kb.par_childs.keys()

    types_mask = []
    ent_mask = []
    pred_mask = []

    possible_problems = []
    not_found_gold = []
    ctr = 0

    # filter here if q.predicate is not empty
    # TODO: Take history into account
    # Extract types also

    for i in tqdm(range(len(dataset))):
        graph = set()
        dialog = dataset.extract(i)[2]
        q = dialog.question
        a = dialog.answer

        predicates = []
        predicates.extend(q.gold_predicates)
        if type(q.history['prev_rel']) is str:
            predicates.append(q.history['prev_rel'])
        else:
            predicates.extend(q.history['prev_rel'])
        # print(q.gold_predicates, q.history['prev_rel'], predicates, q.gold_predicates.extend(q.history['prev_rel']))

        if (len(predicates) != 0):
            for p in predicates:
                for t in kb.pred_etype[p]:
                    if ((t in q.gold_types or len(q.gold_types) == 0) and
                            ((t in kb.type_pred_type and p in kb.type_pred_type[t])
                             or (t in kb.rev_type_pred and p in kb.rev_type_pred[t]))):  # filter types

                        # if(p in kb.type_pred_type[t] or p in kb.rev_type_pred[t]):
                        # code to filter based on the entities
                        for e in kb.par_childs[t]:
                            if (e in q.gold_entities or (len(q.gold_entities) == 0)):
                                add_e_p_triple(e, p, graph, kb)  # Only at this point I add into the graph

                        if len(graph) == 0:
                            for e in q.gold_entities:
                                add_e_p_triple(e, p, graph, kb)

        elif len(q.gold_types) > 0:
            # print(i, q.gold_types)
            for t in q.gold_types:
                for e in kb.par_childs[t]:

                    if (e in q.gold_entities or (len(q.gold_entities) == 0)):
                        for p in kb.triples_[e]:
                            for o in kb.triples_[e][p]:
                                graph.add((e, p, o))

                            if (e in kb.object_invTriples and p in kb.object_invTriples[e]):
                                ctr += len(kb.object_invTriples[e][p])
                                for s in kb.object_invTriples[e][p]:
                                    graph.add((s, p, e))
        else:
            not_found_gold.append(i)
            # print(i, " what is this....")

        if (len(graph) == 0 and len(predicates) != 0):
            possible_problems.append(i)
            # print(i, q.utterance, kb.id_entity[q.gold_entities[0]], kb.id_predicates[q.gold_predicates[0]],
            #      kb.id_entity[q.gold_types[0]])
            # print("#####################")
        elif (len(graph) == 0):
            # print(i, q.utterance, kb.id_entity[q.gold_entities[0]], '-------', kb.id_entity[q.gold_types[0]])
            pass
        graph_lengths.append(len(graph))


    print("Graph lengths...")
    with open("data/graph_lengths.pkl", "wb") as fp:  # Pickling
        pickle.dump(graph_lengths, fp)

    print("Saving possible problems...")
    print("Length: ", len(possible_problems))
    with open("data/possible_problems.pkl", "wb") as fp:  # Pickling
        pickle.dump(possible_problems, fp)

    print("Saving not found any gold...")
    print("Length: ", len(not_found_gold))
    with open("data/not_found_gold.pkl", "wb") as fp:  # Pickling
        pickle.dump(not_found_gold, fp)

