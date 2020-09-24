import json
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch_geometric.data import Data

# import constants
from constants import *

class TypeRelationGraph:
    def __init__(self, vocab, type_path=f'{ROOT_PATH}'):
        self.vocab = vocab
        self.existing_nodes = list(vocab.stoi.keys())
        self.type_triples = json.loads(open(f'{type_path}/knowledge_graph/wikidata_type_dict.json').read())
        self.bert_embeddings = json.loads(open(f'{type_path}/knowledge_graph/node_embeddings.json').read())
        self.nodes = torch.tensor([self.bert_embeddings[node] for node in self.existing_nodes], requires_grad=True)
        self.start = []
        self.end = []
        self.existing_edges = []

        # create edges
        self._create_edges()

        # create PyG graph
        self.data = Data(x=self.nodes, edge_index=torch.LongTensor([self.start, self.end])).to(DEVICE)

    def _create_edges(self):
        # extract graph data from KG
        for head in self.type_triples:
            if head in self.vocab.stoi: # only types that are in vocab
                for relation in self.type_triples[head]:
                    if relation in self.vocab.stoi: # only predicates that are in vocab
                        self._add_edge(head, relation) # add head -> relation edge
                        for tail in self.type_triples[head][relation]:
                            if tail in self.vocab.stoi:
                                self._add_edge(relation, tail) # add relation -> tail edge

    def _add_edge(self, start, end):
        if f'{start}->{end}' not in self.existing_edges:
            self.start.append(self.existing_nodes.index(start))
            self.end.append(self.existing_nodes.index(end))
            self.existing_edges.append(f'{start}->{end}')
