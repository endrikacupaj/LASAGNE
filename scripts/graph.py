import torch
import torch.nn as nn

class Graph:
    def __init__(self, type_vocab, pred_vocab):
        self.data = set()

        self.node_idx = {}
        self.idx_node = []
        self.type_mask = []
        self.pred_mask = []
        self.representations = []
        self.start = []
        self.end = []
        self.type_embeddings = nn.Embedding(len(type_vocab.itos), 100, padding_idx=type_vocab.stoi['NA'])
        self.pred_embeddings = nn.Embedding(len(pred_vocab.itos), 100, padding_idx=pred_vocab.stoi['NA'])
        self.type_vocab = type_vocab
        self.pred_vocab = pred_vocab

    def extract_graph_mode_full(self, kb):
        for type in kb.type_pred_type:
            if self.type_vocab.stoi.get(type, None) is not None: # filter types that are not in vocab
                if type not in self.node_idx:
                        self.add_type_node_info(type)
                for pred in kb.type_pred_type[type]:
                    if self.pred_vocab.stoi.get(pred, None) is not None:
                        if pred not in self.node_idx:
                            self.add_pred_node_info(pred)

                        for end_type in kb.type_pred_type[type][pred]:
                            if self.type_vocab.stoi.get(end_type, None) is not None:
                                if end_type not in self.node_idx:
                                    self.add_type_node_info(end_type)

                                self.add_node(type, pred)
                                self.add_node(pred, end_type)

    def add_type_node_info(self, type):
        self.node_idx[type] = len(self.node_idx)
        self.idx_node.append(type)
        assert self.idx_node.index(type) == self.node_idx[type]

        self.type_mask.append(1)
        self.pred_mask.append(0)
        emb = self.type_embeddings(torch.LongTensor([self.type_vocab.stoi[type]]))
        self.representations.append(emb.unsqueeze(0))

    def add_pred_node_info(self, pred):
        self.node_idx[pred] = len(self.node_idx)
        self.idx_node.append(pred)
        assert self.idx_node.index(pred) == self.node_idx[pred]
        self.type_mask.append(0)
        self.pred_mask.append(1)

        emb = self.pred_embeddings(torch.LongTensor([self.pred_vocab.stoi[pred]]))
        self.representations.append(emb.unsqueeze(0))


    def add_node(self, start, end):
        self.start.append(self.node_idx[start])
        self.end.append(self.node_idx[end])

    def __len__(self):
        return len(self.data)