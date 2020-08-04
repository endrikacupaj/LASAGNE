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
        self.type_embeddings = nn.Embedding(len(type_vocab.itos), 300, padding_idx=type_vocab.stoi['NA'])
        self.pred_embeddings = nn.Embedding(len(pred_vocab.itos), 300, padding_idx=pred_vocab.stoi['NA'])
        self.type_vocab = type_vocab
        self.pred_vocab = pred_vocab

    def extract_graph_mode_one(self, type_list, pred_list, kb):
        for i, type in enumerate(type_list):
            for j in range(i+1, len(type_list)):
                node_end = type_list[j]
                if type not in self.node_idx:
                    self.add_type_node_info(type)
                
                if node_end not in self.node_idx:
                    self.add_type_node_info(node_end)
                    
                self.add_node()
    
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


    def add_node(self, node_start, node_end):
        self.start.append(node_start)
        self.end.append(node_end)


    def add_if_exists(self, head, main_struct, check_struct):
        for p in main_struct[head]:
            if p in check_struct:
                for tail in main_struct[head][p]:
                    if (head, p, tail) not in self.data and tail in self.type_vocab.stoi:
                        self.data.add((head, p, tail))

                        if head not in self.node_idx:
                            self.node_idx[head] = len(self.node_idx)
                            self.type_mask.append(1)
                            self.pred_mask.append(0)

                            emb = self.type_embeddings(torch.LongTensor([self.type_vocab.stoi[head]]))
                            self.representations.append(emb.unsqueeze(0))
                        
                        if p not in self.node_idx:
                            self.node_idx[p] = len(self.node_idx)
                            self.type_mask.append(0)
                            self.pred_mask.append(1)

                            emb = self.pred_embeddings(torch.LongTensor([self.pred_vocab.stoi[p]]))
                            self.representations.append(emb.unsqueeze(0))
                        
                        if tail not in self.node_idx:
                            self.node_idx[tail] = len(self.node_idx)
                            self.type_mask.append(1)
                            self.pred_mask.append(0)

                            emb = self.type_embeddings(torch.LongTensor([self.type_vocab.stoi[tail]]))
                            self.representations.append(emb.unsqueeze(0))
                        
                        self.start.append(self.node_idx[head])
                        self.end.append(self.node_idx[p])

                        self.start.append(self.node_idx[p])
                        self.end.append(self.node_idx[tail])

                    
    
    def __len__(self):
        return len(self.data)