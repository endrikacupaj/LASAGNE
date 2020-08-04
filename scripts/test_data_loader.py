#%%
import os
import sys
sys.path.append('/data/joanplepi/Transformer_GNN')
from pathlib import Path
from csqa_dataset import CSQADataset

import torch
import torch.nn as nn
import torch.optim
from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torchtext.data import BucketIterator

import numpy as np

from const import (INPUT, LOGICAL_FORM, NER, COREF, START_TOKEN, END_TOKEN,
                    CTX_TOKEN, PAD_TOKEN, UNK_TOKEN, SEP_TOKEN, PREDICATE, TYPE)
from models.model import ConvQA
from utils import NoamOpt, AverageMeter, save_checkpoint, init_weights
from utils import NerLoss, CorefLoss, LogicalFormLoss, MultiTaskLoss
from knowledge_graph.knowledge_graph import *
#%%
kb = KnowledgeBase("/data/joanplepi/csqa/data/kb/")
#%%
# set root path
#ROOT_PATH = Path(os.path.dirname(__file__))
# load data
dataset = CSQADataset('/data/sample3')
vocabs = dataset.get_vocabs()
val_data = dataset.get_data()
# %%
train_data, val_data, test_data = dataset.get_data()
print(len(train_data))
dataset.get_vocabs()[LOGICAL_FORM].stoi
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_loader, val_loader = BucketIterator.splits((train_data, val_data),
                                                batch_size=1,
                                                sort_within_batch=False,
                                                #sort_key=lambda x: len(x.input),
                                                device=DEVICE)

# %%
# load model
from models.transformer import Encoder, Decoder
from models.model import NerNet, SeqNet
class ConvQA(nn.Module):
    def __init__(self, vocabs):
        super().__init__()
        self.vocabs = vocabs
        self.encoder = Encoder(vocabs[INPUT], DEVICE)
        self.decoder = Decoder(vocabs[LOGICAL_FORM], DEVICE)
        self.ner = NerNet(len(vocabs[NER]))
        self.coref = SeqNet(len(vocabs[COREF]))
        self.predicate_cls = SeqNet(len(vocabs[PREDICATE]))
        self.type_cls = SeqNet(len(vocabs[TYPE]))

    def forward(self, src_tokens, trg_tokens):
        encoder_out = self.encoder(src_tokens)
        ner_out, ner_h = self.ner(encoder_out)
        coref_out = self.coref(torch.cat([encoder_out, ner_h], dim=-1))
        decoder_out, decoder_h = self.decoder(src_tokens, trg_tokens, encoder_out)
        encoder_ctx = encoder_out[:, -1:, :].expand(decoder_h.shape)
        predicate_out = self.predicate_cls(torch.cat([encoder_ctx, decoder_h], dim=-1))
        type_out = self.type_cls(torch.cat([encoder_ctx, decoder_h], dim=-1))

        print(type_out.size(), predicate_out.size())

        return {
            'ner': ner_out,
            'coref': coref_out,
            'logical_form': decoder_out,
            'predicate': predicate_out,
            'type': type_out
        }

model = ConvQA(vocabs).to(DEVICE)

# initialize model weights
init_weights(model)

criterion = {
        'ner': NerLoss,
        'coref': CorefLoss,
        'logical_form': LogicalFormLoss,
        'predicate': LogicalFormLoss,
        'type': LogicalFormLoss,
        'multi_task': MultiTaskLoss
    }['multi_task'](ignore_index=vocabs[LOGICAL_FORM].stoi[PAD_TOKEN])
optimizer = NoamOpt(torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
# %%
def train(train_loader, model, vocabs, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
        # get inputs
        input = batch.input
        logical_form = batch.logical_form
        ner = batch.ner
        coref = batch.coref
        predicate_cls = batch.predicate
        type_cls = batch.type

        # compute output
        output = model(input, logical_form[:, :-1])

        # prepare targets
        target = {
            'ner': ner.contiguous().view(-1),
            'coref': coref.contiguous().view(-1),
            'logical_form': logical_form[:, 1:].contiguous().view(-1), # (batch_size * trg_len)
            'predicate': predicate_cls[:, 1:].contiguous().view(-1),
            'type': type_cls[:, 1:].contiguous().view(-1),
        }

        # compute loss
        loss = criterion(output, target)

        # record loss
        losses.update(loss.data, input.size(0))

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        print(f'Epoch: {epoch+1} - Train loss: {losses.val:.4f} ({losses.avg:.4f}) - Batch: {((i+1)/len(train_loader))*100:.2f}% - Time: {batch_time.sum:0.2f}s')


# %%
model.train()
end = time.time()
for i, batch in enumerate(train_loader):
    # get inputs
    input = batch.input
    logical_form = batch.logical_form
    ner = batch.ner
    coref = batch.coref
    predicate_cls = batch.predicate
    type_cls = batch.type

    # compute output
    output = model(input, logical_form[:, :-1])

    break
# %%

# %%
# Filter types and predicates indices from NA. 
type_ind = type_cls[type_cls != 0]
pred_ind = predicate_cls[predicate_cls != 0]

# Extract types and predicates from vocab based on indices filtered above. 
type_vocab = dataset.get_vocabs()[TYPE]
pred_vocab = dataset.get_vocabs()[PREDICATE]

type_list = [type_vocab.itos[i] for i in type_ind]
pred_list = [pred_vocab.itos[i] for i in pred_ind]
print(type_list, pred_list)
# %%
from importlib import reload
import gnn.graph as graph_to_reload
reload(graph_to_reload)

from gnn.graph import Graph

all_graphs = []
subgraph = Graph(type_vocab, pred_vocab)
for type in type_list: 
    if type in kb.type_pred_type:
        subgraph.add_if_exists(type, kb.type_pred_type, pred_list)
    if type in kb.rev_type_pred:
        subgraph.add_if_exists(type, kb.rev_type_pred, pred_list)

all_graphs.append(subgraph)

# %%
edge_index = torch.tensor([subgraph.start, subgraph.end], dtype=torch.long)

data = Data(x=torch.cat(subgraph.representations), edge_index=edge_index)
gat = GATConv(300, 100, heads=2, concat=False, dropout=0.2)
out = gat(data.x, data.edge_index)


# %%
# Process csqa dataset for subgraphs.
from glob import glob
import json

root_path = Path(os.path.dirname(__file__)).parent

files_path =  str(root_path) + '/data/sample/train/*'
files = glob(files_path + '/*.json')
print(len(files))
# %%
for file in files:
    with open(file) as f:
        conversation = json.load(f)

        turns = len(conversation) // 2
        for i in range(turns):
            user = conversation[2*i]
            system = conversation[2*i + 1]
            type_list = extract_type_list(system, kb, dataset.get_vocabs()[TYPE])
            pred_list = extract_pred_list(system, kb, dataset.get_vocabs()[PREDICATE])
            
            
        break


# %%
import random
def extract_type_list(system, kb, type_vocab, total_nr=10):
    type_list = [a[1] for a in system['gold_actions'] if a[0] == 'type']    
    sample_size = total_nr - len(type_list)
    type_list.extend(random.sample(type_vocab.itos, sample_size))
    print(type_list)
    return type_list

def extract_pred_list(system, kb, pred_vocab, total_nr=10):
    pred_list = [a[1] for a in system['gold_actions'] if a[0] == 'relation']    
    print(pred_list)
    sample_size = total_nr - len(pred_list)
    pred_list.extend(random.sample(pred_vocab.itos, sample_size))
    print(pred_list)
    return pred_list

# %%
