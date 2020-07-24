from __future__ import division
import os
import re
import json
import h5py
import random
import logging
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from args import get_parser
from more_itertools import unique_everseen
from torchtext.data import Field, Example, Dataset

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

# read parser
parser = get_parser()
args = parser.parse_args()

# set logger
logger = logging.getLogger(__name__)

# define device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# torchtext fields
INPUT = 'input'
LOGICAL_FORM = 'logical_form'
NER = 'ner'
COREF = 'coref'
PREDICATE = 'predicate'
TYPE = 'type'

# helper tokens
START_TOKEN = '[START]'
END_TOKEN = '[END]'
CTX_TOKEN = '[CTX]'
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
SEP_TOKEN = '[SEP]'

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, model_size=args.embDim, factor=1, warmup=4000):
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        return self.factor * \
            (self.model_size ** (-0.5) *
            min(step ** (-0.5), step * self.warmup ** (-1.5)))

    def zero_grad(self):
        self.optimizer.zero_grad()

# meter class for storing results
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class Predictor(object):
    """Predictor class"""
    def __init__(self, model, vocabs, device):
        self.model = model
        self.input_vocab = vocabs[INPUT]
        self.lf_vocab = vocabs[LOGICAL_FORM]
        self.ner_vocab = vocabs[NER]
        self.coref_vocab = vocabs[COREF]
        self.predicate_vocab = vocabs[PREDICATE]
        self.type_vocab = vocabs[TYPE]
        self.device = device

    def predict(self, example):
        """Perform prediction on given input example"""
        self.model.eval()
        # prepare input
        tokenized_sentence = [START_TOKEN] + [t.lower() for t in example.input] + [CTX_TOKEN]
        numericalized = [self.input_vocab.stoi[token] for token in tokenized_sentence]
        src_tensor = torch.LongTensor(numericalized).unsqueeze(0).to(self.device)

        # get ner, coref predictions
        output = self.model._predict(src_tensor)
        output['ner'] = output['ner'].argmax(1).tolist()
        output['coref'] = output['coref'].argmax(1).tolist()

        # get logical form, predicate and type prediction
        lf_out = [self.lf_vocab.stoi[START_TOKEN]]
        predicate_out = [self.lf_vocab.stoi[START_TOKEN]]
        type_out = [self.lf_vocab.stoi[START_TOKEN]]

        for _ in range(self.model.decoder.max_positions):
            lf_tensor = torch.LongTensor(lf_out).unsqueeze(0).to(self.device)

            dec_out, predic_out, typ_out = self.model._predict_other(src_tensor, lf_tensor, output['encoder_out'])

            pred_lf = dec_out.argmax(1)[-1].item()
            pred_predicate = predic_out.argmax(1)[-1].item()
            pred_type = typ_out.argmax(1)[-1].item()

            if pred_lf == self.lf_vocab.stoi[END_TOKEN]:
                break

            lf_out.append(pred_lf)
            predicate_out.append(pred_predicate)
            type_out.append(pred_type)

        # translate top predictions into vocab tokens
        output['ner'] = [self.ner_vocab.itos[i] for i in output['ner']][1:-1]
        output['coref'] = [self.coref_vocab.itos[i] for i in output['coref']][1:-1]
        output['logical_form'] = [self.lf_vocab.itos[i] for i in lf_out][1:]
        output['predicate'] = [self.predicate_vocab.itos[i] for i in predicate_out][1:]
        output['type'] = [self.type_vocab.itos[i] for i in type_out][1:]

        # delete decoder output
        del output['encoder_out']

        return output

class AccuracyScorer(object):
    """Accuracy scorer class"""
    def __init__(self):
        self.results = []
        self.instances = 0
        self.data_dict = []

    def data_score(self, data, predictor):
        """Score complete list of data"""
        for example in data:
            # count example
            self.instances += 1

            # prepare references
            ref_ner = example.ner
            ref_coref = example.coref
            ref_lf = [t.lower() for t in example.logical_form]
            ref_pred = example.predicate
            ref_type = example.type

            # get model hypothesis
            hypothesis = predictor.predict(example)

            # check correctness
            correct_ner = 1 if ref_ner == hypothesis['ner'] else 0
            correct_coref = 1 if ref_coref == hypothesis['coref'] else 0
            correct_lf = 1 if ref_lf == hypothesis['logical_form'] else 0
            correct_pred = 1 if ref_pred == hypothesis['predicate'] else 0
            correct_type = 1 if ref_type == hypothesis['type'] else 0

            # save results
            self.results.append({
                'ner': correct_ner,
                'coref': correct_coref,
                'logical_form': correct_lf,
                'predicate': correct_pred,
                'type': correct_type
            })

            # save data
            self.data_dict.append({
                'input': example.input,
                'ner': hypothesis['ner'],
                'ner_gold': example.ner,
                'coref': hypothesis['coref'],
                'coref_gold': example.coref,
                'logical_form': hypothesis['logical_form'],
                'logical_form_gold': example.logical_form,
                'predicate': hypothesis['predicate'],
                'predicate_gold': example.predicate,
                'type': hypothesis['type'],
                'type_gold': example.type,
                # ------------------------------------
                'ner_correct': correct_ner,
                'coref_correct': correct_coref,
                'logical_form_correct': correct_lf,
                'predicate_correct': correct_pred,
                'type_correct': correct_type,
            })

    def write_results(self):
        save_dict = json.dumps(self.data_dict, indent=4)
        save_dict_no_space_1 = re.sub(r'": \[\s+', '": [', save_dict)
        save_dict_no_space_2 = re.sub(r'",\s+', '", ', save_dict_no_space_1)
        save_dict_no_space_3 = re.sub(r'"\s+\]', '"]', save_dict_no_space_2)
        with open(f'{ROOT_PATH}/{args.path_error_analysis}/error_analysis.json', 'w', encoding='utf-8') as json_file:
            json_file.write(save_dict_no_space_3)

    def ner_accuracy(self):
        """Return accuracy for NER"""
        return float(sum([res['ner'] for res in self.results])) / float(self.instances)

    def coref_accuracy(self):
        """Return accuracy for Coreference"""
        return float(sum([res['coref'] for res in self.results])) / float(self.instances)

    def lf_accuracy(self):
        """Return accuracy for Logical Form"""
        return float(sum([res['logical_form'] for res in self.results])) / float(self.instances)

    def predicate_accuracy(self):
        """Return accuracy for Predicate"""
        return float(sum([res['predicate'] for res in self.results])) / float(self.instances)

    def type_accuracy(self):
        """Return accuracy for Type"""
        return float(sum([res['type'] for res in self.results])) / float(self.instances)

    def total_accuracy(self):
        """Return accuracy for all tasks combined"""
        return float(sum([1 if all([v == 1 for v in res.values()]) else 0 for res in self.results])) / float(self.instances)

    def reset(self):
        """Reset object properties"""
        self.results = []
        self.instances = 0

def save_checkpoint(state):
    filename = f'{ROOT_PATH}/{args.snapshots}/ConvQA_model_e{state["epoch"]}_v-{state["best_val"]:.2f}.pth.tar'
    torch.save(state, filename)

def init_weights(model):
    # initialize model parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

class NerLoss(nn.Module):
    '''NER Loss'''
    def __init__(self, ignore_index):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.criterion(output, target)

class CorefLoss(nn.Module):
    '''Coref Loss'''
    def __init__(self, ignore_index):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.criterion(output, target)

class LogicalFormLoss(nn.Module):
    '''Logical Form Loss'''
    def __init__(self, ignore_index):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, output, target):
        return self.criterion(output, target)

class PredicateLoss(nn.Module):
    '''Predicate Loss'''
    def __init__(self, ignore_index):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.criterion(output, target)

class TypeLoss(nn.Module):
    '''Type Loss'''
    def __init__(self, ignore_index):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.criterion(output, target)

class MultiTaskLoss(nn.Module):
    '''Multi Task Learning Loss'''
    def __init__(self, ignore_index):
        super().__init__()
        self.ner_loss = NerLoss(ignore_index)
        self.coref_loss = CorefLoss(ignore_index)
        self.lf_loss = LogicalFormLoss(ignore_index)
        self.predicate_loss = PredicateLoss(ignore_index)
        self.type_loss = TypeLoss(ignore_index)

        self.mml_emp = torch.Tensor([True, True, True, True, True])
        self.log_vars = torch.nn.Parameter(torch.zeros(len(self.mml_emp)))

    def forward(self, output, target):
        # weighted loss
        task_losses = torch.stack((
            self.ner_loss(output['ner'], target['ner']),
            self.coref_loss(output['coref'], target['coref']),
            self.lf_loss(output['logical_form'], target['logical_form']),
            self.predicate_loss(output['predicate'], target['predicate']),
            self.type_loss(output['type'], target['type'])
        ))

        dtype = task_losses.dtype
        stds = (torch.exp(self.log_vars)**(1/2)).to(DEVICE).to(dtype)
        weights = 1 / ((self.mml_emp.to(DEVICE).to(dtype)+1)*(stds**2))

        losses = weights * task_losses + torch.log(stds)

        return {
            'ner': losses[0],
            'coref': losses[1],
            'logical_form': losses[2],
            'predicate': losses[3],
            'type': losses[4],
            'multi_task': losses.mean()
        }[args.task]

def Embedding(num_embeddings, embedding_dim, padding_idx):
    """Embedding layer"""
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.uniform_(m.weight, -0.1, 0.1)
    nn.init.constant_(m.weight[padding_idx], 0)
    return m

def Linear(in_features, out_features, bias=True):
    """Linear layer"""
    m = nn.Linear(in_features, out_features, bias=bias)
    m.weight.data.uniform_(-0.1, 0.1)
    if bias:
        m.bias.data.uniform_(-0.1, 0.1)
    return m

def LSTM(input_size, hidden_size, **kwargs):
    """LSTM layer"""
    m = nn.LSTM(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m

def GRU(input_size, hidden_size, **kwargs):
    """GRU layer"""
    m = nn.GRU(input_size, hidden_size, **kwargs)
    for name, param in m.named_parameters():
        if 'weight' in name or 'bias' in name:
            param.data.uniform_(-0.1, 0.1)
    return m