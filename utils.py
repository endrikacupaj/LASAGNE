from __future__ import division
import os
import re
import time
import json
import h5py
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from pathlib import Path
from args import get_parser
from unidecode import unidecode
from collections import OrderedDict
from transformers import BertTokenizer
from elasticsearch import Elasticsearch
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
COREF_RANKING = 'coref_ranking'

# helper tokens
START_TOKEN = '[START]'
END_TOKEN = '[END]'
CTX_TOKEN = '[CTX]'
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
SEP_TOKEN = '[SEP]'

class NoamOpt:
    "Optim wrapper that implements rate."
    def __init__(self, optimizer, model_size=args.embDim, factor=args.factor, warmup=args.warmup):
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
        self.coref_ranking_vocab = vocabs[COREF_RANKING]
        self.predicate_vocab = vocabs[PREDICATE]
        self.type_vocab = vocabs[TYPE]
        self.device = device

    def predict(self, input):
        """Perform prediction on given input example"""
        self.model.eval()
        model_out = {}
        # prepare input
        tokenized_sentence = [START_TOKEN] + [t.lower() for t in input] + [CTX_TOKEN]
        numericalized = [self.input_vocab.stoi[token] if token in self.input_vocab.stoi else self.input_vocab.stoi[UNK_TOKEN] for token in tokenized_sentence]
        src_tensor = torch.LongTensor(numericalized).unsqueeze(0).to(self.device)

        # get ner, coref predictions
        encoder_step = self.model._predict_encoder(src_tensor)
        ner_out = encoder_step[NER].argmax(1).tolist()
        coref_out = encoder_step[COREF].argmax(1).tolist()

        # get logical form, predicate and type prediction
        lf_out = [self.lf_vocab.stoi[START_TOKEN]]
        coref_ranking_out = [self.lf_vocab.stoi[START_TOKEN]]
        predicate_out = [self.lf_vocab.stoi[START_TOKEN]]
        type_out = [self.lf_vocab.stoi[START_TOKEN]]

        for _ in range(self.model.decoder.max_positions):
            lf_tensor = torch.LongTensor(lf_out).unsqueeze(0).to(self.device)

            decoder_step = self.model._predict_decoder(src_tensor, lf_tensor, encoder_step['encoder_out'])

            pred_lf = decoder_step['decoder_out'].argmax(1)[-1].item()
            pred_coref_ranking = decoder_step[COREF_RANKING].argmax(1)[-1].item()
            pred_predicate = decoder_step[PREDICATE].argmax(1)[-1].item()
            pred_type = decoder_step[TYPE].argmax(1)[-1].item()

            if pred_lf == self.lf_vocab.stoi[END_TOKEN]:
                break

            lf_out.append(pred_lf)
            coref_ranking_out.append(pred_coref_ranking)
            predicate_out.append(pred_predicate)
            type_out.append(pred_type)

        # translate top predictions into vocab tokens
        model_out[NER] = [self.ner_vocab.itos[i] for i in ner_out][1:-1]
        model_out[COREF] = [self.coref_vocab.itos[i] for i in coref_out][1:-1]
        model_out[COREF_RANKING] = [self.coref_ranking_vocab.itos[i] for i in coref_ranking_out][1:]
        model_out[LOGICAL_FORM] = [self.lf_vocab.itos[i] for i in lf_out][1:]
        model_out[PREDICATE] = [self.predicate_vocab.itos[i] for i in predicate_out][1:]
        model_out[TYPE] = [self.type_vocab.itos[i] for i in type_out][1:]

        return model_out

class AccuracyMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.correct = 0
        self.wrong = 0
        self.accuracy = 0

    def update(self, gold, result):
        if gold == result:
            self.correct += 1
        else:
            self.wrong += 1

        self.accuracy = self.correct / (self.correct + self.wrong)

class Scorer(object):
    """Scorer class"""
    def __init__(self):
        self.tasks = ['total', NER, COREF, COREF_RANKING, LOGICAL_FORM, PREDICATE, TYPE]
        self.results = {
            'Overall': {task:AccuracyMeter() for task in self.tasks},
            'Clarification': {task:AccuracyMeter() for task in self.tasks},
            'Comparative Reasoning (All)': {task:AccuracyMeter() for task in self.tasks},
            'Logical Reasoning (All)': {task:AccuracyMeter() for task in self.tasks},
            'Quantitative Reasoning (All)': {task:AccuracyMeter() for task in self.tasks},
            'Simple Question (Coreferenced)': {task:AccuracyMeter() for task in self.tasks},
            'Simple Question (Direct)': {task:AccuracyMeter() for task in self.tasks},
            'Simple Question (Ellipsis)': {task:AccuracyMeter() for task in self.tasks},
            # -------------------------------------------
            'Verification (Boolean) (All)': {task:AccuracyMeter() for task in self.tasks},
            'Quantitative Reasoning (Count) (All)': {task:AccuracyMeter() for task in self.tasks},
            'Comparative Reasoning (Count) (All)': {task:AccuracyMeter() for task in self.tasks},
        }
        self.data_dict = []
        self.inference_actions = []

    def data_score(self, data, helper, predictor):
        """Score complete list of data"""
        for example, q_type  in zip(data, helper['question_type']):
            # prepare references
            ref_ner = example.ner
            ref_coref = example.coref
            ref_lf = [t.lower() for t in example.logical_form]
            ref_pred = example.predicate
            ref_type = example.type
            ref_coref_ranking = example.coref_ranking

            # get model hypothesis
            hypothesis = predictor.predict(example.input)

            # check correctness
            correct_ner = 1 if ref_ner == hypothesis[NER] else 0
            correct_coref = 1 if ref_coref == hypothesis[COREF] else 0
            correct_coref_ranking = 1 if ref_coref_ranking == hypothesis[COREF_RANKING] else 0
            correct_lf = 1 if ref_lf == hypothesis[LOGICAL_FORM] else 0
            correct_pred = 1 if ref_pred == hypothesis[PREDICATE] else 0
            correct_type = 1 if ref_type == hypothesis[TYPE] else 0

            # save results
            gold = 1
            res = 1 if correct_ner and correct_coref and correct_coref_ranking and correct_lf and correct_pred and correct_type else 0
            # Question type
            self.results[q_type]['total'].update(gold, res)
            self.results[q_type][NER].update(ref_ner, hypothesis[NER])
            self.results[q_type][COREF].update(ref_coref, hypothesis[COREF])
            self.results[q_type][COREF_RANKING].update(ref_coref_ranking, hypothesis[COREF_RANKING])
            self.results[q_type][LOGICAL_FORM].update(ref_lf, hypothesis[LOGICAL_FORM])
            self.results[q_type][PREDICATE].update(ref_pred, hypothesis[PREDICATE])
            self.results[q_type][TYPE].update(ref_type, hypothesis[TYPE])
            # Overall
            self.results['Overall']['total'].update(gold, res)
            self.results['Overall'][NER].update(ref_ner, hypothesis[NER])
            self.results['Overall'][COREF].update(ref_coref, hypothesis[COREF])
            self.results['Overall'][COREF_RANKING].update(ref_coref_ranking, hypothesis[COREF_RANKING])
            self.results['Overall'][LOGICAL_FORM].update(ref_lf, hypothesis[LOGICAL_FORM])
            self.results['Overall'][PREDICATE].update(ref_pred, hypothesis[PREDICATE])
            self.results['Overall'][TYPE].update(ref_type, hypothesis[TYPE])

            # save data
            self.data_dict.append({
                INPUT: example.input,
                NER: hypothesis[NER],
                f'{NER}_gold': example.ner,
                COREF: hypothesis[COREF],
                f'{COREF}_gold': example.coref,
                COREF_RANKING: hypothesis[COREF_RANKING],
                f'{COREF_RANKING}_gold': example.coref_ranking,
                LOGICAL_FORM: hypothesis[LOGICAL_FORM],
                f'{LOGICAL_FORM}_gold': example.logical_form,
                PREDICATE: hypothesis[PREDICATE],
                f'{PREDICATE}_gold': example.predicate,
                TYPE: hypothesis[TYPE],
                f'{TYPE}_gold': example.type,
                # ------------------------------------
                f'{NER}_correct': correct_ner,
                f'{COREF}_correct': correct_coref,
                f'{COREF_RANKING}_correct': correct_coref_ranking,
                f'{LOGICAL_FORM}_correct': correct_lf,
                f'{PREDICATE}_correct': correct_pred,
                f'{TYPE}_correct': correct_type,
                'is_correct': res,
                'question_type': q_type
            })

    def construct_inference_actions(self, inference_data, predictor):
        # prepare bert wordpiece tokenizer
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        tic = time.perf_counter()
        # based on model outpus create a final logical form to execute
        for i, sample in enumerate(inference_data):
            predictions = predictor.predict(sample['context_question'])
            actions = []
            logical_form_prediction = predictions[LOGICAL_FORM]
            for j, action in enumerate(logical_form_prediction):
                if action not in ['entity', 'relation', 'type', 'value', 'prev_answer']:
                    actions.append(['action', action])
                elif action == 'entity':
                    # get context entities
                    context_entities = set(sample['context_entities'])
                    # get predictions
                    context_question = sample['context_question']
                    ner_prediction = predictions[NER]
                    coref_prediction = predictions[COREF]
                    coref_ranking_prediction = predictions[COREF_RANKING]
                    # get their indices
                    ner_indices = [k for k, tag in enumerate(ner_prediction) if tag in ['B', 'I']]
                    coref_indices = [k for k, tag in enumerate(coref_prediction) if tag in ['1']]
                    coref_ranking_indices = {k:tag for k, tag in enumerate(coref_ranking_prediction) if tag not in ['0']}
                    # create a ner dictionary with index as key and entity as value
                    ner_idx_ent = self.create_ner_idx_ent_dict(ner_indices, tokenizer, context_question)
                    coref_idx_ent = OrderedDict({'0': 'NA'})
                    if not coref_indices:
                        # TODO here things get hard, we will need to use all ner entites and see if it works
                        print('No coref indices!')
                    else:
                        for l, cidx in enumerate(coref_indices):
                            if cidx in ner_idx_ent:
                                coref_idx_ent[str(l+1)] = ner_idx_ent[cidx]
                            else:
                                # TODO still not sure what we should do here
                                # coref_idx_ent[str(l+1)] = ...
                                print(f'Coref index {cidx} not in ner entities!')

                    if not coref_ranking_indices:
                        # TODO if no ranking available them use coref results, somehow
                        print('No coref ranking indices!')
                        actions.append(['entity', 'entity'])
                    elif j not in coref_ranking_indices:
                        # TODO if logical form entity index not in coref ranking them we need to do something else
                        print('Entity index not in coref ranking indices!')
                        actions.append(['entity', 'entity'])
                    else:
                        coref_rank = coref_ranking_indices[j]
                        if coref_rank not in coref_idx_ent:
                            # TODO again problem, we need to do something here also
                            print('Coref rank result not in coref list!')
                            actions.append(['entity', 'entity'])
                        else:
                            entity_candidates = coref_idx_ent[coref_rank]
                            gold_entity = entity_candidates[0] if entity_candidates else None # context_entities.intersection(entity_candidates)
                            if not entity_candidates:
                                # TODO do something for here also
                                print('No gold entity!')
                                actions.append(['entity', 'entity'])
                                # if entity_candidates: actions.append(['entity', entity_candidates[0]])
                            else:
                                actions.append(['entity', entity_candidates[0]]) # top entity as gold
                            # elif len(gold_entity) > 1:
                            #     print('More than 1 gold entities!')
                            #     # TODO do something for here also
                            #     actions.append(['entity', next(iter(gold_entity))])
                            # else:
                            #     # finally we got the entity
                            #     actions.append(['entity', next(iter(gold_entity))])
                elif action == 'relation':
                    predicate_prediction = predictions[PREDICATE]
                    if predicate_prediction[j].startswith('P'):
                        actions.append(['relation', predicate_prediction[j]])
                    else: # Predicate
                        print(f'Predicate prediction not in correct position: {sample}')
                elif action == 'type':
                    type_prediction = predictions[TYPE]
                    if type_prediction[j].startswith('Q'):
                        actions.append(['type', type_prediction[j]])
                    else: # Type
                        print(f'Type prediction not in correct position: {sample}')
                elif action == 'value':
                    actions.append(['value', self.get_value(sample['question'])])
                elif action == 'prev_answer':
                    actions.append(['entity', 'prev_answer'])

            self.inference_actions.append({
                'question_type': sample['question_type'],
                'question': sample['question'],
                'answer': sample['answer'],
                'actions': actions,
                'results': sample['results'],
                'prev_results': sample['prev_results'],
            })

            if (i+1) % 1000 == 0:
                toc = time.perf_counter()
                print(f'==> Finished action construction {((i+1)/len(inference_data))*100:.2f}% -- {toc - tic:0.2f}s')

        self.write_inference_actions()

    def create_ner_idx_ent_dict(self, ner_indices, tokenizer, context_question):
        es = Elasticsearch([{'host': 'localhost', 'port': 9200}]) # connect to elastic search server
        ent_idx = []
        ner_idx_ent = {}
        for index in ner_indices:
            if not ent_idx or index-1 == ent_idx[-1]:
                ent_idx.append(index)
            else:
                # get ent tokens from input context
                ent_tokens = [context_question[idx] for idx in ent_idx]
                # get string from tokens using tokenizer
                ent_string = tokenizer.convert_tokens_to_string(ent_tokens).replace('##', '')
                # get elastic search results
                es_results = self.elasticsearch_query(es, ent_string)
                # add idices to dict
                if es_results:
                    for idx in ent_idx:
                        ner_idx_ent[idx] = es_results
                # clean ent_idx
                ent_idx = [index]
        if ent_idx:
            # get ent tokens from input context
            ent_tokens = [context_question[idx] for idx in ent_idx]
            # get string from tokens using tokenizer
            ent_string = tokenizer.convert_tokens_to_string(ent_tokens).replace('##', '')
            # get elastic search results
            es_results = self.elasticsearch_query(es, ent_string)
            # add idices to dict
            if es_results:
                for idx in ent_idx:
                    ner_idx_ent[idx] = es_results
        return ner_idx_ent

    def elasticsearch_query(self, es, query):
        res = es.search(index='csqa_wikidata', doc_type='entities', body={'query': {'match': {'label': {'query': unidecode(query), 'fuzziness': 'AUTO'}}}})
        results = []
        for hit in res['hits']['hits']: results.append(hit["_source"]["id"])
        return results


    def get_value(self, question):
        if 'min' in question.split():
            value = '0'
        elif 'max' in question.split():
            value = '0'
        elif 'exactly' in question.split():
            value = re.search(r'\d+', question.split('exactly')[1]).group()
        elif 'approximately' in question.split():
            value = re.search(r'\d+', question.split('approximately')[1]).group()
        elif 'around' in question.split():
            value = re.search(r'\d+', question.split('around')[1]).group()
        elif 'atmost' in question.split():
            value = re.search(r'\d+', question.split('atmost')[1]).group()
        elif 'atleast' in question.split():
            value = re.search(r'\d+', question.split('atleast')[1]).group()
        else:
            print(f'Could not eract value from question: {question}')
            value = '0'

        return value # int(value)

    def write_inference_actions(self):
        with open(f'{ROOT_PATH}/{args.path_inference}/inference_actions.json', 'w', encoding='utf-8') as json_file:
            json_file.write(json.dumps(self.inference_actions, indent=4))

    def write_results(self):
        save_dict = json.dumps(self.data_dict, indent=4)
        save_dict_no_space_1 = re.sub(r'": \[\s+', '": [', save_dict)
        save_dict_no_space_2 = re.sub(r'",\s+', '", ', save_dict_no_space_1)
        save_dict_no_space_3 = re.sub(r'"\s+\]', '"]', save_dict_no_space_2)
        with open(f'{ROOT_PATH}/{args.path_error_analysis}/error_analysis.json', 'w', encoding='utf-8') as json_file:
            json_file.write(save_dict_no_space_3)

    def reset(self):
        """Reset object properties"""
        self.results = []
        self.instances = 0

def save_checkpoint(state):
    filename = f'{ROOT_PATH}/{args.snapshots}/ConvQA_model_e{state["epoch"]}_v-{state["best_val"]:.4f}.pth.tar'
    torch.save(state, filename)

def init_weights(model):
    # initialize model parameters with Glorot / fan_avg
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

class PredicateType(nn.Module):
    '''Single Task Loss'''
    def __init__(self, ignore_index):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, output, target):
        return self.criterion(output, target)

class SingleTaskLoss(nn.Module):
    '''Single Task Loss'''
    def __init__(self, ignore_index):
        super().__init__()
        self.criterion = nn.CrossEntropyLoss(ignore_index=ignore_index)

    def forward(self, output, target):
        return self.criterion(output, target)

class MultiTaskLoss(nn.Module):
    '''Multi Task Learning Loss'''
    def __init__(self, ignore_index):
        super().__init__()
        self.ner_loss = SingleTaskLoss(ignore_index)
        self.coref_loss = SingleTaskLoss(ignore_index)
        self.coref_ranking_loss = SingleTaskLoss(ignore_index)
        self.lf_loss = SingleTaskLoss(ignore_index)
        self.predicate_loss = PredicateType(ignore_index)
        self.type_loss = PredicateType(ignore_index)

        self.mml_emp = torch.Tensor([True, True, True, True, True, True])
        self.log_vars = torch.nn.Parameter(torch.zeros(len(self.mml_emp)))

    def forward(self, output, target):
        # weighted loss
        task_losses = torch.stack((
            self.ner_loss(output['ner'], target['ner']),
            self.coref_loss(output['coref'], target['coref']),
            self.lf_loss(output['logical_form'], target['logical_form']),
            self.predicate_loss(output['predicate'], target['predicate']),
            self.type_loss(output['type'], target['type']),
            self.coref_ranking_loss(output['coref_ranking'], target['coref_ranking'])
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
            'coref_ranking': losses[5],
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
