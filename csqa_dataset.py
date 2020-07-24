"""CSQADataset"""
import os
import re
import json
import logging
import torch
import numpy as np
from glob import glob
from pathlib import Path
from transformers import BertTokenizer
from torchtext.data import Field, Example, Dataset
from utils import (INPUT, LOGICAL_FORM, NER, COREF, START_TOKEN, END_TOKEN,
                    CTX_TOKEN, PAD_TOKEN, UNK_TOKEN, SEP_TOKEN, PREDICATE, TYPE)

class CSQADataset(object):
    """CSQADataset class"""
    TOKENIZE_SEQ = lambda self, x: x.split()
    ROOT_PATH = Path(os.path.dirname(__file__)).parent

    def __init__(self, data_dir='/data/final'):
        self.train_path = str(self.ROOT_PATH) + data_dir + '/train/*'
        self.val_path = str(self.ROOT_PATH) + data_dir + '/val/*'
        self.test_path = str(self.ROOT_PATH) + data_dir + '/test/*'
        self.load_data_and_fields()

    def _prepare_data(self, data):
        input_data = []
        for conversation in data:
            prev_user = None
            prev_system = None
            is_clarification = False
            turns = len(conversation) // 2
            for i in range(turns):
                input = []
                logical_form = []
                ner_tag = []
                coref = []
                coref_gold_idx = []
                predicate_cls = []
                type_cls = []

                if is_clarification:
                    is_clarification = False
                    continue

                user = conversation[2*i]
                system = conversation[2*i + 1]

                # skip if example is spurious
                if user['question-type'] != 'Clarification':
                    if user['is_ner_spurious'] or system['is_ner_spurious'] or system['is_spurious']:
                        continue

                # skip if previous example was spurious
                if i > 0 and prev_user_conv['question-type'] != 'Clarification':
                    if prev_user_conv['is_ner_spurious'] or prev_system_conv['is_ner_spurious'] or prev_system_conv['is_spurious']:
                        continue

                if i == 0: # NA + [SEP] + NA + [SEP] + current_question
                    input.extend([  'NA', SEP_TOKEN, 'NA', SEP_TOKEN])
                    ner_tag.extend(['O',  'O',       'O',  'O'])
                else:
                    # add prev context user
                    for context in prev_user_conv['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}')

                    # sep token
                    input.append(SEP_TOKEN)
                    ner_tag.append('O')

                    # add prev context answer
                    for context in prev_system_conv['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}')

                    # sep token
                    input.append(SEP_TOKEN)
                    ner_tag.append('O')

                if user['question-type'] == 'Clarification':
                    # get next context
                    is_clarification = True
                    next_user = conversation[2*(i+1)]
                    next_system = conversation[2*(i+1) + 1]

                    # user context
                    for context in user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}')

                    # system context
                    for context in system['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}')

                    # next user context
                    for context in next_user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}')

                    if 'gold_actions' not in next_system:
                        continue

                    # coref entities - prepare coref values
                    coref_ent = set([action[1] for action in next_system['gold_actions'] if action[0] == 'entity'])
                    for context in reversed(user['context'] + system['context'] + next_user['context']):
                        if context[2] in coref_ent and context[4] == 'B':
                            coref.append('1')
                            coref_ent.remove(context[2])
                        else:
                            coref.append('0')

                    if i == 0:
                        coref.extend(['0', '0', '0', '0'])
                    if i > 0:
                        coref.append('0')
                        for context in reversed(prev_system_conv['context']):
                            if context[2] in coref_ent and context[4] == 'B':
                                coref.append('1')
                                coref_ent.remove(context[2])
                            else:
                                coref.append('0')
                        coref.append('0')
                        for context in reversed(prev_user_conv['context']):
                            if context[2] in coref_ent and context[4] == 'B':
                                coref.append('1')
                                coref_ent.remove(context[2])
                            else:
                                coref.append('0')

                    # get gold actions
                    gold_actions = next_system['gold_actions']

                    # track context history
                    prev_user_conv = next_user.copy()
                    prev_system_conv = next_system.copy()
                else:
                    # user context
                    for context in user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}')

                    if 'gold_actions' not in system:
                        continue

                    # coref entities - prepare coref values
                    coref_ent = set([action[1] for action in system['gold_actions'] if action[0] == 'entity'])
                    for context in reversed(user['context']):
                        if context[2] in coref_ent and context[4] == 'B':
                            coref.append('1')
                            coref_ent.remove(context[2])
                        else:
                            coref.append('0')

                    if i == 0:
                        coref.extend(['0', '0', '0', '0'])
                    if i > 0:
                        coref.append('0')
                        for context in reversed(prev_system_conv['context']):
                            if context[2] in coref_ent and context[4] == 'B':
                                coref.append('1')
                                coref_ent.remove(context[2])
                            else:
                                coref.append('0')
                        coref.append('0')
                        for context in reversed(prev_user_conv['context']):
                            if context[2] in coref_ent and context[4] == 'B':
                                coref.append('1')
                                coref_ent.remove(context[2])
                            else:
                                coref.append('0')

                    # get gold actions
                    gold_actions = system['gold_actions']

                    # track context history
                    prev_user_conv = user.copy()
                    prev_system_conv = system.copy()

                # prepare logical form
                for action in gold_actions:
                    if action[0] == 'action':
                        logical_form.append(action[1])
                        predicate_cls.append('NA')
                        type_cls.append('NA')
                    elif action[0] == 'relation':
                        logical_form.append('relation')
                        predicate_cls.append(action[1])
                        type_cls.append('NA')
                    elif action[0] == 'type':
                        logical_form.append('type')
                        predicate_cls.append('NA')
                        type_cls.append(action[1])
                    elif action[0] == 'entity':
                        if action[1] == 'prev_answer':
                            logical_form.append('prev_answer')
                        else:
                            logical_form.append('entity')
                        predicate_cls.append('NA')
                        type_cls.append('NA')
                    elif action[0] == 'value':
                        logical_form.append(action[0])
                        predicate_cls.append('NA')
                        type_cls.append('NA')
                    else:
                        raise Exception(f'Unkown logical form {action[0]}')

                assert len(input) == len(ner_tag)
                assert len(input) == len(coref)
                assert len(logical_form) == len(predicate_cls)
                assert len(logical_form) == len(type_cls)

                input_data.append([input, logical_form, ner_tag, list(reversed(coref)), predicate_cls, type_cls])

        return input_data

    def _make_torchtext_dataset(self, data, fields):
        examples = [Example.fromlist(i, fields) for i in data]
        return Dataset(examples, fields)

    def load_data_and_fields(self):
        """
        Load data
        Create source and target fields
        """
        train, val, test = [], [], []
        # read data
        train_files = glob(self.train_path + '/*.json')
        for f in train_files:
            with open(f) as json_file:
                train.append(json.load(json_file))

        val_files = glob(self.val_path + '/*.json')
        for f in val_files:
            with open(f) as json_file:
                val.append(json.load(json_file))

        test_files = glob(self.test_path + '/*.json')
        for f in test_files:
            with open(f) as json_file:
                test.append(json.load(json_file))

        # prepare data
        train = self._prepare_data(train)
        val = self._prepare_data(val)
        test = self._prepare_data(test)

        # create fields
        self.input_field = Field(init_token=START_TOKEN,
                                eos_token=CTX_TOKEN,
                                pad_token=PAD_TOKEN,
                                unk_token=UNK_TOKEN,
                                lower=True,
                                batch_first=True)

        self.lf_field = Field(init_token=START_TOKEN,
                                eos_token=END_TOKEN,
                                pad_token=PAD_TOKEN,
                                unk_token=UNK_TOKEN,
                                lower=True,
                                batch_first=True)

        self.ner_field = Field(init_token='O',
                                eos_token='O',
                                pad_token='O',
                                unk_token='O',
                                batch_first=True)

        self.coref_field = Field(init_token='0',
                                eos_token='0',
                                pad_token='0',
                                unk_token='0',
                                batch_first=True)

        self.predicate_field = Field(init_token='NA',
                                eos_token='NA',
                                pad_token='NA',
                                unk_token='NA',
                                batch_first=True)

        self.type_field = Field(init_token='NA',
                                eos_token='NA',
                                pad_token='NA',
                                unk_token='NA',
                                batch_first=True)

        fields_tuple = [(INPUT, self.input_field), (LOGICAL_FORM, self.lf_field),
                        (NER, self.ner_field), (COREF, self.coref_field),
                        (PREDICATE, self.predicate_field), (TYPE, self.type_field)]

        # create toechtext datasets
        self.train_data = self._make_torchtext_dataset(train, fields_tuple)
        self.val_data = self._make_torchtext_dataset(val, fields_tuple)
        self.test_data = self._make_torchtext_dataset(test, fields_tuple)

        # build vocabularies
        self.input_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0, vectors='glove.840B.300d')
        self.lf_field.build_vocab(self.train_data, min_freq=0)
        self.ner_field.build_vocab(self.train_data, min_freq=0)
        self.coref_field.build_vocab(self.train_data, min_freq=0)
        self.predicate_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)
        self.type_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)

    def get_data(self):
        """Return train, validation and test data objects"""
        return self.train_data, self.val_data, self.test_data

    def get_fields(self):
        """Return source and target field objects"""
        return {
            INPUT: self.input_field,
            LOGICAL_FORM: self.lf_field,
            NER: self.ner_field,
            COREF: self.coref_field,
            PREDICATE: self.predicate_field,
            TYPE: self.type_field
        }

    def get_vocabs(self):
        """Return source and target vocabularies"""
        return {
            INPUT: self.input_field.vocab,
            LOGICAL_FORM: self.lf_field.vocab,
            NER: self.ner_field.vocab,
            COREF: self.coref_field.vocab,
            PREDICATE: self.predicate_field.vocab,
            TYPE: self.type_field.vocab
        }
