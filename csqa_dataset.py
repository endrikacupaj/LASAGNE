import os
import re
import json
import random
import logging
import torch
import numpy as np
from glob import glob
from pathlib import Path
from transformers import BertTokenizer
from torchtext.data import Field, Example, Dataset

# import constants
from constants import *

class CSQADataset(object):
    TOKENIZE_SEQ = lambda self, x: x.split()
    ROOT_PATH = Path(os.path.dirname(__file__)).parent

    def __init__(self):
        self.train_path = str(self.ROOT_PATH) + args.data_path + '/train/*'
        self.val_path = str(self.ROOT_PATH) + args.data_path + '/val/*'
        self.test_path = str(self.ROOT_PATH) + args.data_path + '/test/*'
        self.load_data_and_fields()

    def _prepare_data(self, data):
        input_data = []
        helper_data = {QUESTION_TYPE: []}
        for conversation in data:
            prev_user_conv = None
            prev_system_conv = None
            is_clarification = False
            is_history_ner_spurious = False
            turns = len(conversation) // 2
            for i in range(turns):
                input = []
                logical_form = []
                ner_tag = []
                coref = []
                predicate_cls = []
                type_cls = []

                if is_clarification:
                    is_clarification = False
                    continue

                user = conversation[2*i]
                system = conversation[2*i + 1]

                if user['question-type'] == 'Clarification':
                    # get next context
                    is_clarification = True
                    next_user = conversation[2*(i+1)]
                    next_system = conversation[2*(i+1) + 1]

                    # skip if ner history is spurious
                    if is_history_ner_spurious:
                        is_history_ner_spurious = False
                        if not next_user['is_ner_spurious'] and not next_system['is_ner_spurious']:
                            prev_user_conv = next_user.copy()
                            prev_system_conv = next_system.copy()
                        else:
                            is_history_ner_spurious = True
                        continue

                    # skip if ner is spurious
                    if user['is_ner_spurious'] or system['is_ner_spurious'] or next_user['is_ner_spurious'] or next_system['is_ner_spurious']:
                        is_history_ner_spurious = True
                        continue

                    # skip if no gold action (or spurious)
                    if 'gold_actions' not in next_system or next_system['is_spurious']:
                        prev_user_conv = next_user.copy()
                        prev_system_conv = next_system.copy()
                        continue

                    if i == 0: # NA + [SEP] + NA + [SEP] + current_question
                        input.extend([NA_TOKEN, SEP_TOKEN, NA_TOKEN, SEP_TOKEN])
                        ner_tag.extend([O, O, O, O])
                    else:
                        # add prev context user
                        for context in prev_user_conv['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append(O)

                        # add prev context answer
                        for context in prev_system_conv['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append(O)

                    # user context
                    for context in user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                    # system context
                    for context in system['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                    # next user context
                    for context in next_user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                    # coref entities - prepare coref values
                    action_entities = [action[1] for action in next_system[GOLD_ACTIONS] if action[0] == ENTITY]
                    for context in reversed(user['context'] + system['context'] + next_user['context']):
                        if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref:
                            coref.append(str(action_entities.index(context[2])))
                        else:
                            coref.append(NA_TOKEN)

                    if i == 0:
                        coref.extend([NA_TOKEN, NA_TOKEN, NA_TOKEN, NA_TOKEN])
                    else:
                        coref.append(NA_TOKEN)
                        for context in reversed(prev_system_conv['context']):
                            if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref:
                                coref.append(str(action_entities.index(context[2])))
                            else:
                                coref.append(NA_TOKEN)

                        coref.append(NA_TOKEN)
                        for context in reversed(prev_user_conv['context']):
                            if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref:
                                coref.append(str(action_entities.index(context[2])))
                            else:
                                coref.append(NA_TOKEN)

                    # get gold actions
                    gold_actions = next_system[GOLD_ACTIONS]

                    # track context history
                    prev_user_conv = next_user.copy()
                    prev_system_conv = next_system.copy()
                else:
                    if is_history_ner_spurious: # skip if history is ner spurious
                        is_history_ner_spurious = False
                        if not user['is_ner_spurious'] and not system['is_ner_spurious']:
                            prev_user_conv = user.copy()
                            prev_system_conv = system.copy()
                        else:
                            is_history_ner_spurious = True

                        continue
                    if user['is_ner_spurious'] or system['is_ner_spurious']: # skip if ner is spurious
                        is_history_ner_spurious = True
                        continue

                    if GOLD_ACTIONS not in system or system['is_spurious']: # skip if logical form is spurious
                        prev_user_conv = user.copy()
                        prev_system_conv = system.copy()
                        continue

                    if i == 0: # NA + [SEP] + NA + [SEP] + current_question
                        input.extend([NA_TOKEN, SEP_TOKEN, NA_TOKEN, SEP_TOKEN])
                        ner_tag.extend([O, O, O, O])
                    else:
                        # add prev context user
                        for context in prev_user_conv['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append(O)

                        # add prev context answer
                        for context in prev_system_conv['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append(O)

                    # user context
                    for context in user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}-{context[-2]}' if context[-1] in [B, I] else context[-1])

                    # coref entities - prepare coref values
                    action_entities = [action[1] for action in system[GOLD_ACTIONS] if action[0] == ENTITY]
                    for context in reversed(user['context']):
                        if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref and user['description'] not in ['Simple Question|Mult. Entity', 'Verification|one entity, multiple entities (as object) referred indirectly']:
                            coref.append(str(action_entities.index(context[2])))
                        else:
                            coref.append(NA_TOKEN)

                    if i == 0:
                        coref.extend([NA_TOKEN, NA_TOKEN, NA_TOKEN, NA_TOKEN])
                    else:
                        coref.append(NA_TOKEN)
                        for context in reversed(prev_system_conv['context']):
                            if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref and user['description'] not in ['Simple Question|Mult. Entity', 'Verification|one entity, multiple entities (as object) referred indirectly']:
                                coref.append(str(action_entities.index(context[2])))
                            else:
                                coref.append(NA_TOKEN)

                        coref.append(NA_TOKEN)
                        for context in reversed(prev_user_conv['context']):
                            if context[2] in action_entities and context[4] == B and str(action_entities.index(context[2])) not in coref and user['description'] not in ['Simple Question|Mult. Entity', 'Verification|one entity, multiple entities (as object) referred indirectly']:
                                coref.append(str(action_entities.index(context[2])))
                            else:
                                coref.append(NA_TOKEN)

                    # get gold actions
                    gold_actions = system[GOLD_ACTIONS]

                    # track context history
                    prev_user_conv = user.copy()
                    prev_system_conv = system.copy()

                # prepare logical form
                for action in gold_actions:
                    if action[0] == ACTION:
                        logical_form.append(action[1])
                        predicate_cls.append(NA_TOKEN)
                        type_cls.append(NA_TOKEN)
                    elif action[0] == RELATION:
                        logical_form.append(RELATION)
                        predicate_cls.append(action[1])
                        type_cls.append(NA_TOKEN)
                    elif action[0] == TYPE:
                        logical_form.append(TYPE)
                        predicate_cls.append(NA_TOKEN)
                        type_cls.append(action[1])
                    elif action[0] == ENTITY:
                        logical_form.append(PREV_ANSWER if action[1] == PREV_ANSWER else ENTITY)
                        predicate_cls.append(NA_TOKEN)
                        type_cls.append(NA_TOKEN)
                    elif action[0] == VALUE:
                        logical_form.append(action[0])
                        predicate_cls.append(NA_TOKEN)
                        type_cls.append(NA_TOKEN)
                    else:
                        raise Exception(f'Unkown logical form action {action[0]}')

                assert len(input) == len(ner_tag)
                assert len(input) == len(coref)
                assert len(logical_form) == len(predicate_cls)
                assert len(logical_form) == len(type_cls)

                input_data.append([input, logical_form, ner_tag, list(reversed(coref)), predicate_cls, type_cls])
                helper_data[QUESTION_TYPE].append(user['question-type'])

        return input_data, helper_data

    def get_inference_data(self, inference_partition):
        if inference_partition == 'val':
            files = glob(self.val_path + '/*.json')
        elif inference_partition == 'test':
            files = glob(self.test_path + '/*.json')
        else:
            raise ValueError(f'Unknown inference partion {inference_partition}')

        partition = []
        for f in files:
            with open(f) as json_file:
                partition.append(json.load(json_file))

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').tokenize
        inference_data = []

        for conversation in partition:
            is_clarification = False
            prev_user_conv = {}
            prev_system_conv = {}
            turns = len(conversation) // 2
            for i in range(turns):
                input = []
                question_type = ''
                gold_entities = []

                if is_clarification:
                    is_clarification = False
                    continue

                user = conversation[2*i]
                system = conversation[2*i + 1]

                if i > 0 and 'context' not in prev_system_conv:
                    if len(prev_system_conv['entities_in_utterance']) > 0:
                        tok_utterance = tokenizer(prev_system_conv['utterance'].lower())
                        prev_system_conv['context'] = [[i, tok] for i, tok in enumerate(tok_utterance)]
                    elif prev_system_conv['utterance'].isnumeric():
                        prev_system_conv['context'] = [[0, 'num']]
                    elif prev_system_conv['utterance'] == 'YES':
                        prev_system_conv['context'] = [[0, 'yes']]
                    elif prev_system_conv['utterance'] == 'NO':
                        prev_system_conv['context'] = [[0, 'no']]
                    elif prev_system_conv['utterance'] == 'YES and NO respectively':
                        prev_system_conv['context'] = [[0, 'no']]
                    elif prev_system_conv['utterance'] == 'NO and YES respectively':
                        prev_system_conv['context'] = [[0, 'no']]
                    elif prev_system_conv['utterance'][0].isnumeric():
                        prev_system_conv['context'] = [[0, 'num']]

                if user['question-type'] == 'Clarification':
                    # get next context
                    is_clarification = True
                    next_user = conversation[2*(i+1)]
                    next_system = conversation[2*(i+1) + 1]

                    if i == 0: # NA + [SEP] + NA + [SEP] + current_question
                        input.extend([NA_TOKEN, SEP_TOKEN, NA_TOKEN, SEP_TOKEN])
                    else:
                        # add prev context user
                        for context in prev_user_conv['context']:
                            input.append(context[1])

                        # sep token
                        input.append(SEP_TOKEN)

                        # add prev context answer
                        for context in prev_system_conv['context']:
                            input.append(context[1])

                        # sep token
                        input.append(SEP_TOKEN)

                    # user context
                    for context in user['context']:
                        input.append(context[1])

                    # system context
                    for context in system['context']:
                        input.append(context[1])

                    # next user context
                    for context in next_user['context']:
                        input.append(context[1])

                    results = next_system['all_entities']
                    answer = next_system['utterance']
                    gold_actions = next_system[GOLD_ACTIONS] if GOLD_ACTIONS in next_system else None
                    prev_answer = prev_system_conv['all_entities'] if 'all_entities' in prev_system_conv else None
                    context_entities = user['entities_in_utterance'] + system['entities_in_utterance']
                    if 'entities_in_utterance' in next_user: context_entities.extend(next_user['entities_in_utterance'])
                    if 'entities_in_utterance' in prev_user_conv: context_entities.extend(prev_user_conv['entities_in_utterance'])
                    if 'entities_in_utterance' in prev_system_conv: context_entities.extend(prev_system_conv['entities_in_utterance'])

                    # track context history
                    prev_user_conv = next_user.copy()
                    prev_system_conv = next_system.copy()
                else:
                    if i == 0: # NA + [SEP] + NA + [SEP] + current_question
                        input.extend([NA_TOKEN, SEP_TOKEN, NA_TOKEN, SEP_TOKEN])
                    else:
                        # add prev context user
                        for context in prev_user_conv['context']:
                            input.append(context[1])

                        # sep token
                        input.append(SEP_TOKEN)

                        # add prev context answer
                        for context in prev_system_conv['context']:
                            input.append(context[1])

                        # sep token
                        input.append(SEP_TOKEN)

                    if 'context' not in user:
                        tok_utterance = tokenizer(user['utterance'].lower())
                        user['context'] = [[i, tok] for i, tok in enumerate(tok_utterance)]

                    # user context
                    for context in user['context']:
                        input.append(context[1])

                    question = user['utterance']
                    results = system['all_entities']
                    answer = system['utterance']
                    gold_actions = system[GOLD_ACTIONS] if GOLD_ACTIONS in system else None
                    prev_results = prev_system_conv['all_entities'] if 'all_entities' in prev_system_conv else None
                    context_entities = user['entities_in_utterance'] + system['entities_in_utterance']
                    if 'entities_in_utterance' in prev_user_conv: context_entities.extend(prev_user_conv['entities_in_utterance'])
                    if 'entities_in_utterance' in prev_system_conv: context_entities.extend(prev_system_conv['entities_in_utterance'])

                    # track context history
                    prev_user_conv = user.copy()
                    prev_system_conv = system.copy()

                inference_data.append({
                    QUESTION_TYPE: user['question-type'],
                    QUESTION: user['utterance'],
                    CONTEXT_QUESTION: input,
                    CONTEXT_ENTITIES: context_entities,
                    ANSWER: answer,
                    RESULTS: results,
                    PREV_RESULTS: prev_results,
                    GOLD_ACTIONS: gold_actions
                })

        return inference_data

    def _make_torchtext_dataset(self, data, fields):
        examples = [Example.fromlist(i, fields) for i in data]
        return Dataset(examples, fields)

    def load_data_and_fields(self):
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
        train, self.train_helper = self._prepare_data(train)
        val, self.val_helper = self._prepare_data(val)
        test, self.test_helper = self._prepare_data(test)

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

        self.ner_field = Field(init_token=O,
                                eos_token=O,
                                pad_token=PAD_TOKEN,
                                unk_token=O,
                                batch_first=True)

        self.coref_field = Field(init_token='0',
                                eos_token='0',
                                pad_token=PAD_TOKEN,
                                unk_token='0',
                                batch_first=True)

        self.predicate_field = Field(init_token=NA_TOKEN,
                                eos_token=NA_TOKEN,
                                pad_token=PAD_TOKEN,
                                unk_token=NA_TOKEN,
                                batch_first=True)

        self.type_field = Field(init_token=NA_TOKEN,
                                eos_token=NA_TOKEN,
                                pad_token=PAD_TOKEN,
                                unk_token=NA_TOKEN,
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
        self.lf_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)
        self.ner_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)
        self.coref_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)
        self.predicate_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)
        self.type_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)

    def get_data(self):
        return self.train_data, self.val_data, self.test_data

    def get_data_helper(self):
        return self.train_helper, self.val_helper, self.test_helper

    def get_fields(self):
        return {
            INPUT: self.input_field,
            LOGICAL_FORM: self.lf_field,
            NER: self.ner_field,
            COREF: self.coref_field,
            PREDICATE: self.predicate_field,
            TYPE: self.type_field
        }

    def get_vocabs(self):
        return {
            INPUT: self.input_field.vocab,
            LOGICAL_FORM: self.lf_field.vocab,
            NER: self.ner_field.vocab,
            COREF: self.coref_field.vocab,
            PREDICATE: self.predicate_field.vocab,
            TYPE: self.type_field.vocab
        }
