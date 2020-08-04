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

# import constants
from constants import *

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
        helper_data = {
            'question_type': []
        }
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
                coref_type = []
                coref_ranking = []
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

                    action_entities = [action[1] for action in next_system['gold_actions'] if action[0] == 'entity' and action[1] != 'prev_answer']
                    coref_ranking_entities = ['NA']

                    if i == 0: # NA + [SEP] + NA + [SEP] + current_question
                        input.extend([  'NA', SEP_TOKEN, 'NA', SEP_TOKEN])
                        ner_tag.extend(['O',  'O',       'O',  'O'])
                    else:
                        # add prev context user
                        for context in prev_user_conv['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append('O')

                        # add prev context answer
                        for context in prev_system_conv['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append('O')

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

                    # coref entities - prepare coref values
                    coref_ent = set([action[1] for action in next_system['gold_actions'] if action[0] == 'entity'])
                    for context in reversed(user['context'] + system['context'] + next_user['context']):
                        if context[2] in coref_ent and context[4] == 'B':
                            coref.append('1')
                            if context[3] == 'NA':
                                coref_type.append('NO_TYPE')
                            else:
                                if len(context[3]) == 1: raise ValueError('Was ist das!')
                                coref_type.append(context[3])
                            coref_ent.remove(context[2])
                        else:
                            coref.append('0')
                            coref_type.append('NA')

                        # coref ranking
                        if context[2] in action_entities and context[2] not in coref_ranking_entities:
                            coref_ranking_entities.append(context[2])

                    if i == 0:
                        coref.extend(['0', '0', '0', '0'])
                        coref_type.extend(['NA', 'NA', 'NA', 'NA'])
                    else:
                        coref.append('0')
                        coref_type.append('NA')
                        for context in reversed(prev_system_conv['context']):
                            if context[2] in coref_ent and context[4] == 'B':
                                coref.append('1')
                                if context[3] == 'NA':
                                    coref_type.append('NO_TYPE')
                                else:
                                    if len(context[3]) == 1: raise ValueError('Was ist das!')
                                    coref_type.append(context[3])
                                coref_ent.remove(context[2])
                            else:
                                coref.append('0')
                                coref_type.append('NA')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        coref.append('0')
                        coref_type.append('NA')
                        for context in reversed(prev_user_conv['context']):
                            if context[2] in coref_ent and context[4] == 'B':
                                coref.append('1')
                                if context[3] == 'NA':
                                    coref_type.append('NO_TYPE')
                                else:
                                    if len(context[3]) == 1: raise ValueError('Was ist das!')
                                    coref_type.append(context[3])
                                coref_ent.remove(context[2])
                            else:
                                coref.append('0')
                                coref_type.append('NA')

                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                    # get gold actions
                    gold_actions = next_system['gold_actions']

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

                    if 'gold_actions' not in system or system['is_spurious']: # skip if logical form is spurious
                        prev_user_conv = user.copy()
                        prev_system_conv = system.copy()
                        continue

                    action_entities = [action[1] for action in system['gold_actions'] if action[0] == 'entity' and action[1] != 'prev_answer']
                    coref_ranking_entities = ['NA']

                    if i == 0: # NA + [SEP] + NA + [SEP] + current_question
                        input.extend([  'NA', SEP_TOKEN, 'NA', SEP_TOKEN])
                        ner_tag.extend(['O',  'O',       'O',  'O'])
                    else:
                        # add prev context user
                        for context in prev_user_conv['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append('O')

                        # add prev context answer
                        for context in prev_system_conv['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append('O')

                    # user context
                    for context in user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}')

                    # coref entities - prepare coref values
                    coref_ent = set([action[1] for action in system['gold_actions'] if action[0] == 'entity'])
                    for context in reversed(user['context']):
                        if context[2] in coref_ent and context[4] == 'B' and user['description'] not in ['Simple Question|Mult. Entity', 'Verification|one entity, multiple entities (as object) referred indirectly']:
                            coref.append('1')
                            if context[3] == 'NA':
                                coref_type.append('NO_TYPE')
                            else:
                                if len(context[3]) == 1: raise ValueError('Was ist das!')
                                coref_type.append(context[3])
                            coref_ent.remove(context[2])
                        else:
                            coref.append('0')
                            coref_type.append('NA')

                        # coref ranking
                        if context[2] in action_entities and context[2] not in coref_ranking_entities:
                            coref_ranking_entities.append(context[2])

                    if i == 0:
                        coref.extend(['0', '0', '0', '0'])
                        coref_type.extend(['NA', 'NA', 'NA', 'NA'])
                    else:
                        coref.append('0')
                        coref_type.append('NA')
                        for context in reversed(prev_system_conv['context']):
                            if context[2] in coref_ent and context[4] == 'B' and user['description'] not in ['Simple Question|Mult. Entity', 'Verification|one entity, multiple entities (as object) referred indirectly']:
                                coref.append('1')
                                if context[3] == 'NA':
                                    coref_type.append('NO_TYPE')
                                else:
                                    if len(context[3]) == 1: raise ValueError('Was ist das!')
                                    coref_type.append(context[3])
                                coref_ent.remove(context[2])
                            else:
                                coref.append('0')
                                coref_type.append('NA')

                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        coref.append('0')
                        coref_type.append('NA')
                        for context in reversed(prev_user_conv['context']):
                            if context[2] in coref_ent and context[4] == 'B' and user['description'] not in ['Simple Question|Mult. Entity', 'Verification|one entity, multiple entities (as object) referred indirectly']:
                                coref.append('1')
                                if context[3] == 'NA':
                                    coref_type.append('NO_TYPE')
                                else:
                                    if len(context[3]) == 1: raise ValueError('Was ist das!')
                                    coref_type.append(context[3])
                                coref_ent.remove(context[2])
                            else:
                                coref.append('0')
                                coref_type.append('NA')

                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

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
                        coref_ranking.append(str(coref_ranking_entities.index('NA')))
                    elif action[0] == 'relation':
                        logical_form.append('relation')
                        predicate_cls.append(action[1])
                        type_cls.append('NA')
                        coref_ranking.append(str(coref_ranking_entities.index('NA')))
                    elif action[0] == 'type':
                        logical_form.append('type')
                        predicate_cls.append('NA')
                        type_cls.append(action[1])
                        coref_ranking.append(str(coref_ranking_entities.index('NA')))
                    elif action[0] == 'entity':
                        if action[1] == 'prev_answer':
                            logical_form.append('prev_answer')
                            coref_ranking.append(str(coref_ranking_entities.index('NA')))
                        else:
                            logical_form.append('entity')
                            if action[1] in coref_ranking_entities:
                                coref_ranking.append(str(coref_ranking_entities.index(action[1])))
                            else:
                                coref_ranking.append(str(coref_ranking_entities.index('NA')))
                        predicate_cls.append('NA')
                        type_cls.append('NA')
                    elif action[0] == 'value':
                        logical_form.append(action[0])
                        predicate_cls.append('NA')
                        type_cls.append('NA')
                        coref_ranking.append(str(coref_ranking_entities.index('NA')))
                    else:
                        raise Exception(f'Unkown logical form {action[0]}')

                assert len(input) == len(ner_tag)
                assert len(input) == len(coref)
                assert len(input) == len(coref_type)
                assert len(logical_form) == len(predicate_cls)
                assert len(logical_form) == len(type_cls)
                assert len(logical_form) == len(coref_ranking)

                input_data.append([input, logical_form, ner_tag, list(reversed(coref)), list(reversed(coref_type)), coref_ranking, predicate_cls, type_cls])
                helper_data['question_type'].append(user['question-type'])

        return input_data, helper_data

    # We consider the context from 2 previous questions as history and not only 1
    def _prepare_data_2_context_history(self, data):
        input_data = []
        helper_data = {
            'question_type': []
        }
        for conversation in data:
            prev_user_conv_1 = {}
            prev_system_conv_1 = {}
            prev_user_conv_2 = {}
            prev_system_conv_2 = {}
            is_clarification = False
            is_history_ner_spurious = False
            turns = len(conversation) // 2
            for i in range(turns):
                input = []
                logical_form = []
                ner_tag = []
                coref = []
                coref_ranking = []
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
                        prev_user_conv_2 = user.copy()
                        prev_system_conv_2 = system.copy()
                        prev_user_conv_1 = next_user.copy()
                        prev_system_conv_1 = next_system.copy()
                        if prev_user_conv_2['is_ner_spurious'] or \
                            prev_system_conv_2['is_ner_spurious'] or \
                            prev_user_conv_1['is_ner_spurious'] or \
                            prev_system_conv_1['is_ner_spurious']:
                            is_history_ner_spurious = True
                        continue

                    # skip if current ner is spurious
                    if user['is_ner_spurious'] or \
                        system['is_ner_spurious'] or \
                        next_user['is_ner_spurious'] or \
                        next_system['is_ner_spurious']:
                        is_history_ner_spurious = True
                        prev_user_conv_2 = user.copy()
                        prev_system_conv_2 = system.copy()
                        prev_user_conv_1 = next_user.copy()
                        prev_system_conv_1 = next_system.copy()
                        continue

                    # skip if no gold action (or spurious)
                    if 'gold_actions' not in next_system or next_system['is_spurious']:
                        prev_user_conv_2 = user.copy()
                        prev_system_conv_2 = system.copy()
                        prev_user_conv_1 = next_user.copy()
                        prev_system_conv_1 = next_system.copy()
                        continue

                    action_entities = [action[1] for action in next_system['gold_actions'] if action[0] == 'entity' and action[1] != 'prev_answer']
                    coref_ranking_entities = ['NA']

                    if i == 0:
                        input.extend([  'NA', SEP_TOKEN, 'NA', SEP_TOKEN])
                        ner_tag.extend(['0',  '0',       '0',  '0'])
                    else:
                        # add prev context user
                        for context in prev_user_conv_1['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append('O')

                        # add prev context answer
                        for context in prev_system_conv_1['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append('O')

                    # add prev context user
                    for context in user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}')
                        # coref ranking
                        if context[2] in action_entities and context[2] not in coref_ranking_entities:
                            coref_ranking_entities.append(context[2])

                    # sep token
                    input.append(SEP_TOKEN)
                    ner_tag.append('O')

                    # add prev context answer
                    for context in system['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}')
                        # coref ranking
                        if context[2] in action_entities and context[2] not in coref_ranking_entities:
                            coref_ranking_entities.append(context[2])

                    # sep token
                    input.append(SEP_TOKEN)
                    ner_tag.append('O')

                    # next user context
                    for context in next_user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}')

                    # coref entities - prepare coref values
                    coref_ent = set([action[1] for action in next_system['gold_actions'] if action[0] == 'entity'])
                    for context in reversed(next_user['context']):
                        if context[2] in coref_ent and context[4] == 'B':
                            coref.append('1')
                            coref_ent.remove(context[2])
                        else:
                            coref.append('0')

                        # coref ranking
                        if context[2] in action_entities and context[2] not in coref_ranking_entities:
                            coref_ranking_entities.append(context[2])

                    coref.append('0')
                    for context in reversed(system['context']):
                        if context[2] in coref_ent and context[4] == 'B':
                            coref.append('1')
                            coref_ent.remove(context[2])
                        else:
                            coref.append('0')

                        # coref ranking
                        if context[2] in action_entities and context[2] not in coref_ranking_entities:
                            coref_ranking_entities.append(context[2])

                    coref.append('0')
                    for context in reversed(user['context']):
                        if context[2] in coref_ent and context[4] == 'B':
                            coref.append('1')
                            coref_ent.remove(context[2])
                        else:
                            coref.append('0')

                        # coref ranking
                        if context[2] in action_entities and context[2] not in coref_ranking_entities:
                            coref_ranking_entities.append(context[2])

                    if i == 0:
                        coref.extend(['0', '0', '0', '0'])
                    else:
                        coref.append('0')
                        for context in reversed(prev_system_conv_1['context']):
                            if context[2] in coref_ent and context[4] == 'B':
                                coref.append('1')
                                coref_ent.remove(context[2])
                            else:
                                coref.append('0')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        coref.append('0')
                        for context in reversed(prev_user_conv_1['context']):
                            if context[2] in coref_ent and context[4] == 'B':
                                coref.append('1')
                                coref_ent.remove(context[2])
                            else:
                                coref.append('0')

                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                    # get gold actions
                    gold_actions = next_system['gold_actions']

                    # track context history
                    prev_user_conv_2 = user.copy()
                    prev_system_conv_2 = system.copy()
                    prev_user_conv_1 = next_user.copy()
                    prev_system_conv_1 = next_system.copy()

                else:

                    # skip if ner history is spurious
                    if is_history_ner_spurious:
                        is_history_ner_spurious = False
                        if i == 0:
                            prev_user_conv_2 = None
                            prev_system_conv_2 = None
                            prev_user_conv_1 = user.copy()
                            prev_system_conv_1 = system.copy()
                        else:
                            prev_user_conv_2 = prev_user_conv_1.copy()
                            prev_system_conv_2 = prev_system_conv_1.copy()
                            prev_user_conv_1 = user.copy()
                            prev_system_conv_1 = system.copy()
                        if prev_user_conv_2['is_ner_spurious'] or \
                            prev_system_conv_2['is_ner_spurious'] or \
                            prev_user_conv_1['is_ner_spurious'] or \
                            prev_system_conv_1['is_ner_spurious']:
                            is_history_ner_spurious = True
                        continue

                    # skip if current ner is spurious
                    if user['is_ner_spurious'] or system['is_ner_spurious']:
                        is_history_ner_spurious = True
                        if i == 0:
                            prev_user_conv_2 = None
                            prev_system_conv_2 = None
                            prev_user_conv_1 = user.copy()
                            prev_system_conv_1 = system.copy()
                        else:
                            prev_user_conv_2 = prev_user_conv_1.copy()
                            prev_system_conv_2 = prev_system_conv_1.copy()
                            prev_user_conv_1 = user.copy()
                            prev_system_conv_1 = system.copy()
                        continue

                    # skip if logical form is spurious
                    if 'gold_actions' not in system or system['is_spurious']:
                        if i == 0:
                            prev_user_conv_2 = None
                            prev_system_conv_2 = None
                            prev_user_conv_1 = user.copy()
                            prev_system_conv_1 = system.copy()
                        else:
                            prev_user_conv_2 = prev_user_conv_1.copy()
                            prev_system_conv_2 = prev_system_conv_1.copy()
                            prev_user_conv_1 = user.copy()
                            prev_system_conv_1 = system.copy()
                        continue

                    action_entities = [action[1] for action in system['gold_actions'] if action[0] == 'entity' and action[1] != 'prev_answer']
                    coref_ranking_entities = ['NA']

                    if i == 0: # NA + [SEP] NA + [SEP] + NA + [SEP] + NA + [SEP] + current_question
                        input.extend([  'NA', SEP_TOKEN, 'NA', SEP_TOKEN, 'NA', SEP_TOKEN, 'NA', SEP_TOKEN])
                        ner_tag.extend(['0',  '0',       '0',  '0',       'O',  'O',       'O',  'O'])
                    elif i == 1: # NA + [SEP] NA + [SEP] + prev_question_1 + [SEP] + prev_answer_1 + [SEP] + current_question
                        input.extend([  'NA', SEP_TOKEN, 'NA', SEP_TOKEN])
                        ner_tag.extend(['0',  '0',       '0',  '0'])
                        # add prev context user
                        for context in prev_user_conv_1['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append('O')

                        # add prev context answer
                        for context in prev_system_conv_1['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append('O')

                    else: # prev_question_2 + [SEP] prev_answer_2 + [SEP] + prev_question_1 + [SEP] + prev_answer_1 + [SEP] + current_question
                        # add prev context user
                        for context in prev_user_conv_2['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append('O')

                        # add prev context answer
                        for context in prev_system_conv_2['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append('O')

                        # add prev context user
                        for context in prev_user_conv_1['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append('O')

                        # add prev context answer
                        for context in prev_system_conv_1['context']:
                            input.append(context[1])
                            ner_tag.append(f'{context[-1]}')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        # sep token
                        input.append(SEP_TOKEN)
                        ner_tag.append('O')

                    # user context
                    for context in user['context']:
                        input.append(context[1])
                        ner_tag.append(f'{context[-1]}')

                    # coref entities - prepare coref values
                    coref_ent = set([action[1] for action in system['gold_actions'] if action[0] == 'entity'])
                    for context in reversed(user['context']):
                        if context[2] in coref_ent and context[4] == 'B' and user['description'] not in ['Simple Question|Mult. Entity', 'Verification|one entity, multiple entities (as object) referred indirectly']:
                            coref.append('1')
                            coref_ent.remove(context[2])
                        else:
                            coref.append('0')

                        # coref ranking
                        if context[2] in action_entities and context[2] not in coref_ranking_entities:
                            coref_ranking_entities.append(context[2])

                    if i == 0:
                        coref.extend(['0', '0', '0', '0', 'O', 'O', 'O', 'O'])
                    elif i == 1:
                        coref.append('0')
                        for context in reversed(prev_system_conv_1['context']):
                            if context[2] in coref_ent and context[4] == 'B':
                                coref.append('1')
                                coref_ent.remove(context[2])
                            else:
                                coref.append('0')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        coref.append('0')
                        for context in reversed(prev_user_conv_1['context']):
                            if context[2] in coref_ent and context[4] == 'B':
                                coref.append('1')
                                coref_ent.remove(context[2])
                            else:
                                coref.append('0')

                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        coref.extend(['0', '0', '0', '0'])
                    else:
                        coref.append('0')
                        for context in reversed(prev_system_conv_1['context']):
                            if context[2] in coref_ent and context[4] == 'B':
                                coref.append('1')
                                coref_ent.remove(context[2])
                            else:
                                coref.append('0')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        coref.append('0')
                        for context in reversed(prev_user_conv_1['context']):
                            if context[2] in coref_ent and context[4] == 'B':
                                coref.append('1')
                                coref_ent.remove(context[2])
                            else:
                                coref.append('0')

                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        coref.append('0')
                        for context in reversed(prev_system_conv_2['context']):
                            if context[2] in coref_ent and context[4] == 'B':
                                coref.append('1')
                                coref_ent.remove(context[2])
                            else:
                                coref.append('0')
                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                        coref.append('0')
                        for context in reversed(prev_user_conv_2['context']):
                            if context[2] in coref_ent and context[4] == 'B':
                                coref.append('1')
                                coref_ent.remove(context[2])
                            else:
                                coref.append('0')

                            # coref ranking
                            if context[2] in action_entities and context[2] not in coref_ranking_entities:
                                coref_ranking_entities.append(context[2])

                    # get gold actions
                    gold_actions = system['gold_actions']

                    # track context history
                    if i == 0:
                        prev_user_conv_2 = None
                        prev_system_conv_2 = None
                        prev_user_conv_1 = user.copy()
                        prev_system_conv_1 = system.copy()
                    else:
                        prev_user_conv_2 = prev_user_conv_1.copy()
                        prev_system_conv_2 = prev_system_conv_1.copy()
                        prev_user_conv_1 = user.copy()
                        prev_system_conv_1 = system.copy()

                # prepare logical form
                for action in gold_actions:
                    if action[0] == 'action':
                        logical_form.append(action[1])
                        predicate_cls.append('NA')
                        type_cls.append('NA')
                        coref_ranking.append(str(coref_ranking_entities.index('NA')))
                    elif action[0] == 'relation':
                        logical_form.append('relation')
                        predicate_cls.append(action[1])
                        type_cls.append('NA')
                        coref_ranking.append(str(coref_ranking_entities.index('NA')))
                    elif action[0] == 'type':
                        logical_form.append('type')
                        predicate_cls.append('NA')
                        type_cls.append(action[1])
                        coref_ranking.append(str(coref_ranking_entities.index('NA')))
                    elif action[0] == 'entity':
                        if action[1] == 'prev_answer':
                            logical_form.append('prev_answer')
                            coref_ranking.append(str(coref_ranking_entities.index('NA')))
                        else:
                            logical_form.append('entity')
                            if action[1] in coref_ranking_entities:
                                coref_ranking.append(str(coref_ranking_entities.index(action[1])))
                            else:
                                coref_ranking.append(str(coref_ranking_entities.index('NA')))
                        predicate_cls.append('NA')
                        type_cls.append('NA')
                    elif action[0] == 'value':
                        logical_form.append(action[0])
                        predicate_cls.append('NA')
                        type_cls.append('NA')
                        coref_ranking.append(str(coref_ranking_entities.index('NA')))
                    else:
                        raise Exception(f'Unkown logical form {action[0]}')

                assert len(input) == len(ner_tag)
                assert len(input) == len(coref)
                assert len(logical_form) == len(predicate_cls)
                assert len(logical_form) == len(type_cls)
                assert len(logical_form) == len(coref_ranking)

                input_data.append([input, logical_form, ner_tag, list(reversed(coref)), predicate_cls, type_cls, coref_ranking])
                helper_data['question_type'].append(user['question-type'])

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
                        input.extend([  'NA', SEP_TOKEN, 'NA', SEP_TOKEN])
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
                    gold_actions = next_system['gold_actions'] if 'gold_actions' in next_system else None
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
                        input.extend([  'NA', SEP_TOKEN, 'NA', SEP_TOKEN])
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
                    gold_actions = system['gold_actions'] if 'gold_actions' in system else None
                    prev_results = prev_system_conv['all_entities'] if 'all_entities' in prev_system_conv else None
                    context_entities = user['entities_in_utterance'] + system['entities_in_utterance']
                    if 'entities_in_utterance' in prev_user_conv: context_entities.extend(prev_user_conv['entities_in_utterance'])
                    if 'entities_in_utterance' in prev_system_conv: context_entities.extend(prev_system_conv['entities_in_utterance'])

                    # track context history
                    prev_user_conv = user.copy()
                    prev_system_conv = system.copy()

                inference_data.append({
                    'question_type': user['question-type'],
                    'question': user['utterance'],
                    'context_question': input,
                    'context_entities': context_entities,
                    'answer': answer,
                    'results': results,
                    'prev_results': prev_results,
                    'gold_actions': gold_actions
                })

        return inference_data

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

        self.coref_type_field = Field(init_token=NA_TOKEN,
                                eos_token=NA_TOKEN,
                                pad_token=PAD_TOKEN,
                                unk_token=NA_TOKEN,
                                batch_first=True)

        self.coref_ranking_field = Field(init_token='0',
                                eos_token='0',
                                pad_token=PAD_TOKEN,
                                unk_token='0',
                                batch_first=True)

        self.predicate_field = Field(init_token=NA_TOKEN,
                                eos_token=NA_TOKEN,
                                pad_token=NA_TOKEN,
                                unk_token=NA_TOKEN,
                                batch_first=True)

        self.type_field = Field(init_token=NA_TOKEN,
                                eos_token=NA_TOKEN,
                                pad_token=NA_TOKEN,
                                unk_token=NA_TOKEN,
                                batch_first=True)

        fields_tuple = [(INPUT, self.input_field), (LOGICAL_FORM, self.lf_field),
                        (NER, self.ner_field), (COREF, self.coref_field),
                        (COREF_TYPE, self.coref_type_field), (COREF_RANKING, self.coref_ranking_field),
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
        self.coref_type_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)
        self.coref_ranking_field.build_vocab(self.train_data, min_freq=0)
        self.predicate_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)
        self.type_field.build_vocab(self.train_data, self.val_data, self.test_data, min_freq=0)

    def get_data(self):
        """Return train, validation and test data objects"""
        return self.train_data, self.val_data, self.test_data

    def get_data_helper(self):
        """Return train, validation and test data helpers"""
        return self.train_helper, self.val_helper, self.test_helper

    def get_fields(self):
        """Return source and target field objects"""
        return {
            INPUT: self.input_field,
            LOGICAL_FORM: self.lf_field,
            NER: self.ner_field,
            COREF: self.coref_field,
            COREF_TYPE: self.coref_type_field,
            COREF_RANKING: self.coref_ranking_field,
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
            COREF_TYPE: self.coref_type_field.vocab,
            COREF_RANKING: self.coref_ranking_field.vocab,
            PREDICATE: self.predicate_field.vocab,
            TYPE: self.type_field.vocab
        }
