# pylint: disable=relative-beyond-top-level
# pylint: disable=import-error
# ----------------------------
# reload for debugging
import sys
sys.path.append("..")
import os
import sys
from glob import glob
from pathlib import Path
from unidecode import unidecode
from transformers import BertTokenizer
from ner_annotators.simple import Simple
from ner_annotators.verification import Verification
from ner_annotators.logical import Logical
from ner_annotators.quantitative import Quantitative
from ner_annotators.comparative import Comparative
from ner_annotators.clarification import Clarification
ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent

class NERAnnotator:
    def __init__(self, kg, partition):
        # read preprocessed data
        self.preprocessed_data = self._read_preprocessed_data(partition)

        # set tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased').tokenize

        # create annotators
        self.simple_annotator = Simple(kg, self.preprocessed_data, self.tokenizer)
        self.verification_annotator = Verification(kg, self.preprocessed_data, self.tokenizer)
        self.quantitative_annotator = Quantitative(kg, self.preprocessed_data, self.tokenizer)
        self.logical_annotator = Logical(kg, self.preprocessed_data, self.tokenizer)
        self.comparative_annotator = Comparative(kg, self.preprocessed_data, self.tokenizer)
        self.clarification_annotator = Clarification(kg, self.preprocessed_data, self.tokenizer, self.simple_annotator, self.quantitative_annotator, self.comparative_annotator)

    def _read_preprocessed_data(self, partition):
        # read preprocessed data
        preprocessed_utterance_paths = glob(f'{ROOT_PATH}/data/preprocessed_data/{partition}/*/*_context_utterance.txt')
        preprocessed_context_paths = glob(f'{ROOT_PATH}/data/preprocessed_data/{partition}/*/*_context.txt')
        preprocessed_entities_paths = glob(f'{ROOT_PATH}/data/preprocessed_data/{partition}/*/*_context_entities.txt')
        preprocessed_states_paths = glob(f'{ROOT_PATH}/data/preprocessed_data/{partition}/*/*_state.txt')

        # create preprocesssed data dict
        preprocessed_data = {
            'utterances': [],
            'contexts': [],
            'entities': [],
            'states': []
        }

        for u_path, c_path, e_path, s_path in zip(preprocessed_utterance_paths, preprocessed_context_paths, preprocessed_entities_paths, preprocessed_states_paths):
            partition = c_path.rsplit('/', 1)[0].rsplit('/', 1)[0].rsplit('/', 1)[-1]
            folder = c_path.rsplit('/', 1)[0].rsplit('/', 1)[-1]

            utterances = [line.rstrip('\n') for line in open(u_path)]
            contexts = [line.rstrip('\n') for line in open(c_path)]
            entities = [line.rstrip('\n') for line in open(e_path)]
            states = [line.rstrip('\n') for line in open(s_path)]

            assert len(utterances) == len(entities)

            preprocessed_data['utterances'].extend(utterances)
            preprocessed_data['contexts'].extend(contexts)
            preprocessed_data['entities'].extend(entities)
            preprocessed_data['states'].extend(states)

        return preprocessed_data

    def __call__(self, conversation):
        # process conversation
        prev_user = None
        prev_system = None
        new_conversation = []

        is_clarification = False
        turns = len(conversation) // 2

        for i in range(turns):
            if is_clarification:
                is_clarification = False
                continue
            user = conversation[2*i]
            system = conversation[2*i + 1]

            # apply unicode to all utterances
            user['utterance'] = unidecode(user['utterance'])
            system['utterance'] = unidecode(system['utterance'])

            if user['question-type'] == 'Simple Question (Direct)' or user['question-type'] == 'Simple Question (Coreferenced)' or user['question-type'] == 'Simple Question (Ellipsis)':
                user, system = self.simple_annotator(user, system)
            elif user['question-type'] == 'Verification (Boolean) (All)':
                user, system = self.verification_annotator(user, system, prev_system)
            elif user['question-type'] == 'Quantitative Reasoning (Count) (All)' or user['question-type'] == 'Quantitative Reasoning (All)':
                user, system = self.quantitative_annotator(user, system)
            elif user['question-type'] == 'Logical Reasoning (All)':
                user, system = self.logical_annotator(user, system)
            elif user['question-type'] == 'Comparative Reasoning (Count) (All)' or user['question-type'] == 'Comparative Reasoning (All)':
                user, system = self.comparative_annotator(user, system, prev_user, prev_system)
            elif user['question-type'] == 'Clarification':
                is_clarification = True
                next_user = conversation[2*(i+1)]
                next_system = conversation[2*(i+1) + 1]

                # apply unicode to all utterances
                next_user['utterance'] = unidecode(next_user['utterance'])
                next_system['utterance'] = unidecode(next_system['utterance'])

                user, system, next_user, next_system = self.clarification_annotator({
                    'user': user,
                    'system': system,
                    'next_user': next_user,
                    'next_system': next_system
                })
            else:
                raise Exception(f'Unknown question type {user["question-type"]}')

            prev_user = user
            prev_system = system
            new_conversation.append(user)
            new_conversation.append(system)

            if is_clarification:
                prev_user = next_user
                prev_system = next_system
                new_conversation.append(next_user)
                new_conversation.append(next_system)

        return new_conversation