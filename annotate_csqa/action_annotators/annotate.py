# pylint: disable=relative-beyond-top-level
# pylint: disable=import-error
# ----------------------------
# reload for debugging
import sys
sys.path.append("..")
from action_annotators.actions import ActionOperator
from action_annotators.simple import Simple
from action_annotators.verification import Verification
from action_annotators.logical import Logical
from action_annotators.quantitative import Quantitative
from action_annotators.comparative import Comparative
from action_annotators.clarification import Clarification

class ActionAnnotator:
    def __init__(self, kg):
        # load operations and annotators
        self.operator = ActionOperator(kg)
        self.simple_annotator = Simple(self.operator)
        self.verification_annotator = Verification(self.operator)
        self.quantitative_annotator = Quantitative(self.operator)
        self.logical_annotator = Logical(self.operator)
        self.comparative_annotator = Comparative(self.operator)
        self.clarification_annotator = Clarification(self.simple_annotator, self.quantitative_annotator, self.comparative_annotator)

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

            if user['question-type'] in ['Simple Question (Direct)', 'Simple Question (Coreferenced)', 'Simple Question (Ellipsis)']:
                user, system = self.simple_annotator(user, system)
            elif user['question-type'] == 'Verification (Boolean) (All)':
                user, system = self.verification_annotator(user, system, prev_system)
            elif user['question-type'] in ['Quantitative Reasoning (Count) (All)', 'Quantitative Reasoning (All)']:
                user, system = self.quantitative_annotator(user, system)
            elif user['question-type'] == 'Logical Reasoning (All)':
                user, system = self.logical_annotator(user, system)
            elif user['question-type'] in ['Comparative Reasoning (Count) (All)', 'Comparative Reasoning (All)']:
                user, system = self.comparative_annotator(user, system, prev_user, prev_system)
            elif user['question-type'] == 'Clarification':
                is_clarification = True
                next_user = conversation[2*(i+1)]
                next_system = conversation[2*(i+1) + 1]
                user, next_system = self.clarification_annotator({
                    'prev_user': prev_user,
                    'prev_system': prev_system,
                    'user': user,
                    'system': system,
                    'next_user': next_user,
                    'next_system': next_system
                })
            else:
                raise Exception(f'Unknown question type: {user}')

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