"""
Comparative Reasoning (All):
- Comparative|More/Less|Single entity type - Done
- Comparative|More/Less|Mult. entity type - Done
- Comparative|More/Less|Single entity type|Indirect - Done
- Comparative|More/Less|Mult. entity type|Incomplete - Done
- Comparative|More/Less|Mult. entity type|Indirect - Done
- Comparative|More/Less|Single entity type|Incomplete - Done

Comparative Reasoning (Count) (All):
- Comparative|Count over More/Less|Single entity type|Incomplete - Done
- Comparative|Count over More/Less|Mult. entity type - Done
- Comparative|Count over More/Less|Single entity type|Indirect - Done
- Comparative|Count over More/Less|Mult. entity type|Indirect - Done
- Comparative|Count over More/Less|Single entity type - Done
- Comparative|Count over More/Less|Mult. entity type|Incomplete - Done
"""
from ner_annotators.ner_base import NERBase
class Comparative(NERBase):
    def __init__(self, kg, preprocessed_data, tokenizer):
        super().__init__(kg, preprocessed_data, tokenizer)

    def __call__(self, user, system, prev_user, prev_system):
        # Clarification questions, will be handled from clarification class
        if 'description' not in user:
            return user, system

        # Comparative Reasoning (All)
        if user['description'] == 'Comparative|More/Less|Single entity type':
            return self.new_direct_question(user, system)

        if user['description'] == 'Comparative|More/Less|Mult. entity type':
            return self.new_direct_question(user, system)

        if user['description'] == 'Comparative|More/Less|Single entity type|Indirect':
            return self.indirect_question(user, system)

        if user['description'] == 'Comparative|More/Less|Mult. entity type|Incomplete':
            return self.ellipsis_question(user, system)

        if user['description'] == 'Comparative|More/Less|Mult. entity type|Indirect':
            return self.indirect_question(user, system)

        if user['description'] == 'Comparative|More/Less|Single entity type|Incomplete':
            return self.ellipsis_question(user, system)

        # Comparative Reasoning (Count) (All)
        if user['description'] == 'Comparative|Count over More/Less|Single entity type|Incomplete':
            return self.ellipsis_question(user, system)

        if user['description'] == 'Comparative|Count over More/Less|Mult. entity type':
            return self.new_direct_question(user, system)

        if user['description'] == 'Comparative|Count over More/Less|Single entity type|Indirect':
            return self.indirect_question(user, system)

        if user['description'] == 'Comparative|Count over More/Less|Mult. entity type|Indirect':
            return self.indirect_question(user, system)

        if user['description'] == 'Comparative|Count over More/Less|Single entity type':
            return self.new_direct_question(user, system)

        if user['description'] == 'Comparative|Count over More/Less|Mult. entity type|Incomplete':
            return self.ellipsis_question(user, system)

        raise Exception(f'Description could not be found: {user["description"]}')
