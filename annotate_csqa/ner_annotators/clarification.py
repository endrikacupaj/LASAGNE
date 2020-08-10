"""
Clarification:
- Simple Question|Single Entity|Indirect - Done
- Comparative|More/Less|Single entity type|Indirect - Done
- Quantitative|Count|Single entity type|Indirect - Done
- Comparative|Count over More/Less|Single entity type|Indirect - Done
- Quantitative|Count|Logical operators|Indirect - Done
- Comparative|Count over More/Less|Mult. entity type|Indirect - Done
- Comparative|More/Less|Mult. entity type|Indirect - Done
"""
from ner_annotators.ner_base import NERBase
class Clarification(NERBase):
    def __init__(self, kg, preprocessed_data, tokenizer, simple, comparative, quantitative):
        super().__init__(kg, preprocessed_data, tokenizer)
        self.simple = simple
        self.comparative = comparative
        self.quantitative = quantitative

    def __call__(self, conv_chunk):
        # Clarification
        if 'description' not in conv_chunk['user']:
            raise Exception(f'No description for question:\n{conv_chunk["user"]}')

        if conv_chunk['user']['description'] == 'Simple Question|Single Entity|Indirect':
            user, system = self.indirect_question(conv_chunk['user'], conv_chunk['system'])
            next_user, next_system = self.clarification_question(conv_chunk['next_user'], conv_chunk['next_system'])
            return user, system, next_user, next_system

        if conv_chunk['user']['description'] == 'Comparative|More/Less|Single entity type|Indirect':
            user, system = self.indirect_question(conv_chunk['user'], conv_chunk['system'])
            next_user, next_system = self.clarification_question(conv_chunk['next_user'], conv_chunk['next_system'])
            return user, system, next_user, next_system

        if conv_chunk['user']['description'] == 'Quantitative|Count|Single entity type|Indirect':
            user, system = self.indirect_question(conv_chunk['user'], conv_chunk['system'])
            next_user, next_system = self.clarification_question(conv_chunk['next_user'], conv_chunk['next_system'])
            return user, system, next_user, next_system

        if conv_chunk['user']['description'] == 'Comparative|Count over More/Less|Single entity type|Indirect':
            user, system = self.indirect_question(conv_chunk['user'], conv_chunk['system'])
            next_user, next_system = self.clarification_question(conv_chunk['next_user'], conv_chunk['next_system'])
            return user, system, next_user, next_system

        if conv_chunk['user']['description'] == 'Quantitative|Count|Logical operators|Indirect':
            user, system = self.indirect_question(conv_chunk['user'], conv_chunk['system'])
            next_user, next_system = self.clarification_question(conv_chunk['next_user'], conv_chunk['next_system'])
            return user, system, next_user, next_system

        if conv_chunk['user']['description'] == 'Comparative|Count over More/Less|Mult. entity type|Indirect':
            user, system = self.indirect_question(conv_chunk['user'], conv_chunk['system'])
            next_user, next_system = self.clarification_question(conv_chunk['next_user'], conv_chunk['next_system'])
            return user, system, next_user, next_system

        if conv_chunk['user']['description'] == 'Comparative|More/Less|Mult. entity type|Indirect':
            user, system = self.indirect_question(conv_chunk['user'], conv_chunk['system'])
            next_user, next_system = self.clarification_question(conv_chunk['next_user'], conv_chunk['next_system'])
            return user, system, next_user, next_system

        raise Exception(f'Description could not be found: {conv_chunk["user"]["description"]}')
