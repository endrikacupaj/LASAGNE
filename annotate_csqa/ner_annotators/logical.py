"""
Logical Reasoning (All):
- Logical|Difference|Multiple_Relation - Done
- Logical|Union|Single_Relation - Done
- Logical|Union|Multiple_Relation - Done
- Logical|Intersection|Single_Relation|Incomplete - Done
- Logical|Difference|Single_Relation|Incomplete - Done
- Logical|Difference|Single_Relation - Done
- Logical|Intersection|Single_Relation - Done
- Logical|Intersection|Multiple_Relation - Done
- Logical|Union|Single_Relation|Incomplete - Done
"""
from ner_annotators.ner_base import NERBase
class Logical(NERBase):
    def __init__(self, kg, preprocessed_data, tokenizer):
        super().__init__(kg, preprocessed_data, tokenizer)

    def __call__(self, user, system):
        # Logical Reasoning (All)
        if 'description' not in user:
            raise Exception(f'No description for question:\n{user}')

        if user['description'] == 'Logical|Difference|Multiple_Relation':
            return self.new_direct_question(user, system)

        if user['description'] == 'Logical|Union|Single_Relation':
            return self.new_direct_question(user, system)

        if user['description'] == 'Logical|Union|Multiple_Relation':
            return self.new_direct_question(user, system)

        if user['description'] == 'Logical|Intersection|Single_Relation|Incomplete':
            return self.ellipsis_question(user, system, key_word='also')

        if user['description'] == 'Logical|Difference|Single_Relation|Incomplete':
            return self.ellipsis_question(user, system, key_word='not')

        if user['description'] == 'Logical|Difference|Single_Relation':
            return self.new_direct_question(user, system)

        if user['description'] == 'Logical|Intersection|Single_Relation':
            return self.new_direct_question(user, system)

        if user['description'] == 'Logical|Intersection|Multiple_Relation':
            return self.new_direct_question(user, system)

        if user['description'] == 'Logical|Union|Single_Relation|Incomplete':
            return self.ellipsis_question(user, system, key_word='or')

        raise Exception(f'Description could not be found: {user["description"]}')
