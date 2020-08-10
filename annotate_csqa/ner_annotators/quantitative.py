"""
Quantitative Reasoning (All):
- Quantitative|Min/Max|Single entity type - Done
- Quantitative|Min/Max|Mult. entity type - Done
- Quantitative|Atleast/ Atmost/ Approx. the same/Equal|Mult. entity type - Done
- Quantitative|Atleast/ Atmost/ Approx. the same/Equal|Single entity type - Done

Quantitative Reasoning (Count) (All):
- Quantitative|Count|Logical operators - Done
- Quantitative|Count|Mult. entity type - Done
- Quantitative|Count|Single entity type - Done
- Quantitative|Count|Single entity type|Indirect - Done
- Quantitative|Count|Logical operators|Indirect - Done
- Quantitative|Count|Single entity type|Incomplete - Done
- Quantitative|Count over Atleast/ Atmost/ Approx. the same/Equal|Mult. entity type - Done
- Quantitative|Count over Atleast/ Atmost/ Approx. the same/Equal|Single entity type - Done
- Incomplete count-based ques - Done
"""
from ner_annotators.ner_base import NERBase
class Quantitative(NERBase):
    def __init__(self, kg, preprocessed_data, tokenizer):
        super().__init__(kg, preprocessed_data, tokenizer)

    def __call__(self, user, system):
        # Clarification questions, will be handled from clarification class
        if 'description' not in user:
            return user, system

        # Quantitative Reasoning (All)
        if user['description'] == 'Quantitative|Min/Max|Single entity type':
            return self.indirect_question(user, system)

        if user['description'] == 'Quantitative|Min/Max|Mult. entity type':
            return self.indirect_question(user, system)

        if user['description'] == 'Quantitative|Atleast/ Atmost/ Approx. the same/Equal|Mult. entity type':
            return self.indirect_question(user, system)

        if user['description'] == 'Quantitative|Atleast/ Atmost/ Approx. the same/Equal|Single entity type':
            return self.indirect_question(user, system)

        # Quantitative Reasoning (Count) (All)
        if user['description'] == 'Quantitative|Count|Logical operators':
            return self.new_direct_question(user, system)

        if user['description'] == 'Quantitative|Count|Mult. entity type':
            return self.new_direct_question(user, system)

        if user['description'] == 'Quantitative|Count|Single entity type':
            return self.new_direct_question(user, system)

        if user['description'] == 'Quantitative|Count|Single entity type|Indirect':
            return self.indirect_question(user, system)

        if user['description' ] == 'Quantitative|Count|Logical operators|Indirect':
            return self.indirect_question(user, system)

        if user['description'] == 'Quantitative|Count|Single entity type|Incomplete':
            return self.ellipsis_question(user, system)

        if user['description'] == 'Quantitative|Count over Atleast/ Atmost/ Approx. the same/Equal|Mult. entity type':
            return self.indirect_question(user, system)

        if user['description'] == 'Quantitative|Count over Atleast/ Atmost/ Approx. the same/Equal|Single entity type':
            return self.indirect_question(user, system)

        if user['description'] == 'Incomplete count-based ques':
            return self.ellipsis_question(user, system)

        raise Exception(f'Description could not be found: {user["description"]}')
