"""
Verification (Boolean) (All):
- Verification|3 entities, 2 direct, 2(direct) are query entities, subject is indirect -
- Verification|3 entities, all direct, 2 are query entities -
- Verification|2 entities, one direct and one indirect, object is indirect -
- Verification|2 entities, one direct and one indirect, subject is indirect -
- Verification|2 entities, both direct -
- Verification|one entity, multiple entities (as object) referred indirectly -
"""
from ner_annotators.ner_base import NERBase
class Verification(NERBase):
    def __init__(self, kg, preprocessed_data, tokenizer):
        super().__init__(kg, preprocessed_data, tokenizer)

    def __call__(self, user, system, prev_system):
        # Verification (Boolean) (All)
        if user['description'] == 'Verification|3 entities, 2 direct, 2(direct) are query entities, subject is indirect':
            return self.new_direct_question(user, system, True)

        if user['description'] == 'Verification|3 entities, all direct, 2 are query entities':
            return self.new_direct_question(user, system, True)

        if user['description'] == 'Verification|2 entities, one direct and one indirect, object is indirect':
            return self.new_direct_question(user, system, True)

        if user['description'] == 'Verification|2 entities, one direct and one indirect, subject is indirect':
            return self.new_direct_question(user, system, True)

        if user['description'] == 'Verification|2 entities, both direct':
            return self.new_direct_question(user, system, True)

        if user['description'] == 'Verification|one entity, multiple entities (as object) referred indirectly':
            return self.new_direct_question(user, system, True)

        raise Exception(f'Description could not be found: {user["description"]}')
