"""
Simple Question (Direct)
- Simple Question|Single Entity - Done
- Simple Question - Done
- Simple Question|Mult. Entity|Indirect - Done

Simple Question (Coreferenced)
- Simple Question|Mult. Entity - Done
- Simple Question|Single Entity|Indirect - Done

Simple Question (Ellipsis)
- only subject is changed, parent and predicate remains same - DONE
- Incomplete|object parent is changed, subject and predicate remain same - DONE
"""
from ner_annotators.ner_base import NERBase
class Simple(NERBase):
    def __init__(self, kg, preprocessed_data, tokenizer):
        super().__init__(kg, preprocessed_data, tokenizer)

    def __call__(self, user, system):
        # Clarification questions, will be handled from clarification class
        if 'description' not in user:
            return user, system

        # Simple Question (Direct)
        if user['description'] == 'Simple Question|Single Entity':
            return self.new_direct_question(user, system)

        if user['description'] == 'Simple Question':
            return self.new_direct_question(user, system)

        if user['description'] == 'Simple Question|Mult. Entity|Indirect':
            return self.new_direct_question(user, system)

        # Simple Question (Coreferenced)
        if user['description'] == 'Simple Question|Mult. Entity':
            return self.indirect_question(user, system)

        if user['description'] == 'Simple Question|Single Entity|Indirect':
            return self.indirect_question(user, system)

        # Simple Question (Ellipsis)
        if user['description'] == 'only subject is changed, parent and predicate remains same':
            return self.ellipsis_question(user, system)

        if user['description'] == 'Incomplete|object parent is changed, subject and predicate remain same':
            return self.ellipsis_question(user, system, key_word='which')

        raise Exception(f'Description could not be found: {user["description"]}')
