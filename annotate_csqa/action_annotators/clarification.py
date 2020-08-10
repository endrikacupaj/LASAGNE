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
class Clarification:
    def __init__(self, simple, quantitative, comparative):
        self.simple = simple
        self.quantitative = quantitative
        self.comparative = comparative

    def __call__(self, conv_chunk):
        # Clarification
        if 'description' not in conv_chunk['user']:
            raise Exception(f'No description for question:\n{conv_chunk["user"]}')

        if conv_chunk['user']['description'] != 'Simple Question|Single Entity|Indirect':
            return conv_chunk['user'], conv_chunk['next_system']

        if conv_chunk['user']['description'] == 'Simple Question|Single Entity|Indirect':
            return self.simple.simple_question_single_entity(conv_chunk['user'], conv_chunk['next_system'])

        if conv_chunk['user']['description'] == 'Comparative|More/Less|Single entity type|Indirect':
            prev_ent = conv_chunk['system']['entities_in_utterance'] if conv_chunk['next_user']['utterance'] == 'Yes' else conv_chunk['next_user']['entities']
            return self.comparative.comparative_single_entity(conv_chunk['user'], conv_chunk['next_system'], prev_ent=prev_ent[0])

        if conv_chunk['user']['description'] == 'Quantitative|Count|Single entity type|Indirect':
            return self.quantitative.quantitative_simple_single_type(conv_chunk['user'], conv_chunk['next_system'])

        if conv_chunk['user']['description'] == 'Comparative|Count over More/Less|Single entity type|Indirect':
            prev_ent = conv_chunk['system']['entities_in_utterance'] if conv_chunk['next_user']['utterance'] == 'Yes' else conv_chunk['next_user']['entities']
            return self.comparative.comparative_single_entity(conv_chunk['user'], conv_chunk['next_system'], prev_ent=prev_ent[0], is_count=True)

        if conv_chunk['user']['description'] == 'Quantitative|Count|Logical operators|Indirect':
            return self.quantitative.quantitative_simple_multi_type(conv_chunk['user'], conv_chunk['next_system'])

        if conv_chunk['user']['description'] == 'Comparative|Count over More/Less|Mult. entity type|Indirect':
            return self.comparative.comparative_multi_entities(conv_chunk['user'], conv_chunk['next_system'], prev_ent=conv_chunk['prev_system']['entities_in_utterance'][0], is_count=True)

        if conv_chunk['user']['description'] == 'Comparative|More/Less|Mult. entity type|Indirect':
            return self.comparative.comparative_multi_entities(conv_chunk['user'], conv_chunk['next_system'], prev_ent=conv_chunk['prev_system']['entities_in_utterance'][0])

        raise Exception(f'Description could not be found: {conv_chunk["user"]["description"]}')
