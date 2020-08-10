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
class Comparative:
    def __init__(self, operator):
        self.operator = operator

    def __call__(self, user, system, prev_user, prev_system):
        # Clarification questions, will be handled from clarification class
        if 'description' not in user:
            return user, system

        # Comparative Reasoning (All)
        if user['description'] == 'Comparative|More/Less|Single entity type':
            return self.comparative_single_entity(user, system)

        if user['description'] == 'Comparative|More/Less|Mult. entity type':
            return self.comparative_multi_entities(user, system)

        if user['description'] == 'Comparative|More/Less|Single entity type|Indirect':
            return self.comparative_single_entity(user, system, prev_ent=prev_system['entities_in_utterance'][0])

        if user['description'] == 'Comparative|More/Less|Mult. entity type|Incomplete':
            return self.comparative_multi_entities(user, system, prev_q=prev_user['utterance'])

        if user['description'] == 'Comparative|More/Less|Mult. entity type|Indirect':
            return self.comparative_multi_entities(user, system, prev_ent=prev_system['entities_in_utterance'][0])

        if user['description'] == 'Comparative|More/Less|Single entity type|Incomplete':
            return self.comparative_single_entity(user, system, prev_q=prev_user['utterance'])

        # Comparative Reasoning (Count) (All)
        if user['description'] == 'Comparative|Count over More/Less|Single entity type|Incomplete':
            return self.comparative_single_entity(user, system, prev_q=prev_user['utterance'], is_count=True)

        if user['description'] == 'Comparative|Count over More/Less|Mult. entity type':
            return self.comparative_multi_entities(user, system, is_count=True)

        if user['description'] == 'Comparative|Count over More/Less|Single entity type|Indirect':
            return self.comparative_single_entity(user, system, prev_ent=prev_system['entities_in_utterance'][0], is_count=True)

        if user['description'] == 'Comparative|Count over More/Less|Mult. entity type|Indirect':
            return self.comparative_multi_entities(user, system, prev_ent=prev_system['entities_in_utterance'][0], is_count=True)

        if user['description'] == 'Comparative|Count over More/Less|Single entity type':
            return self.comparative_single_entity(user, system, is_count=True)

        if user['description'] == 'Comparative|Count over More/Less|Mult. entity type|Incomplete':
            return self.comparative_multi_entities(user, system, prev_q=prev_user['utterance'], is_count=True)

        raise Exception(f'Description could not be found: {user["description"]}')

    def comparative_single_entity(self, user, system, prev_q=None, prev_ent=None, is_count=False):
        # parse input
        data = self.parse_comparative_single_entity(user, system, prev_q, prev_ent)

        # get results
        data = self.get_comparative_single_entity_results(data, user['utterance'] if not prev_q else prev_q)

        # for some comparative single entity we do not have results at all, we need to exclude them
        if not data:
            system['is_spurious'] = True
            return user, system

        # extract values
        count_operator = data['count_operator']
        comperative_operator = data['comperative_operator']
        filter_operator = data['filter_operator']
        tuple_operator = data['tuple_operator']
        find_operator = data['find_operator']
        entity = data['entity']
        relation = data['relation']
        ent_type = data['ent_type']
        type_1 = data['type_1']
        type_2 = data['type_2']
        gold = data['gold']
        type_dict = data['type_dict']
        result = data['result']

        assert gold == result or gold.issubset(result) or gold.issubset(set(type_dict.keys()))

        system['gold_actions'] = [
            ['action', comperative_operator.__name__],
            ['action', tuple_operator.__name__],
            ['relation', relation],
            ['type', type_1],
            ['type', type_2],
            ['action', count_operator.__name__],
            ['action', filter_operator.__name__],
            ['action', find_operator.__name__],
            ['entity', entity],
            ['relation', relation],
            ['type', ent_type]
        ]

        if is_count:
            system['gold_actions'].insert(0, ['action', count_operator.__name__])

        system['is_spurious'] = False if gold == result or gold.issubset(result) else True

        return user, system

    def comparative_multi_entities(self, user, system, prev_q=None, prev_ent=None, is_count=False):
        # parse input
        data = self.parse_comparative_multi_entities(user, system, prev_q, prev_ent)

        # extract values
        count_operator = data['count_operator']
        comperative_operator = data['comperative_operator']
        filter_operator = data['filter_operator']
        logical_operator = data['logical_operator']
        tuple_operator = data['tuple_operator']
        find_operator = data['find_operator']
        entity = data['entity']
        relation = data['relation']
        base_type = data['base_type']
        type_1 = data['type_1']
        type_2 = data['type_2']
        gold = data['gold']

        # get results
        ent = find_operator(entity, relation)
        filter_ent = filter_operator(ent, type_1, type_2)
        ent_count = count_operator(filter_ent)
        type_dict_1 = tuple_operator(relation, base_type, type_1)
        type_dict_2 = tuple_operator(relation, base_type, type_2)
        type_dict = logical_operator(type_dict_1, type_dict_2)
        result = comperative_operator(type_dict, ent_count)

        assert gold == result or gold.issubset(result) or gold.issubset(set(type_dict.keys()))

        system['gold_actions'] = [
            ['action', comperative_operator.__name__],
            ['action', logical_operator.__name__],
            ['action', tuple_operator.__name__],
            ['relation', relation],
            ['type', base_type],
            ['type', type_1],
            ['action', tuple_operator.__name__],
            ['relation', relation],
            ['type', base_type],
            ['type', type_2],
            ['action', count_operator.__name__],
            ['action', filter_operator.__name__],
            ['action', find_operator.__name__],
            ['entity', entity],
            ['relation', relation],
            ['type', type_1],
            ['type', type_2]
        ]

        if is_count:
            system['gold_actions'].insert(0, ['action', count_operator.__name__])

        system['is_spurious'] = False if gold == result or gold.issubset(result) else True

        return user, system

    def parse_comparative_single_entity(self, user, system, prev_q, prev_ent):
        assert len(user['relations']) == 1
        assert len(user['type_list']) == 2
        assert len(user['entities_in_utterance']) == 1 or prev_ent != None

        active_set = system['active_set'][0][1:-1].split(',')

        entity = user['entities_in_utterance'][0] if user['entities_in_utterance'] else prev_ent
        relation = user['relations'][0]
        type_1 = user['type_list'][0]
        type_2 = user['type_list'][1]

        return {
            'count_operator': self.operator.count,
            'comperative_operator': self.get_comperative_operator(user, prev_q),
            'filter_operator': self.operator.filter_type,
            'tuple_operator': [self.operator.find_tuple_counts, self.operator.find_reverse_tuple_counts],
            'find_operator': [self.operator.find, self.operator.find_reverse],
            'entity': entity,
            'relation': relation,
            'type_1': type_1,
            'type_2': type_2,
            'gold': set(system['all_entities'])
        }

    def parse_comparative_multi_entities(self, user, system, prev_q, prev_ent):
        assert len(user['relations']) == 1
        assert len(user['type_list']) == 3
        assert len(user['entities_in_utterance']) == 1 or prev_ent != None

        active_set = system['active_set'][0][1:-1].split(',')

        entity = user['entities_in_utterance'][0] if user['entities_in_utterance'] else prev_ent
        relation = active_set[1]

        if '|' in active_set[0]:
            find_operator = self.operator.find_reverse
            tuple_operator = self.operator.find_reverse_tuple_counts
            base_type = active_set[-1][2:-1]
            sep_idx = active_set[0].index('|')
            type_1 = active_set[0][2:sep_idx-1]
            type_2 = active_set[0][sep_idx+3:-1]
        elif '|' in active_set[-1]:
            find_operator = self.operator.find
            tuple_operator = self.operator.find_tuple_counts
            base_type = active_set[0][2:-1]
            sep_idx = active_set[-1].index('|')
            type_1 = active_set[-1][2:sep_idx-1]
            type_2 = active_set[-1][sep_idx+3:-1]
        else:
            raise Exception(f'Wrong active set {user}')

        assert set([base_type, type_1, type_2]) == set(user['type_list']), f'Active set and type list are inconsistent: {user, system}'
        assert relation in user['relations'], f'Active set and relation list are inconsistent: {user, system}'

        return {
            'count_operator': self.operator.count,
            'comperative_operator': self.get_comperative_operator(user, prev_q),
            'filter_operator': self.operator.filter_multi_types,
            'tuple_operator': tuple_operator,
            'logical_operator': self.operator.union,
            'find_operator': find_operator,
            'entity': entity,
            'relation': relation,
            'base_type': base_type,
            'type_1': type_1,
            'type_2': type_2,
            'gold': set(system['all_entities'])
        }

    def get_comparative_single_entity_results(self, data, user_utterance):
        # extract values
        count_operator = data['count_operator']
        comperative_operator = data['comperative_operator']
        filter_operator = data['filter_operator']
        tuple_operator = data['tuple_operator']
        find_operator = data['find_operator']
        entity = data['entity']
        relation = data['relation']
        type_1 = data['type_1']
        type_2 = data['type_2']
        gold = data['gold']

        # find correct tuple_dict
        for to in tuple_operator:
            type_dict = to(relation, type_1, type_2)
            if gold.issubset(type_dict):
                tuple_operator = to
                break
            type_dict = to(relation, type_2, type_1)
            if gold.issubset(type_dict):
                type_1, type_2 = type_2, type_1
                tuple_operator = to
                break
            type_dict = None

        if type_dict == None:
            raise Exception(f'Could not find correct type_dict: {user_utterance}')

        symmetric_difference = []
        for find in find_operator:
            for typ in [type_1, type_2]:
                ent = find(entity, relation)
                filter_ent = filter_operator(ent, typ)
                count = count_operator(filter_ent)
                if 'approximately' in user_utterance.split() or 'around' in user_utterance.split():
                    result = set(type_dict.keys())
                else:
                    result = comperative_operator(type_dict, count)
                # if len(result) == 0:
                #     continue
                new_data = {
                    'tuple_operator': tuple_operator,
                    'find_operator': find,
                    'ent_type': typ,
                    'type_1': type_1,
                    'type_2': type_2,
                    'type_dict': type_dict,
                    'result': result
                }
                if gold == result or gold.issubset(result):
                    return dict(data, **new_data)
                elif len(result) > 0 or len(ent) > 0:
                # else:
                    symmetric_difference.append([new_data, len(gold.symmetric_difference(result))])

        # If none of the above worked then there is a problem with the question.
        # In this case we return the result with the lowest symmetric_difference from our gold.
        # A problematic question is: Which administrative territories have diplomatic relationships with less number of administrative territories than Honduras ?
        # Where on gold we have 25 entities and all of them have 0 count but one has 15 (on our type_dict).
        # Our comperative operator is (<) and the filtered entity is count 1, wich means we can find 24 gold entities except the one with count 15!!!
        # Therefore questions like this cannot be queried even though we have the correct query/logical form/action.
        if len(symmetric_difference) > 0:
            return dict(data, **sorted(symmetric_difference, key=lambda x: x[1])[0][0])
        else:
            return dict()

    def get_comperative_operator(self, user, prev_q):
        question = prev_q if prev_q != None else user['utterance']
        if 'more' in question.split():
            operator = self.operator.greater
        elif 'greater' in question.split():
            operator = self.operator.greater
        elif 'less' in question.split():
            operator = self.operator.less
        elif 'lesser' in question.split():
            operator = self.operator.less
        elif 'equal' in question.split():
            operator = self.operator.equal
        elif 'exactly' in question.split():
            operator = self.operator.equal
        elif 'approximately' in question.split():
            operator = self.operator.approx
        elif 'around' in question.split():
            operator = self.operator.approx
        else:
            raise Exception(f'Unkown quantitative operator for question: {user["utterance"]}')

        return operator
