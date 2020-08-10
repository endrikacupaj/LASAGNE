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
import re
class Quantitative:
    def __init__(self, operator):
        self.operator = operator

    def __call__(self, user, system):
        # Clarification questions, will be handled from clarification class
        if 'description' not in user:
            return user, system

        # Quantitative Reasoning (All)
        if user['description'] == 'Quantitative|Min/Max|Single entity type':
            return self.quantitative_complex_single_type(user, system)

        if user['description'] == 'Quantitative|Min/Max|Mult. entity type':
            return self.quantitative_complex_multi_type(user, system)

        if user['description'] == 'Quantitative|Atleast/ Atmost/ Approx. the same/Equal|Mult. entity type':
            return self.quantitative_complex_multi_type(user, system)

        if user['description'] == 'Quantitative|Atleast/ Atmost/ Approx. the same/Equal|Single entity type':
            return self.quantitative_complex_single_type(user, system)

        # Quantitative Reasoning (Count) (All)
        if user['description'] == 'Quantitative|Count|Logical operators':
            return self.quantitative_simple_multi_type(user, system)

        if user['description'] == 'Quantitative|Count|Mult. entity type':
            return self.quantitative_simple_multi_type(user, system)

        if user['description'] == 'Quantitative|Count|Single entity type':
            return self.quantitative_simple_single_type(user, system)

        if user['description'] == 'Quantitative|Count|Single entity type|Indirect':
            return self.quantitative_simple_single_type(user, system)

        if user['description' ] == 'Quantitative|Count|Logical operators|Indirect':
            return self.quantitative_simple_multi_type(user, system)

        if user['description'] == 'Quantitative|Count|Single entity type|Incomplete':
            return self.quantitative_simple_single_type(user, system)

        if user['description'] == 'Quantitative|Count over Atleast/ Atmost/ Approx. the same/Equal|Mult. entity type':
            return self.quantitative_complex_multi_type(user, system, True)

        if user['description'] == 'Quantitative|Count over Atleast/ Atmost/ Approx. the same/Equal|Single entity type':
            return self.quantitative_complex_single_type(user, system, True)

        if user['description'] == 'Incomplete count-based ques':
            return self.quantitative_simple_single_type(user, system)

        raise Exception(f'Description could not be found: {user["description"]}')

    def quantitative_simple_single_type(self, user, system):
        # parse input
        data = self.parse_quantitative_simple_single_type(user, system)

        # extract values
        count_operator = data['count_operator']
        filter_operator = data['filter_operator']
        find_operator = data['find_operator']
        entity = data['entity']
        relation = data['relation']
        typ = data['type']
        gold = data['gold']

        # get results
        ent = find_operator(entity, relation)
        result = filter_operator(ent, typ)

        assert gold == result or system['utterance'] == str(len(result))

        system['gold_actions'] = [
            ['action', count_operator.__name__],
            ['action', filter_operator.__name__],
            ['action', find_operator.__name__],
            ['entity', entity],
            ['relation', relation],
            ['type', typ]
        ]

        system['is_spurious'] = False if gold == result or system['utterance'] == str(len(result)) else True

        return user, system

    def quantitative_simple_multi_type(self, user, system):
        # parse input
        data = self.parse_quantitative_simple_multi_type(user, system)

        # extract values
        count_operator = data['count_operator']
        logical_operator = data['logical_operator']
        filter_operator = data['filter_operator']
        find_operator = data['find_operator']
        entities = data['entities']
        relations = data['relations']
        types = data['types']
        gold = data['gold']

        # get results
        ent_1 = find_operator(entities[0], relations[0])
        ent_2 = find_operator(entities[1], relations[1])
        filter_ent_1 = filter_operator(ent_1, types[0])
        filter_ent_2 = filter_operator(ent_2, types[1])

        for lo in logical_operator:
            result = lo(filter_ent_1, filter_ent_2)
            if gold == result or gold.issubset(result):
                logical_operator = lo
                break

        assert gold == result

        system['gold_actions'] = [
            ['action', count_operator.__name__],
            ['action', logical_operator.__name__],
            ['action', filter_operator.__name__],
            ['action', find_operator.__name__],
            ['entity', entities[0]],
            ['relation', relations[0]],
            ['type', types[0]],
            ['action', filter_operator.__name__],
            ['action', find_operator.__name__],
            ['entity', entities[1]],
            ['relation', relations[1]],
            ['type', types[1]],
        ]

        system['is_spurious'] = False if gold == result else True

        return user, system

    def quantitative_complex_single_type(self, user, system, is_count=False):
        # parse input
        data = self.parse_quantitative_complex_single_type(user, system)

        # extract values
        count_operator = data['count_operator']
        quantitative_operator = data['quantitative_operator']
        tuple_operator = data['tuple_operator']
        quantitative_value = data['quantitative_value']
        relation = data['relation']
        type_1 = data['type_1']
        type_2 = data['type_2']
        gold = data['gold']

        # get results
        for to in tuple_operator:
            type_dict = to(relation, type_1, type_2)
            if len(type_dict) > 0:
                result = quantitative_operator(type_dict, quantitative_value)
                if gold == result or gold.issubset(result) or gold.issubset(set(type_dict.keys())):
                    tuple_operator = to
                    break
            type_dict = to(relation, type_2, type_1)
            if len(type_dict) > 0:
                result = quantitative_operator(type_dict, quantitative_value)
                if gold == result or gold.issubset(result) or gold.issubset(set(type_dict.keys())):
                    tuple_operator = to
                    type_1, type_2 = type_2, type_1
                    break

        assert gold == result or gold.issubset(result) or gold.issubset(set(type_dict.keys()))

        system['gold_actions'] = [
            ['action', quantitative_operator.__name__],
            ['action', tuple_operator.__name__],
            ['relation', relation],
            ['type', type_1],
            ['type', type_2],
        ]

        if is_count:
            system['gold_actions'].insert(0, ['action', count_operator.__name__])

        if quantitative_operator.__name__ not in ['argmin', 'argmax']:
            system['gold_actions'].append(['value', str(quantitative_value)])

        system['is_spurious'] = False if gold == result or gold.issubset(result) else True

        return user, system

    def quantitative_complex_multi_type(self, user, system, is_count=False):
        # parse input
        data = self.parse_quantitative_complex_multi_type(user, system)

        # extract values
        count_operator = data['count_operator']
        quantitative_operator = data['quantitative_operator']
        logical_operator = data['logical_operator']
        tuple_operator = data['tuple_operator']
        quantitative_value = data['quantitative_value']
        relation = data['relation']
        base_type = data['base_type']
        type_1 = data['type_1']
        type_2 = data['type_2']
        gold = data['gold']

        # get results
        type_dict_1 = tuple_operator(relation, base_type, type_1)
        type_dict_2 = tuple_operator(relation, base_type, type_2)
        type_dict = logical_operator(type_dict_1, type_dict_2)
        result = quantitative_operator(type_dict, quantitative_value)

        assert gold == result or gold.issubset(result) or gold.issubset(set(type_dict.keys()))

        system['gold_actions'] = [
            ['action', quantitative_operator.__name__],
            ['action', logical_operator.__name__],
            ['action', tuple_operator.__name__],
            ['relation', relation],
            ['type', base_type],
            ['type', type_1],
            ['action', tuple_operator.__name__],
            ['relation', relation],
            ['type', base_type],
            ['type', type_2]
        ]

        if is_count:
            system['gold_actions'].insert(0, ['action', count_operator.__name__])

        if quantitative_operator.__name__ not in ['argmin', 'argmax']:
            system['gold_actions'].append(['value', str(quantitative_value)])

        system['is_spurious'] = False if gold == result or gold.issubset(result) else True

        return user, system

    def parse_quantitative_simple_single_type(self, user, system):
        active_set = system['active_set'][0][1:-1].split(',')
        if active_set[0].startswith('c'):
            find_operator = self.operator.find_reverse
            entity = active_set[2]
            relation = active_set[1]
            typ = active_set[0][2:-1]
        elif active_set[0].startswith('Q'):
            find_operator = self.operator.find
            entity = active_set[0]
            relation = active_set[1]
            typ = active_set[2][2:-1]
        else:
            raise Exception(f'Wrong active set: {user}')

        return {
            'count_operator': self.operator.count,
            'filter_operator': self.operator.filter_type,
            'find_operator': find_operator,
            'entity': entity,
            'relation': relation,
            'type': typ,
            'gold': set(system['all_entities'])
        }

    def parse_quantitative_complex_single_type(self, user, system):
        assert len(user['relations']) == 1
        assert len(user['type_list']) == 2

        operators = self.get_quantitative_operators(user)

        return {
            'count_operator': self.operator.count,
            'quantitative_operator': operators['quantitative_operator'],
            'tuple_operator': [self.operator.find_tuple_counts, self.operator.find_reverse_tuple_counts],
            'quantitative_value': operators['value'],
            'relation': user['relations'][0],
            'type_1': user['type_list'][0],
            'type_2': user['type_list'][1],
            'gold': set(system['all_entities'])
        }

    def parse_quantitative_simple_multi_type(self, user, system):
        entities = []
        relations = []
        types = []

        for active_set in system['active_set']:
            active_set = active_set[1:-1].split(',')
            if active_set[0].startswith('c'):
                find_operator = self.operator.find_reverse
                entities.append(active_set[2])
                relations.append(active_set[1])
                types.append(active_set[0][2:-1])
            elif active_set[0].startswith('Q'):
                find_operator = self.operator.find
                entities.append(active_set[0])
                relations.append(active_set[1])
                types.append(active_set[2][2:-1])
            else:
                raise Exception(f'Wrong active set: {user}')

        return {
            'count_operator': self.operator.count,
            'logical_operator': [self.operator.intersection, self.operator.union],
            'filter_operator': self.operator.filter_type,
            'find_operator': find_operator,
            'entities': entities,
            'relations': relations,
            'types': types,
            'gold': set(system['all_entities'])
        }

    def parse_quantitative_complex_multi_type(self, user, system):
        assert len(user['relations']) == 1
        assert len(user['type_list']) == 3

        active_set = system['active_set'][0][1:-1].split(',')
        if '|' in active_set[0]:
            tuple_operator = self.operator.find_reverse_tuple_counts
            relation = active_set[1]
            base_type = active_set[-1][2:-1]
            sep_idx = active_set[0].index('|')
            type_1 = active_set[0][2:sep_idx-1]
            type_2 = active_set[0][sep_idx+3:-1]
        elif '|' in active_set[-1]:
            tuple_operator = self.operator.find_tuple_counts
            relation = active_set[1]
            base_type = active_set[0][2:-1]
            sep_idx = active_set[-1].index('|')
            type_1 = active_set[-1][2:sep_idx-1]
            type_2 = active_set[-1][sep_idx+3:-1]
        else:
            raise Exception(f'Wrong active set {user}')

        assert set([base_type, type_1, type_2]) == set(user['type_list']), f'Active set and type list are inconsistent: {user, system}'
        assert relation in user['relations'], f'Active set and relation list are inconsistent: {user, system}'

        operators = self.get_quantitative_operators(user)

        return {
            'count_operator': self.operator.count,
            'quantitative_operator': operators['quantitative_operator'],
            'logical_operator': operators['logical_operator'],
            'tuple_operator': tuple_operator,
            'quantitative_value': operators['value'],
            'relation': relation,
            'base_type': base_type,
            'type_1': type_1,
            'type_2': type_2,
            'gold': set(system['all_entities'])
        }

    def get_quantitative_operators(self, user):
        if 'min' in user['utterance'].split():
            operator = self.operator.argmin
            value = '0'
        elif 'max' in user['utterance'].split():
            operator = self.operator.argmax
            value = '0'
        elif 'exactly' in user['utterance'].split():
            operator = self.operator.equal
            value = re.search(r'\d+', user['utterance'].split('exactly')[1]).group()
        elif 'approximately' in user['utterance'].split():
            operator = self.operator.approx
            value = re.search(r'\d+', user['utterance'].split('approximately')[1]).group()
        elif 'around' in user['utterance'].split():
            operator = self.operator.approx
            value = re.search(r'\d+', user['utterance'].split('around')[1]).group()
        elif 'atmost' in user['utterance'].split():
            operator = self.operator.atmost
            value = re.search(r'\d+', user['utterance'].split('atmost')[1]).group()
        elif 'atleast' in user['utterance'].split():
            operator = self.operator.atleast
            value = re.search(r'\d+', user['utterance'].split('atleast')[1]).group()
        else:
            raise Exception(f'Unkown quantitative operator for question: {user["utterance"]}')

        return {
            'quantitative_operator': operator,
            'value': int(value),
            'logical_operator': self.operator.union
        }
