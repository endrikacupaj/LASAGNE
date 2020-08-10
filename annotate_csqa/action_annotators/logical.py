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
class Logical:
    def __init__(self, operator):
        self.operator = operator

    def __call__(self, user, system):
        # Logical Reasoning (All)
        if 'description' not in user:
            raise Exception(f'No description for question:\n{user}')

        if user['description'] == 'Logical|Difference|Multiple_Relation':
            return self.logical(user, system)

        if user['description'] == 'Logical|Union|Single_Relation':
            return self.logical(user, system)

        if user['description'] == 'Logical|Union|Multiple_Relation':
            return self.logical(user, system)

        if user['description'] == 'Logical|Intersection|Single_Relation|Incomplete':
            return self.logical(user, system)

        if user['description'] == 'Logical|Difference|Single_Relation|Incomplete':
            return self.logical(user, system)

        if user['description'] == 'Logical|Difference|Single_Relation':
            return self.logical(user, system)

        if user['description'] == 'Logical|Intersection|Single_Relation':
            return self.logical(user, system)

        if user['description'] == 'Logical|Intersection|Multiple_Relation':
            return self.logical(user, system)

        if user['description'] == 'Logical|Union|Single_Relation|Incomplete':
            return self.logical(user, system)

        raise Exception(f'Description could not be found: {user["description"]}')

    def logical(self, user, system):
        # parse input
        data = self.parse_logical(user, system)

        # extract values
        logical_operator = data['logical_operator']
        filter_operator = data['filter_operator']
        first_set = data['first_set']
        second_set = data['second_set']
        typ = data['type']
        gold = data['gold']

        # get results
        set_1 = first_set['find_operator'](first_set['entity'], first_set['relation'])
        set_2 = second_set['find_operator'](second_set['entity'], second_set['relation'])
        set_result = logical_operator(set_1, set_2)
        result = filter_operator(set_result, typ)

        assert gold == result

        system['gold_actions'] = [
            ['action', filter_operator.__name__],
            ['action', logical_operator.__name__],
            ['action', first_set['find_operator'].__name__],
            ['entity', first_set['entity']],
            ['relation', first_set['relation']],
            ['action', second_set['find_operator'].__name__],
            ['entity', second_set['entity']],
            ['relation', second_set['relation']],
            ['type', typ],
        ]

        system['is_spurious'] = False if gold == result else True

        return user, system

    def parse_logical(self, user, system):
        assert len(system['active_set']) == 1

        if system['active_set'][0].startswith('AND') and 'NOT' in system['active_set'][0]: # Difference
            logical_operator = self.operator.difference
            active_set = system['active_set'][0].split(', NOT')
            first_set = self.parse_active_set(active_set[0].replace('AND(', ''))
            second_set = self.parse_active_set(active_set[1][1:-2])
        elif system['active_set'][0].startswith('OR'): # Union
            logical_operator = self.operator.union
            active_set = system['active_set'][0].split(', ')
            first_set = self.parse_active_set(active_set[0].replace('OR(', ''))
            second_set = self.parse_active_set(active_set[1][:-1])
        elif system['active_set'][0].startswith('AND'): # Intersection
            logical_operator = self.operator.intersection
            active_set = system['active_set'][0].split(', ')
            first_set = self.parse_active_set(active_set[0].replace('AND(', ''))
            second_set = self.parse_active_set(active_set[1][:-1])
        else:
            raise Exception(f'Wrong active set {user}')

        assert first_set['type'] == second_set['type']

        return {
            'logical_operator': logical_operator,
            'filter_operator': self.operator.filter_type,
            'first_set': first_set,
            'second_set': second_set,
            'type': first_set['type'],
            'gold': set(system['all_entities'])
        }

    def parse_active_set(self, active_set):
        active_set_splited = active_set[1:-1].split(',')
        if active_set_splited[0].startswith('c'):
            find_operator = self.operator.find_reverse
            entity = active_set_splited[2]
            relation = active_set_splited[1]
            typ = active_set_splited[0][2:-1]
        elif active_set_splited[0].startswith('Q'):
            find_operator = self.operator.find
            entity = active_set_splited[0]
            relation = active_set_splited[1]
            typ = active_set_splited[2][2:-1]
        else:
            raise Exception(f"Can not parse active set: {active_set}")

        return {
            'find_operator': find_operator,
            'entity': entity,
            'relation': relation,
            'type': typ
        }
