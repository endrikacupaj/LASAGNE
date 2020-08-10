"""
Verification (Boolean) (All):
- Verification|3 entities, 2 direct, 2(direct) are query entities, subject is indirect - Done
- Verification|3 entities, all direct, 2 are query entities - Done
- Verification|2 entities, one direct and one indirect, object is indirect - Done
- Verification|2 entities, one direct and one indirect, subject is indirect - Done
- Verification|2 entities, both direct - Done
- Verification|one entity, multiple entities (as object) referred indirectly - Done
"""
class Verification:
    def __init__(self, operator):
        self.operator = operator

    def __call__(self, user, system, prev_system):
        # Verification (Boolean) (All)
        if user['description'] == 'Verification|3 entities, 2 direct, 2(direct) are query entities, subject is indirect':
            return self.verification_multi_entities(user, system)

        if user['description'] == 'Verification|3 entities, all direct, 2 are query entities':
            return self.verification_multi_entities(user, system)

        if user['description'] == 'Verification|2 entities, one direct and one indirect, object is indirect':
            return self.verification_direct(user, system)

        if user['description'] == 'Verification|2 entities, one direct and one indirect, subject is indirect':
            return self.verification_direct(user, system)

        if user['description'] == 'Verification|2 entities, both direct':
            return self.verification_direct(user, system)

        if user['description'] == 'Verification|one entity, multiple entities (as object) referred indirectly':
            return self.verification_indirect(user, system, prev_ent=prev_system['entities_in_utterance'])

        raise Exception(f'Description could not be found: {user["description"]}')

    def verification_direct(self, user, system):
        # parse input
        data = self.parse_verification_direct(user, system)

        # extract values
        verification_operator = data['verification_operator']
        find_operator = data['find_operator']
        entities = data['entities']
        relation = data['relation']
        gold = data['gold']

        # get results
        for find in find_operator:
            for entities_ in [entities, list(reversed(entities))]:
                set_ent = find(entities_[0], relation)
                result = verification_operator([entities_[1]], set_ent)
                if gold == result:
                    find_operator = find
                    entities = [entities_[0], entities_[1]]
                    break
            if gold == result:
                break

        assert gold == result

        system['gold_actions'] = [
            ['action', verification_operator.__name__],
            ['entity', entities[1]],
            ['action', find_operator.__name__],
            ['entity', entities[0]],
            ['relation', relation]
        ]

        system['is_spurious'] = False if gold == result else True

        return user, system

    def verification_multi_entities(self, user, system):
        # parse input
        data = self.parse_verification_multi_entities(user, system)

        # extract values
        verification_operator = data['verification_operator']
        logical_operator = data['logical_operator']
        find_operator = data['find_operator']
        base_entity = data['base_entity']
        relation = data['relation']
        entities = data['entities']
        gold = data['gold']

        # get results
        ent_1 = find_operator(entities[0], relation)
        ent_2 = find_operator(entities[1], relation)
        intersection = logical_operator(ent_1, ent_2)
        result = verification_operator([base_entity], intersection)

        assert gold == result

        system['gold_actions'] = [
            ['action', verification_operator.__name__],
            ['entity', base_entity],
            ['action', logical_operator.__name__],
            ['action', find_operator.__name__],
            ['entity', entities[0]],
            ['relation', relation],
            ['action', find_operator.__name__],
            ['entity', entities[1]],
            ['relation', relation]
        ]

        system['is_spurious'] = False if gold == result else True

        return user, system

    def verification_indirect(self, user, system, prev_ent):
        # parse input
        data = self.parse_verification_indirect(user, system, prev_ent)

        # extract values
        verification_operator = data['verification_operator']
        find_operator = data['find_operator']
        base_entity = data['base_entity']
        relation = data['relation']
        entities = data['entities']
        gold = data['gold']

        # get results
        for find in find_operator:
            set_ent = find(base_entity, relation)
            result = verification_operator(entities, set_ent)
            if gold == result:
                find_operator = find
                break

        assert gold == result

        system['gold_actions'] = [
            ['action', verification_operator.__name__],
            ['entity', 'prev_answer'],
            ['action', find_operator.__name__],
            ['entity', base_entity],
            ['relation', relation]
        ]

        system['is_spurious'] = False if gold == result else True

        return user, system

    def parse_verification_direct(self, user, system):
        assert len(user['entities_in_utterance']) == 2
        assert len(user['relations']) == 1

        return {
            'verification_operator': self.operator.is_in,
            'find_operator': [self.operator.find, self.operator.find_reverse],
            'entities': user['entities_in_utterance'],
            'relation': user['relations'][0],
            'gold': True if system['utterance'] == 'YES' else False
        }

    def parse_verification_multi_entities(self, user, system):
        assert len(system['active_set']) == 2

        active_set_1 = system['active_set'][0][1:-1].split(',')
        active_set_2 = system['active_set'][1][1:-1].split(',')

        if active_set_1[0] == active_set_2[0]:
            find_operator = self.operator.find_reverse
            base_entity = active_set_1[0]
            relation = active_set_1[1]
            entities = [active_set_1[-1], active_set_2[-1]]
        elif active_set_1[-1] == active_set_2[-1]:
            find_operator = self.operator.find
            base_entity = active_set_1[-1]
            relation = active_set_1[1]
            entities = [active_set_1[0], active_set_2[0]]
        else:
            raise Exception(f'Wrong active set {user}')

        return {
            'verification_operator': self.operator.is_in,
            'logical_operator': self.operator.intersection,
            'find_operator': find_operator,
            'base_entity': base_entity,
            'relation': relation,
            'entities': entities,
            'gold': True if system['utterance'] == 'YES' else False
        }

    def parse_verification_indirect(self, user, system, prev_ent):
        assert len(user['entities_in_utterance']) == 1
        assert len(user['relations']) == 1

        return {
            'verification_operator': self.operator.is_in,
            'find_operator': [self.operator.find, self.operator.find_reverse],
            'base_entity': user['entities_in_utterance'][0],
            'relation': user['relations'][0],
            'entities': prev_ent,
            'gold': True if system['utterance'] == 'YES' else False
        }
