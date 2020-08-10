from collections import OrderedDict
from actions import ActionOperator
action_input_size_constrain = {'find': 2, 'find_reverse': 2, 'filter_type': 2,
                            'filter_multi_types': 3, 'find_tuple_counts': 3, 'find_reverse_tuple_counts': 3,
                            'greater': 2, 'less': 2, 'equal': 2, 'approx': 2, 'atmost': 2, 'atleast': 2, 'argmin': 1,
                            'argmax': 1, 'is_in': 2, 'count': 1, 'union': 10, 'intersection': 2, 'difference': 2}

class ActionExecutor:
    def __init__(self, kg):
        self.operator = ActionOperator(kg)

    def _parse_actions(self, actions):
        actions_to_execute = OrderedDict()
        for i, action in enumerate(actions):
            if i == 0:
                actions_to_execute[str(i)] = [action[1]]
                continue

            if action[0] == 'action':
                actions_to_execute[str(i)] = [action[1]]
                if actions[i-1][0] == 'action':
                    # child of previous action
                    actions_to_execute[str(i-1)].append(str(i))
                else:
                    for k in reversed(list(actions_to_execute.keys())[:-1]):
                        if (len(actions_to_execute[k])-1) < action_input_size_constrain[actions_to_execute[k][0]]:
                            actions_to_execute[str(k)].append(str(i))
                            break
            else:
                for j in reversed(list(actions_to_execute.keys())):
                    j = int(j)
                    if actions[j][0] == 'action' and len(actions_to_execute[str(j)]) < action_input_size_constrain[actions[j][1]] + 1:
                        # child of previous action
                        if action[1].isnumeric():
                            actions_to_execute[str(j)].append(int(action[1]))
                        else:
                            actions_to_execute[str(j)].append(action[1])
                        break
                    elif actions[j][0] != 'action' and len(actions_to_execute[str(j)]) < action_input_size_constrain[actions[j][1]] + 1:
                        actions_to_execute[str(j)].append(action[1])

        return actions_to_execute

    def _execute_actions(self, actions_to_execute):
        # execute actions on kg
        partial_results = OrderedDict()
        for key, value in reversed(actions_to_execute.items()):
            if key == list(actions_to_execute.keys())[0] and value[0] == 'count':
                continue
            # create new values in case getting children results
            new_value = []
            for v in value:
                if isinstance(v, str) and v.isnumeric():
                    new_value.append(partial_results[v])
                    continue
                new_value.append(v)

            value = new_value.copy()

            # execute action
            action = value[0]
            if action == 'union' and len(value) >= 2:
                partial_results[key] = getattr(self.operator, action)(*value[1:])
            elif len(value) == 2:
                arg = value[1]
                partial_results[key] = getattr(self.operator, action)(arg)
            elif len(value) == 3:
                arg_1 = value[1]
                arg_2 = value[2]
                partial_results[key] = getattr(self.operator, action)(arg_1, arg_2)
            elif len(value) == 4:
                arg_1 = value[1]
                arg_2 = value[2]
                arg_3 = value[3]
                partial_results[key] = getattr(self.operator, action)(arg_1, arg_2, arg_3)
            else:
                raise NotImplementedError('Not implemented for more than 3 inputs!')

        return next(reversed(partial_results.values()))


    def __call__(self, actions, prev_results, question_type):
        if question_type in ['Logical Reasoning (All)', 'Quantitative Reasoning (All)', 'Comparative Reasoning (All)', 'Clarification', 'Quantitative Reasoning (Count) (All)', 'Comparative Reasoning (Count) (All)']:
            action_input_size_constrain['union'] = 2
        # parse actions
        actions_to_execute = self._parse_actions(actions)
        for key, value in actions_to_execute.items():
            if actions_to_execute[key][1] == 'prev_answer':
                actions_to_execute[key][1] = prev_results
            elif actions_to_execute[key][0] == 'is_in' and actions_to_execute[key][1].startswith('Q'):
                actions_to_execute[key][1] = [actions_to_execute[key][1]]
        # execute actions and return results
        return self._execute_actions(actions_to_execute)
