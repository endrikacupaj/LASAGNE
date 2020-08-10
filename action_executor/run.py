# %%
import os
import time
import json
import argparse
from glob import glob
from pathlib import Path
from knowledge_graph.knowledge_graph import KnowledgeGraph
from executor import ActionExecutor
ROOT_PATH = Path(os.path.dirname(__file__)).parent
# %%
# add arguments to parser
# parser = argparse.ArgumentParser(description='Execute actions')
# parser.add_argument('--file_path', default='/data/final/csqa/process/test.json', help='json file to execute actions')
# args = parser.parse_args()
# %%
# load kg
kg = KnowledgeGraph()
# %%
# load data
file_path = '/transformer_gnn/experiments/inference/old/test_Clarification.json'
# file_path = '/data/final/csqa/processed/test.json'
data_path = f'{str(ROOT_PATH)}{file_path}'
data = []
with open(data_path) as json_file:
    data = json.load(json_file)
print(len(data))
# %%
from importlib import reload
import executor as executor
import meters as meters
import actions as actions
reload(executor)
reload(meters)
reload(actions)
from executor import ActionExecutor
action_executor = ActionExecutor(kg)
from meters import AccuracyMeter, F1scoreMeter

# define question type meters
question_types_meters = {
    'Clarification': F1scoreMeter(),
    'Comparative Reasoning (All)': F1scoreMeter(),
    'Logical Reasoning (All)': F1scoreMeter(),
    'Quantitative Reasoning (All)': F1scoreMeter(),
    'Simple Question (Coreferenced)': F1scoreMeter(),
    'Simple Question (Direct)': F1scoreMeter(),
    'Simple Question (Ellipsis)': F1scoreMeter(),
    # -------------------------------------------
    'Verification (Boolean) (All)': AccuracyMeter(),
    'Quantitative Reasoning (Count) (All)': AccuracyMeter(),
    'Comparative Reasoning (Count) (All)': AccuracyMeter()
}

q_type = 'Clarification'
max_results = 1000 # based on dataset answers, no mare are allowed
enable_approx_constraint = False
count_no_answer = 0
count_total = 0
tic = time.perf_counter()
for i, d in enumerate(data):
    if d['question_type'] != q_type:
        continue

    count_total += 1
    try:
        if d['actions'] is not None:
            all_actions = [action[1] for action in d['actions']]
            if 'entity' not in all_actions:
                result = action_executor(d['actions'], d['prev_results'], d['question_type'])

                # if the gold is a subset of results then we have the correct logical form
                all_actions = [action[1] for action in d['actions']]
                if enable_approx_constraint and 'approx' in all_actions and set(d['results']).issubset(result):
                    result = set(d['results'])
            else:
                count_no_answer += 1
                result = set([])
        else:
            count_no_answer += 1
            result = set([])
    except Exception as ex:
        print(d['question'])
        print(d['actions'])
        print(ex)
        count_no_answer += 1
        result = set([])

    try:
        if d['question_type'] == 'Verification (Boolean) (All)':
            answer = True if d['answer'] == 'YES' else False
            question_types_meters[d['question_type']].update(answer, result)
        else:
            if d['question_type'] in ['Quantitative Reasoning (Count) (All)', 'Comparative Reasoning (Count) (All)']:
                if d['answer'].isnumeric():
                    question_types_meters[d['question_type']].update(int(d['answer']), len(result))
                else:
                    question_types_meters[d['question_type']].update(len(d['results']), len(result))
            else:
                # limit results to max_results number
                # keep all golds and fill with random others
                if result != set(d['results']) and len(result) > max_results:
                    new_result = result.intersection(set(d['results']))
                    for res in result:
                        if res not in result: new_result.add(res)
                        if len(new_result) == max_results: break
                    result = new_result.copy()
                gold = set(d['results']) # set(list(d['results'])[:1]) if result != set(d['results']) and len(result) == 0 else set(d['results'])
                question_types_meters[d['question_type']].update(gold, result)
    except Exception as ex:
        print(d['question'])
        print(d['actions'])
        raise ValueError(ex)
    toc = time.perf_counter()
    print(f'==> Finished {((i+1)/len(data))*100:.2f}% -- {toc - tic:0.2f}s')

print(q_type)
print(f'NA actions: {count_no_answer}')
print(f'Total samples: {count_total}')
print(f'Precision: {question_types_meters[q_type].precision}')
print(f'Recall: {question_types_meters[q_type].recall}')
print(f'F1-score: {question_types_meters[q_type].f1_score}')
# print(f'Accuracy: {question_types_meters[q_type].accuracy}')

# %%
from importlib import reload
import executor as executor
reload(executor)
from executor import ActionExecutor
action_executor = ActionExecutor(kg)
action_example = [['action', 'count'], ['action', 'greater'], ['action', 'union'], ['action', 'find_reverse_tuple_counts'], ['relation', 'P921'], ['type', 'Q12737077'], ['type', 'Q838948'], ['action', 'find_reverse_tuple_counts'], ['relation', 'P921'], ['type', 'Q12737077'], ['type', 'Q2342494'], ['action', 'count'], ['action', 'filter_multi_types'], ['action', 'find_reverse'], ['entity', 'Q1856798'], ['relation', 'P921'], ['type', 'Q838948'], ['type', 'Q2342494']]
print(action_executor(action_example, [], 'Comparative Reasoning (Count) (All)'))
# %%
