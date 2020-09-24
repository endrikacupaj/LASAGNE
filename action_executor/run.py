import os
import time
import json
import argparse
from glob import glob
from pathlib import Path
from knowledge_graph.knowledge_graph import KnowledgeGraph
from executor import ActionExecutor
from meters import AccuracyMeter, F1scoreMeter
ROOT_PATH = Path(os.path.dirname(__file__)).parent

# add arguments to parser
parser = argparse.ArgumentParser(description='Execute actions')
parser.add_argument('--file_path', default='/data/final/csqa/process/test.json', help='json file with actions')
parser.add_argument('--question_type', default='Clarification', help='json file with actions')
parser.add_argument('--max_rresults', default=1000, help='json file with actions')
args = parser.parse_args()

# load kg
kg = KnowledgeGraph()

# load data
data_path = f'{str(ROOT_PATH)}{args.file_path}'
data = []
with open(data_path) as json_file:
    data = json.load(json_file)

# load action executor
action_executor = ActionExecutor(kg)

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

count_no_answer = 0
count_total = 0
tic = time.perf_counter()
for i, d in enumerate(data):
    if d['question_type'] != args.question_type:
        continue

    count_total += 1
    try:
        if d['actions'] is not None:
            all_actions = [action[1] for action in d['actions']]
            if 'entity' not in all_actions:
                result = action_executor(d['actions'], d['prev_results'], d['question_type'])
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
                if result != set(d['results']) and len(result) > args.max_results:
                    new_result = result.intersection(set(d['results']))
                    for res in result:
                        if res not in result: new_result.add(res)
                        if len(new_result) == args.max_results: break
                    result = new_result.copy()
                gold = set(d['results'])
                question_types_meters[d['question_type']].update(gold, result)
    except Exception as ex:
        print(d['question'])
        print(d['actions'])
        raise ValueError(ex)

    toc = time.perf_counter()
    print(f'==> Finished {((i+1)/len(data))*100:.2f}% -- {toc - tic:0.2f}s')

# print results
print(args.question_type)
print(f'NA actions: {count_no_answer}')
print(f'Total samples: {count_total}')
if args.question_type in ['Verification (Boolean) (All)', 'Quantitative Reasoning (Count) (All)', 'Comparative Reasoning (Count) (All)']:
    print(f'Accuracy: {question_types_meters[args.question_type].accuracy}')
else:
    print(f'Precision: {question_types_meters[args.question_type].precision}')
    print(f'Recall: {question_types_meters[args.question_type].recall}')
    print(f'F1-score: {question_types_meters[args.question_type].f1_score}')