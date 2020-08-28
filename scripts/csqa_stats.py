# %%
import os
import sys
import json
from glob import glob
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__)).parent.parent
# %%
train, val, test = [], [], []
# read data
# train_files = glob(f'{ROOT_PATH}/data/final/csqa/train/*' + '/*.json')
# for f in train_files:
#     with open(f) as json_file:
#         train.append(json.load(json_file))

val_files = glob(f'{ROOT_PATH}/data/final/csqa/val/*' + '/*.json')
for f in val_files:
    with open(f) as json_file:
        val.append(json.load(json_file))

# test_files = glob(f'{ROOT_PATH}/data/final/csqa/test/*' + '/*.json')
# for f in test_files:
#     with open(f) as json_file:
#         test.append(json.load(json_file))

# %%
ner_spurious = {}
total_examples = {}
for data in val: # train + val + test:
    for d in data:
        if d['speaker'] == 'USER' and 'question-type' in d and d['question-type'] in ['Logical Reasoning (All)', 'Verification (Boolean) (All)']:
            if d['question-type'] not in total_examples:
                total_examples[d['question-type']] = {}
                ner_spurious[d['question-type']] = {}

            if d['description'] not in total_examples[d['question-type']]:
                total_examples[d['question-type']][d['description']] = 0
                ner_spurious[d['question-type']][d['description']] = 0

            total_examples[d['question-type']][d['description']] += 1

            if 'is_ner_spurious' not in d or d['is_ner_spurious']:
                ner_spurious[d['question-type']][d['description']] += 1

for qt in total_examples.keys():
    print(f'Question Type: {qt}')
    for desc in total_examples[qt].keys():
        print(f'Description: {desc}')
        print(f'Spurious: {ner_spurious[qt][desc]}')
        print(f'Total: {total_examples[qt][desc]}')
        print(f'Percentage of spurious: {(ner_spurious[qt][desc] / total_examples[qt][desc]):.4f}')


# %%
count = 0
for data in test: # train + val + test:
    for d in data:
        if d['speaker'] == 'USER' and 'question-type' in d and d['question-type'] in ['Comparative Reasoning (Count) (All)', 'Quantitative Reasoning (Count) (All)'] and 'description' not in d:
            count += 1
print(count)
# %%
answer_len = []
for data in val: # train + val + test:
    for d in data:
        if d['speaker'] == 'SYSTEM' and 'all_entities' in d:
            answer_len.append(len(d['all_entities']))
print(max(answer_len))
print(min(answer_len))
print(sum(answer_len) / len(answer_len))
# %%
