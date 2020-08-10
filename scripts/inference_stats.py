# %%
import os
import sys
import json
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__)).parent
# %%
# question_types_meters = [
#     'Clarification',
#     'Comparative Reasoning (All)',
#     'Logical Reasoning (All)',
#     'Quantitative Reasoning (All)',
#     'Simple Question (Coreferenced)',
#     'Simple Question (Direct)',
#     'Simple Question (Ellipsis)',
#     # -------------------------------------------
#     'Verification (Boolean) (All)',
#     'Quantitative Reasoning (Count) (All)',
#     'Comparative Reasoning (Count) (All)'
# ]
train, val, test = [], [], []
# read data
q_type = 'Simple Question (Ellipsis)'
inference_file = f'{ROOT_PATH}/experiments/inference/test_{q_type}.json'
data = []
with open(inference_file) as json_file:
    data = json.load(json_file)
print(len(data))
# %%
correct_counter = 0
wrong_counter = 0
total_len = len(data)
for d in data:
    if d['is_correct'] == 1: correct_counter += 1
    if d['is_correct'] == 0: wrong_counter += 1

assert correct_counter + wrong_counter == total_len

print(f'Correct % {correct_counter/total_len}')
print(f'Worng % {wrong_counter/total_len}')

# %%
