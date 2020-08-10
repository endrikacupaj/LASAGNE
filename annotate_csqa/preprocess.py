import os
import time
import logging
import json
import argparse
from glob import glob
from pathlib import Path
from knowledge_graph.knowledge_graph import KnowledgeGraph
from action_annotators.annotate import ActionAnnotator
from ner_annotators.annotate import NERAnnotator
ROOT_PATH = Path(os.path.dirname(__file__)).parent

# add arguments to parser
parser = argparse.ArgumentParser(description='Preprocess CSQA dataset')
parser.add_argument('--partition', default='train', choices=['train', 'val', 'test'], type=str, help='partition to preprocess')
parser.add_argument('--annotation_task', default='all', choices=['actions', 'ner', 'all'], help='annotation task to perform')
parser.add_argument('--read_folder', default='/data/CSQA_v9', help='folder to read conversations')
parser.add_argument('--write_folder', default='/data/final/csqa', help='folder to write the annotated conversations')
args = parser.parse_args()

# read data and create csqa data dict
# dict: partition -> folder -> file -> conversation
csqa_files = glob(f'{ROOT_PATH}{args.read_folder}/{args.partition}/*/*.json')
csqa_data = {}
print(f'Reading folders for partition {args.partition}')
for path in csqa_files:
    folder = path.rsplit('/', 1)[0].rsplit('/', 1)[-1]
    file = path.rsplit('/', 1)[-1]
    if folder not in csqa_data:
        csqa_data[folder] = {}

    with open(path) as json_file:
        csqa_data[folder][file] = json.load(json_file)
print(f'Done, {len(csqa_data)} folders loaded!')

# load kg
kg = KnowledgeGraph()

# create ner and action annotator
if args.annotation_task == 'all':
    action_annotator = ActionAnnotator(kg)
    ner_annotator = NERAnnotator(kg, args.partition)
elif args.annotation_task == 'actions':
    action_annotator = ActionAnnotator(kg)
elif args.annotation_task == 'ner':
    ner_annotator = NERAnnotator(kg, args.partition)

# create annotated data
total_conv = len(csqa_files)
conv_counter = 0
tic = time.perf_counter()
for folder in csqa_data.keys():
    csqa_folder = csqa_data[folder]
    for file in csqa_folder.keys():
        # get conversation
        conversation = csqa_folder[file]

        # annotate conversation
        if args.annotation_task == 'all':
            conversation = action_annotator(conversation)
            conversation = ner_annotator(conversation)
        elif args.annotation_task == 'actions':
            conversation = action_annotator(conversation)
        elif args.annotation_task == 'ner':
            conversation = ner_annotator(conversation)

        # create path
        conv_path = f'{str(ROOT_PATH)}{args.write_folder}/{args.partition}/{folder}'
        Path(conv_path).mkdir(parents=True, exist_ok=True)

        # write conversation
        with open(f'{conv_path}/{file}', 'w') as json_file:
            json.dump(conversation, json_file, ensure_ascii=False, indent=4)

        conv_counter += 1
        toc = time.perf_counter()
        print(f'====> Finished /{args.partition}/{folder}/{file} -- {((conv_counter)/total_conv)*100:.2f}% -- {toc - tic:0.2f}s')
