#%%
from kb.knowledge_base import *
from kb.sampling import *
from kb.operations import OPs
from kb.actions import Actions
from kb import agent

# %%
kb = KnowledgeBase(wikidata_path="data/kb/")
# %%
from glob import glob
data = []
dataset_files = glob('/data/joanplepi/csqa/data/sample/train/*/*.json')
for f in dataset_files:
    with open(f) as json_file:
        data.append(json.load(json_file))
dialogs = []

question_types = {}
for conv_nr, conversation in enumerate(data):
    prev_user = None
    prev_system = None
    is_clarification = False
    turns = len(conversation) // 2
    for i in range(turns):
        if is_clarification:
            is_clarification = False
            continue

        user = conversation[2*i]
        system = conversation[2*i + 1]

        key = user['question-type']
        if key in question_types:
            question_types[key] += 1
        else:
            question_types[key] = 1
        
        dialogs.append((user, system))
print(len(dialogs))
print(question_types)
# %%
from importlib import reload
import graph as graph_to_reload
reload(graph_to_reload)

from graph import Graph

#prev_dialog = None
all_graphs = []

for i, dialog in enumerate(dialogs):
    subgraph = Graph()
    user = dialog[0]
    system = dialog[1]
    
    type_list = set(user['type_list'])
    pred_list = set(user['relations'])

    #if prev_dialog is not None:
    #    type_list|= set(prev_dialog[0]['type_list'])
    #    pred_list |=  set(prev_dialog[0]['relations'])

    #prev_dialog = dialog
    # Q618123
    # P54
    for type in type_list: 
        if type in kb.type_pred_type:
            subgraph.add_if_exists(type, kb.type_pred_type, pred_list)
        if type in kb.rev_type_pred:
            subgraph.add_if_exists(type, kb.rev_type_pred, pred_list)
    
    all_graphs.append(subgraph)

# %%
import torch
from torch_geometric.data import Data
from torch_geometric.nn import GATConv

typ