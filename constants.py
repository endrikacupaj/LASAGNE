import os
import torch
from pathlib import Path
from args import get_parser

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

# read parser
parser = get_parser()
args = parser.parse_args()

# define device
CUDA = 'cuda'
CPU = 'cpu'
DEVICE = torch.device(CUDA if torch.cuda.is_available() else CPU)

# fields
INPUT = 'input'
LOGICAL_FORM = 'logical_form'
NER = 'ner'
COREF = 'coref'
PREDICATE = 'predicate'
TYPE = 'type'
MULTITASK = 'multitask'

# helper tokens
START_TOKEN = '[START]'
END_TOKEN = '[END]'
CTX_TOKEN = '[CTX]'
PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'
SEP_TOKEN = '[SEP]'
NA_TOKEN = 'NA'

# ner tag
B = 'B'
I = 'I'
O = 'O'

# model
ENCODER_OUT = 'encoder_out'
DECODER_OUT = 'decoder_out'

# training
EPOCH = 'epoch'
STATE_DICT = 'state_dict'
BEST_VAL = 'best_val'
OPTIMIZER = 'optimizer'
CURR_VAL = 'curr_val'

# question types
TOTAL = 'total'
OVERALL = 'Overall'
CLARIFICATION = 'Clarification'
COMPARATIVE = 'Comparative Reasoning (All)'
LOGICAL = 'Logical Reasoning (All)'
QUANTITATIVE = 'Quantitative Reasoning (All)'
SIMPLE_COREFERENCED = 'Simple Question (Coreferenced)'
SIMPLE_DIRECT = 'Simple Question (Direct)'
SIMPLE_ELLIPSIS = 'Simple Question (Ellipsis)'
VERIFICATION = 'Verification (Boolean) (All)'
QUANTITATIVE_COUNT = 'Quantitative Reasoning (Count) (All)'
COMPARATIVE_COUNT = 'Comparative Reasoning (Count) (All)'

# action related
ENTITY = 'entity'
RELATION = 'relation'
VALUE = 'value'
PREV_ANSWER = 'prev_answer'
ACTION = 'action'

# other
QUESTION_TYPE = 'question_type'
IS_CORRECT = 'is_correct'
QUESTION = 'question'
ANSWER = 'answer'
ACTIONS = 'actions'
GOLD_ACTIONS = 'gold_actions'
RESULTS = 'results'
PREV_RESULTS = 'prev_results'
CONTEXT_QUESTION = 'context_question'
CONTEXT_ENTITIES = 'context_entities'
BERT_BASE_UNCASED = 'bert-base-uncased'