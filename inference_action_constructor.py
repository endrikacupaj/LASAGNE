import os
import time
import random
import logging
import torch
import numpy as np
import torch.optim
import torch.nn as nn
from pathlib import Path
from args import get_parser
from models.model import ConvQA
from csqa_dataset import CSQADataset
from torchtext.data import BucketIterator
from utils import AverageMeter, Scorer, Predictor
from utils import (INPUT, LOGICAL_FORM, NER, COREF, PAD_TOKEN, COREF_RANKING, PREDICATE, TYPE)
from utils import SingleTaskLoss, MultiTaskLoss

# set root path
ROOT_PATH = Path(os.path.dirname(__file__))

# read parser
parser = get_parser()
args = parser.parse_args()

# set logger
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[
                        # logging.FileHandler(f'{args.path_results}/test.log', 'w'),
                        logging.StreamHandler()
                    ])
logger = logging.getLogger(__name__)

# set a seed value
random.seed(args.seed)
np.random.seed(args.seed)
if torch.cuda.is_available():
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

# define device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(3)

def main():
    # load data
    dataset = CSQADataset(args.data_path)
    vocabs = dataset.get_vocabs()
    _, val_data, test_data = dataset.get_data()
    test_inference_data = dataset.get_inference_data()

    # load model
    model = ConvQA(vocabs).to(DEVICE)

    logger.info(f"=> loading checkpoint '{args.model_path}'")
    if DEVICE.type=='cpu':
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1')
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"=> loaded checkpoint '{args.model_path}' (epoch {checkpoint['epoch']})")

    logger.info('Test data prepared.')
    logger.info(f"Test data: {len(test_inference_data)}")

    # construct actions
    predictor = Predictor(model, vocabs, DEVICE)
    test_scorer = Scorer()
    test_scorer.construct_inference_actions(test_inference_data, predictor)

if __name__ == '__main__':
    main()