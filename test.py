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
from utils import AverageMeter, AccuracyScorer, Predictor
from utils import (INPUT, LOGICAL_FORM, NER, COREF, PAD_TOKEN)

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
                        logging.FileHandler(f'{args.path_results}/test.log', 'w'),
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
torch.cuda.set_device(1)

def main():
    # load data
    dataset = CSQADataset(args.data_path)
    vocabs = dataset.get_vocabs()
    _, val_data, test_data = dataset.get_data()

    # load model
    model = ConvQA(vocabs).to(DEVICE)

    # define loss function (criterion)
    ner_criterion = nn.CrossEntropyLoss()
    coref_criterion = nn.CrossEntropyLoss()
    lf_criterion = nn.CrossEntropyLoss(ignore_index=vocabs[LOGICAL_FORM].stoi[PAD_TOKEN])

    criterion = {
        'ner': ner_criterion,
        'coref': coref_criterion,
        'logical_form': lf_criterion
    }

    logger.info(f"=> loading checkpoint '{args.model_path}'")
    if DEVICE.type=='cpu':
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1')
    args.start_epoch = checkpoint['epoch']
    model.load_state_dict(checkpoint['state_dict'])
    logger.info(f"=> loaded checkpoint '{args.model_path}' (epoch {checkpoint['epoch']})")

    # prepare training and validation loader
    val_loader, test_loader = BucketIterator.splits((val_data, test_data),
                                                    batch_size=args.batch_size,
                                                    sort_within_batch=False,
                                                    sort_key=lambda x: len(x.input),
                                                    device=DEVICE)

    logger.info('Loaders prepared.')
    logger.info(f"Validation data: {len(val_data.examples)}")
    logger.info(f"Test data: {len(test_data.examples)}")

    # calculate loss
    val_loss = test(val_loader, model, vocabs, criterion)
    logger.info(f'* Val Loss: {val_loss:.3f}')
    test_loss = test(test_loader, model, vocabs, criterion)
    logger.info(f'* Test Loss: {test_loss:.3f}')

    # calculate accuracy
    predictor = Predictor(model, vocabs, DEVICE)
    val_scorer = AccuracyScorer()
    test_scorer = AccuracyScorer()
    val_scorer.data_score(val_data.examples, predictor)
    test_scorer.data_score(test_data.examples, predictor)

    # log results
    logger.info(f'* Val Data Results:')
    logger.info(f'\t NER: {val_scorer.ner_accuracy():.4f}')
    logger.info(f'\t Coref: {val_scorer.coref_accuracy():.4f}')
    logger.info(f'\t Logical Form: {val_scorer.lf_accuracy():.4f}')
    logger.info(f'\t Total Accuracy: {val_scorer.total_accuracy():.4f}')

    logger.info(f'* Test Data Results:')
    logger.info(f'\t NER: {test_scorer.ner_accuracy():.4f}')
    logger.info(f'\t Coref: {test_scorer.coref_accuracy():.4f}')
    logger.info(f'\t Logical Form: {test_scorer.lf_accuracy():.4f}')
    logger.info(f'\t Total Accuracy: {test_scorer.total_accuracy():.4f}')

def test(loader, model, vocabs, criterion):
    losses = AverageMeter()

    model.eval()

    with torch.no_grad():
        for _, batch in enumerate(loader):
            # get inputs
            input = batch.input
            logical_form = batch.logical_form
            ner = batch.ner
            coref = batch.coref

            # compute output
            output = model(input, logical_form[:, :-1])

            # prepare targets
            logical_form = logical_form[:, 1:].contiguous().view(-1) # (batch_size * trg_len)
            ner = ner.contiguous().view(-1)
            coref = coref.contiguous().view(-1)

            # compute loss
            loss = criterion['logical_form'](output['logical_form'], logical_form) * args.lf_weight
            loss += criterion['ner'](output['ner'], ner) * args.ner_weight
            loss += criterion['coref'](output['coref'], coref) * args.coref_weight

            # record loss
            losses.update(loss.data, input.size(0))

    return losses.avg

if __name__ == '__main__':
    main()