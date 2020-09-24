import os
import time
import random
import logging
import torch
import numpy as np
import torch.optim
import torch.nn as nn
from pathlib import Path
from model import ConvQA
from dataset import CSQADataset
from torchtext.data import BucketIterator
from utils import SingleTaskLoss, MultiTaskLoss, AverageMeter, Scorer, Predictor

# import constants
from constants import *

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

def main():
    # load data
    dataset = CSQADataset()
    vocabs = dataset.get_vocabs()
    _, val_data, test_data = dataset.get_data()
    _, val_helper, test_helper = dataset.get_data_helper()

    # load model
    model = ConvQA(vocabs).to(DEVICE)

    # define loss function (criterion)
    criterion = {
        LOGICAL_FORM: SingleTaskLoss,
        NER: SingleTaskLoss,
        COREF: SingleTaskLoss,
        GRAPH: SingleTaskLoss,
        MULTITASK: MultiTaskLoss
    }[args.task](ignore_index=vocabs[LOGICAL_FORM].stoi[PAD_TOKEN])

    logger.info(f"=> loading checkpoint '{args.model_path}'")
    if DEVICE.type=='cpu':
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1', map_location='cpu')
    else:
        checkpoint = torch.load(f'{ROOT_PATH}/{args.model_path}', encoding='latin1')
    args.start_epoch = checkpoint[EPOCH]
    model.load_state_dict(checkpoint[STATE_DICT])
    logger.info(f"=> loaded checkpoint '{args.model_path}' (epoch {checkpoint[EPOCH]})")

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
    logger.info(f'* Val Loss: {val_loss:.4f}')
    test_loss = test(test_loader, model, vocabs, criterion)
    logger.info(f'* Test Loss: {test_loss:.4f}')

    # calculate accuracy
    predictor = Predictor(model, vocabs, DEVICE)
    # val_scorer = Scorer()
    test_scorer = Scorer()
    # val_scorer.data_score(val_data.examples, val_helper, predictor)
    test_scorer.data_score(test_data.examples, test_helper, predictor)
    test_scorer.write_results()

    # log results
    for partition, results in [['Test', test_scorer.results]]: # [['Val', val_scorer.results], ['Test', test_scorer.results]]:
        logger.info(f'* {partition} Data Results:')
        for question_type, question_type_results in results.items():
            logger.info(f'\t{question_type}:')
            for task, task_result in question_type_results.items():
                logger.info(f'\t\t{task}: {task_result.accuracy:.4f}')

def test(loader, model, vocabs, criterion):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for _, batch in enumerate(loader):
            # get inputs
            input = batch.input
            logical_form = batch.logical_form
            ner = batch.ner
            coref = batch.coref
            graph = batch.graph

            # compute output
            output = model(input, logical_form[:, :-1])

            # prepare targets
            target = {
                LOGICAL_FORM: logical_form[:, 1:].contiguous().view(-1), # (batch_size * trg_len)
                NER: ner.contiguous().view(-1),
                COREF: coref.contiguous().view(-1),
                GRAPH: graph[:, 1:].contiguous().view(-1)
            }

            # compute loss
            loss = criterion(output, target) if args.task == MULTITASK else criterion(output[args.task], target[args.task])

            # record loss
            losses.update(loss.data, input.size(0))

    return losses.avg

if __name__ == '__main__':
    main()