import os
import sys
import time
import random
import logging
import torch
import numpy as np
import torch.optim
import torch.nn as nn
from pathlib import Path
from args import get_parser
from model import ConvQA
from dataset import CSQADataset
from torchtext.data import BucketIterator
from utils import (NoamOpt, AverageMeter,
                    SingleTaskLoss, MultiTaskLoss,
                    save_checkpoint, init_weights)

# import constants
from constants import *

# set logger
logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%d/%m/%Y %I:%M:%S %p',
                    level=logging.INFO,
                    handlers=[
                        logging.FileHandler(f'{args.path_results}/train_{args.task}.log', 'w'),
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
    train_data, val_data, _ = dataset.get_data()

    # load model
    model = ConvQA(vocabs).to(DEVICE)

    # initialize model weights
    init_weights(model)

    logger.info(f'The model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters')

    # define loss function (criterion)
    criterion = {
        LOGICAL_FORM: SingleTaskLoss,
        NER: SingleTaskLoss,
        COREF: SingleTaskLoss,
        GRAPH: SingleTaskLoss,
        MULTITASK: MultiTaskLoss
    }[args.task](ignore_index=vocabs[LOGICAL_FORM].stoi[PAD_TOKEN])

    # define optimizer
    optimizer = NoamOpt(torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"=> loading checkpoint '{args.resume}''")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint[EPOCH]
            best_val = checkpoint[BEST_VAL]
            model.load_state_dict(checkpoint[STATE_DICT])
            optimizer.optimizer.load_state_dict(checkpoint[OPTIMIZER])
            logger.info(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint[EPOCH]})")
        else:
            logger.info(f"=> no checkpoint found at '{args.resume}'")
            best_val = float('inf')
    else:
        best_val = float('inf')

    # prepare training and validation loader
    train_loader, val_loader = BucketIterator.splits((train_data, val_data),
                                                    batch_size=args.batch_size,
                                                    sort_within_batch=False,
                                                    sort_key=lambda x: len(x.input),
                                                    device=DEVICE)

    logger.info('Loaders prepared.')
    logger.info(f"Training data: {len(train_data.examples)}")
    logger.info(f"Validation data: {len(val_data.examples)}")
    logger.info(f'Question example: {train_data.examples[0].input}')
    logger.info(f'Logical form example: {train_data.examples[0].logical_form}')
    logger.info(f"Unique tokens in input vocabulary: {len(vocabs[INPUT])}")
    logger.info(f"Unique tokens in logical form vocabulary: {len(vocabs[LOGICAL_FORM])}")
    logger.info(f"Unique tokens in ner vocabulary: {len(vocabs[NER])}")
    logger.info(f"Unique tokens in coref vocabulary: {len(vocabs[COREF])}")
    logger.info(f"Number of nodes in the graph: {len(vocabs[GRAPH])}")
    logger.info(f'Batch: {args.batch_size}')
    logger.info(f'Epochs: {args.epochs}')

    # run epochs
    for epoch in range(args.start_epoch, args.epochs):
        # train for one epoch
        train(train_loader, model, vocabs, criterion, optimizer, epoch)

        # evaluate on validation set
        if (epoch+1) % args.valfreq == 0:
            val_loss = validate(val_loader, model, vocabs, criterion)
            # if val_loss < best_val:
            best_val = min(val_loss, best_val) # log every validation step
            save_checkpoint({
                EPOCH: epoch + 1,
                STATE_DICT: model.state_dict(),
                BEST_VAL: best_val,
                OPTIMIZER: optimizer.optimizer.state_dict(),
                CURR_VAL: val_loss})
            logger.info(f'* Val loss: {val_loss:.4f}')

def train(train_loader, model, vocabs, criterion, optimizer, epoch):
    batch_time = AverageMeter()
    losses = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()
    for i, batch in enumerate(train_loader):
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

        # compute gradient and do Adam step
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        logger.info(f'Epoch: {epoch+1} - Train loss: {losses.val:.4f} ({losses.avg:.4f}) - Batch: {((i+1)/len(train_loader))*100:.2f}% - Time: {batch_time.sum:0.2f}s')

def validate(val_loader, model, vocabs, criterion):
    losses = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for _, batch in enumerate(val_loader):
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