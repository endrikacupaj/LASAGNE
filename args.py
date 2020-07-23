import argparse

def get_parser():

    parser = argparse.ArgumentParser(description='ConvQA')
    # general
    parser.add_argument('--seed', default=1234, type=int)
    parser.add_argument('--no-cuda', action='store_true')
    parser.add_argument('--cuda_device', default=0, type=int)

    # data
    parser.add_argument('--data_path', default='/data/final/csqa/sample')
    parser.add_argument('--dataset', default='csqa', choices=['csqa', 'convqarr'], type=str)

    # experiments
    parser.add_argument('--snapshots', default='experiments/snapshots', type=str)
    parser.add_argument('--path_results', default='experiments/results', type=str)

    # model
    parser.add_argument('--embDim', default=300, type=int)
    parser.add_argument('--dropout', default=0.1, type=int)
    parser.add_argument('--heads', default=6, type=int)
    parser.add_argument('--layers', default=2, type=int)
    parser.add_argument('--max_positions', default=500, type=int)
    parser.add_argument('--pf_dim', default=300, type=int)

    # training
    parser.add_argument('--lr', default=0.0001, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--epochs', default=5, type=int)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--valfreq', default=1, type=int)
    parser.add_argument('--resume', default='', type=str)
    parser.add_argument('--clip', default=5, type=int)
    parser.add_argument('--ner_weight', default=1.0, type=float)
    parser.add_argument('--coref_weight', default=1.0, type=float)
    parser.add_argument('--lf_weight', default=1.0, type=float)
    parser.add_argument('--batch_size', default=100, type=int)

    # test
    parser.add_argument('--model_path', default='experiments/snapshots/ConvQARR_model_e5_v-0.18.pth.tar', type=str)

    return parser