import torch
import torch.nn as nn
from args import get_parser
from models.transformer import Encoder, Decoder
from utils import (INPUT, LOGICAL_FORM, NER, COREF)

# read parser
parser = get_parser()
args = parser.parse_args()

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(-1, x.shape[-1])

class CorefNet(nn.Module):
    def __init__(self, tags, dropout=args.dropout):
        super(CorefNet, self).__init__()
        self.coref_net = nn.Sequential(
            nn.Linear(args.embDim*2, args.embDim),
            nn.LeakyReLU(),
            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(args.embDim, tags),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        return self.coref_net(x)
