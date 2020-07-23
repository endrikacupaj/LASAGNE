import torch
import torch.nn as nn
from args import get_parser
from models.transformer import Encoder, Decoder
from utils import (INPUT, LOGICAL_FORM, NER, COREF)

# read parser
parser = get_parser()
args = parser.parse_args()

class LstmFlatten(nn.Module):
    def forward(self, x):
        return x[0].squeeze(1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(-1, x.shape[-1])

class NerNet(nn.Module):
    def __init__(self, tags, dropout=args.dropout):
        super(NerNet, self).__init__()
        self.ner_lstm = nn.Sequential(
            nn.LSTM(input_size=args.embDim, hidden_size=args.embDim, batch_first=True),
            LstmFlatten(),
            nn.LeakyReLU()
        )

        self.ner_linear = nn.Sequential(
            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(args.embDim, tags),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        h = self.ner_lstm(x)
        return self.ner_linear(h), h
