import torch
import torch.nn as nn
from args import get_parser
from models.transformer import Encoder, Decoder
from utils import (INPUT, LOGICAL_FORM, NER, COREF)

# define device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read parser
parser = get_parser()
args = parser.parse_args()

class ConvQA(nn.Module):
    def __init__(self, vocabs):
        super().__init__()
        # define model
        self.vocabs = vocabs
        self.encoder = Encoder(vocabs[INPUT], DEVICE)
        self.decoder = Decoder(vocabs[LOGICAL_FORM], DEVICE)
        self.ner = SeqNet(len(vocabs[NER]))
        self.coref = SeqNet(len(vocabs[COREF]))

    def forward(self, src_tokens, trg_tokens):
        encoder_out = self.encoder(src_tokens)
        ner_out, ner_h = self.ner(encoder_out)
        coref_out, _ = self.coref(encoder_out * ner_h)
        lf_out = self.decoder(src_tokens, trg_tokens, encoder_out)

        return {
            'ner': ner_out,
            'coref': coref_out,
            'logical_form': lf_out
        }

    def _predict(self, src_tensor):
        with torch.no_grad():
            encoder_out = self.encoder(src_tensor)
            ner_out, ner_h = self.ner(encoder_out)
            coref_out, coref_h = self.coref(encoder_out * ner_h)

        return {
            'encoder_out': encoder_out,
            'ner': ner_out,
            'coref': coref_out
        }

    def _predict_lf(self, src_tokens, trg_tokens, encoder_out):
        with torch.no_grad():
            return self.decoder(src_tokens, trg_tokens, encoder_out)

class SeqNet(nn.Module):
    def __init__(self, tags, dropout=args.dropout):
        super(SeqNet, self).__init__()

        self.lstm = nn.LSTM(input_size=args.embDim, hidden_size=args.embDim, batch_first=True)

        self.dropout = nn.Dropout(dropout)
        # self.leaky_relu = nn.LeakyReLU()

        self.linear = nn.Linear(args.embDim, tags)
        self.log_softmax = nn.LogSoftmax(dim=1)


    def forward(self, x):
        x, _ = self.lstm(x)

        x = x.squeeze(1)

        h = self.dropout(x)
        # h = self.leaky_relu(x)

        x = h.contiguous().view(-1, x.shape[-1])

        x = self.linear(x)

        x = self.log_softmax(x)

        return x, h
