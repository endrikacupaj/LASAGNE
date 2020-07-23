import torch
import torch.nn as nn
from args import get_parser
from models.transformer import Encoder, Decoder
from models.ner import NerNet
from models.coref import CorefNet
from utils import (INPUT, LOGICAL_FORM, NER, COREF)

# define device
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# read parser
parser = get_parser()
args = parser.parse_args()

class ConvQA(nn.Module):
    def __init__(self, vocabs):
        super().__init__()
        self.vocabs = vocabs
        self.encoder = Encoder(vocabs[INPUT], DEVICE)
        self.decoder = Decoder(vocabs[LOGICAL_FORM], DEVICE)
        self.ner = NerNet(len(vocabs[NER]))
        self.coref = CorefNet(len(vocabs[COREF]))

    def forward(self, src_tokens, trg_tokens):
        encoder_out = self.encoder(src_tokens)
        ner_out, ner_h = self.ner(encoder_out)
        coref_out = self.coref(torch.cat([encoder_out, ner_h], dim=-1))
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
            coref_out = self.coref(torch.cat([encoder_out, ner_h], dim=-1))

        return {
            'encoder_out': encoder_out,
            'ner': ner_out,
            'coref': coref_out
        }

    def _predict_lf(self, src_tokens, trg_tokens, encoder_out):
        with torch.no_grad():
            return self.decoder(src_tokens, trg_tokens, encoder_out)
