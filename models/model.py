import torch
import torch.nn as nn
#from args import get_parser
from models.transformer import Encoder, Decoder

# import constants
from constants import *

class ConvQA(nn.Module):
    def __init__(self, vocabs):
        super().__init__()
        self.vocabs = vocabs
        self.encoder = Encoder(vocabs[INPUT], DEVICE)
        self.decoder = Decoder(vocabs[LOGICAL_FORM], DEVICE)
        self.ner = NerNet(len(vocabs[NER]))
        self.coref = SeqNet(len(vocabs[COREF]))
        self.predicate_cls = SeqNet(len(vocabs[PREDICATE]))
        self.type_cls = SeqNet(len(vocabs[TYPE]))

    def forward(self, src_tokens, trg_tokens):
        encoder_out = self.encoder(src_tokens)
        ner_out, ner_h = self.ner(encoder_out)
        coref_out = self.coref(torch.cat([encoder_out, ner_h], dim=-1))
        decoder_out, decoder_h = self.decoder(src_tokens, trg_tokens, encoder_out)
        encoder_ctx = encoder_out[:, -1:, :].expand(decoder_h.shape)
        predicate_out = self.predicate_cls(torch.cat([encoder_ctx, decoder_h], dim=-1))
        type_out = self.type_cls(torch.cat([encoder_ctx, decoder_h], dim=-1))

        return {
            LOGICAL_FORM: decoder_out,
            NER: ner_out,
            COREF: coref_out,
            PREDICATE: predicate_out,
            TYPE: type_out
        }

    def _predict_encoder(self, src_tensor):
        with torch.no_grad():
            encoder_out = self.encoder(src_tensor)
            ner_out, ner_h = self.ner(encoder_out)
            coref_out = self.coref(torch.cat([encoder_out, ner_h], dim=-1))

        return {
            ENCODER_OUT: encoder_out,
            NER: ner_out,
            COREF: coref_out
        }

    def _predict_decoder(self, src_tokens, trg_tokens, encoder_out):
        with torch.no_grad():
            decoder_out, decoder_h = self.decoder(src_tokens, trg_tokens, encoder_out)
            encoder_ctx = encoder_out[:, -1:, :].expand(decoder_h.shape)
            predicate_out = self.predicate_cls(torch.cat([encoder_ctx, decoder_h], dim=-1))
            type_out = self.type_cls(torch.cat([encoder_ctx, decoder_h], dim=-1))

            return {
                DECODER_OUT: decoder_out,
                PREDICATE: predicate_out,
                TYPE: type_out
            }

class LstmFlatten(nn.Module):
    def forward(self, x):
        return x[0].squeeze(1)

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(-1, x.shape[-1])

class NerNet(nn.Module):
    def __init__(self, tags, dropout=dropout):
        super(NerNet, self).__init__()
        self.ner_lstm = nn.Sequential(
            nn.LSTM(input_size=args.emb_dim, hidden_size=args.emb_dim, batch_first=True),
            LstmFlatten(),
            nn.LeakyReLU()
        )

        self.ner_linear = nn.Sequential(
            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(args.emb_dim, tags)
        )

    def forward(self, x):
        h = self.ner_lstm(x)
        return self.ner_linear(h), h

class SeqNet(nn.Module):
    def __init__(self, tags, dropout=dropout):
        super(SeqNet, self).__init__()
        self.seq_net = nn.Sequential(
            nn.Linear(args.emb_dim*2, args.emb_dim),
            nn.LeakyReLU(),
            Flatten(),
            nn.Dropout(dropout),
            nn.Linear(args.emb_dim, tags)
        )

    def forward(self, x):
        return self.seq_net(x)
