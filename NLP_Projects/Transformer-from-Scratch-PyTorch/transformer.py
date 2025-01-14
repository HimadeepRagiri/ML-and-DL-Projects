import torch.nn as nn
from .encoder import Encoder
from .decoder import Decoder

# =========================================
# Transformer
# =========================================
class Transformer(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, d_ff, src_vocab_size, tgt_vocab_size, max_len, dropout=0.1):
        super().__init__()
        self.encoder = Encoder(num_layers, d_model, num_heads, d_ff, src_vocab_size, max_len, dropout)
        self.decoder = Decoder(num_layers, d_model, num_heads, d_ff, tgt_vocab_size, max_len, dropout)
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        enc_output = self.encoder(src, src_mask)
        dec_output = self.decoder(tgt, enc_output, src_mask, tgt_mask)
        output = self.fc_out(dec_output)
        return output

