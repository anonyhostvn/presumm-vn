from torch import nn
from config_const import MODEL_DIM

from model_builder.utils import DEFAULT_MASK_MATRIX


class AbsDecoder(nn.Module):
    def __init__(self, n_head=8, n_decoder_block=1):
        super(AbsDecoder, self).__init__()
        decoder_layer = nn.TransformerDecoderLayer(batch_first=True, d_model=MODEL_DIM, nhead=n_head)
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=n_decoder_block)

    # tgt: đầu vào của decoder (batch_size * tgt_seq_len * embed_dim)
    # memory: embed của input sequence (batch_size * src_seq_len * embed_dim)
    # tgt_key_padding_mask: mask các token padding của tgt (batch_size * tgt_seq_len)
    # memory_key_padding_mask: mask các token padding của src (batch_size * src_seq_len)
    # tgt_mask: attention mask của decoder (để mô hình không nhìn được token ở tương lai)
    def forward(self, tgt, memory, tgt_key_padding_mask, memory_key_padding_mask, tgt_mask=DEFAULT_MASK_MATRIX):
        out = self.transformer_decoder(tgt=tgt, memory=memory,
                                       tgt_key_padding_mask=tgt_key_padding_mask,
                                       memory_key_padding_mask=memory_key_padding_mask,
                                       tgt_mask=tgt_mask)
        return out
