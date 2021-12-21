import torch

from model_builder.abs_bert_summ import AbsBertSumm
from model_builder.abs_decoder import AbsDecoder
from config_const import MAX_SEQ_LENGTH, MODEL_DIM
from model_builder.phobert_model import PhoBert
from tokenize_input.summ_tokenize import SummTokenize
from torch import nn
from torchinfo import summary

if __name__ == '__main__':
    tokenizer = SummTokenize()
    abs_bert_summ = AbsBertSumm(vocab_size=tokenizer.phobert_tokenizer.vocab_size)

    BATCH_SIZE = 8
    src_ids = torch.randint(size=(BATCH_SIZE, MAX_SEQ_LENGTH), low=0, high=100)
    tok_type_ids = torch.randint(size=(BATCH_SIZE, MAX_SEQ_LENGTH), low=0, high=1)
    src_mask = torch.randint(size=(BATCH_SIZE, MAX_SEQ_LENGTH), low=0, high=1).bool()
    lis_cls_pos = [torch.randint(size=(i + 1,), low=0, high=256) for i in range(BATCH_SIZE)]

    tgt_ids = torch.randint(size=(BATCH_SIZE, MAX_SEQ_LENGTH), low=0, high=100)
    tgt_mask = torch.randint(size=(BATCH_SIZE, MAX_SEQ_LENGTH), low=0, high=1).bool()
    tgt_token_type = torch.randint(size=(BATCH_SIZE, MAX_SEQ_LENGTH), low=0, high=1)
    # src_ids, src_pad_mask, src_token_type, src_cls_pos
    # , tgt_ids, tgt_pad_mask, tgt_token_type
    summary(model=abs_bert_summ,
            input_data=[
                src_ids
                , src_mask
                , tok_type_ids
                , lis_cls_pos
                , tgt_ids
                , tgt_mask
                , tgt_token_type
            ])

    logits = abs_bert_summ(src_ids=src_ids, src_pad_mask=src_mask,
                           src_token_type=tok_type_ids,
                           # is_freeze_phase1=True,
                           src_cls_pos=lis_cls_pos,
                           tgt_ids=tgt_ids, tgt_pad_mask=tgt_mask,
                           tgt_token_type=tgt_token_type)

    # prob = nn.Softmax(dim=2)(logits)
    pred_ids = torch.argmax(logits, dim=2)
    print(pred_ids.shape)

# # Đọc ở đây để hiểu input
# if __name__ == '__main__':
#     abs_enc = AbsDecoder()
#     BATCH_SIZE = 32
#
#     inp_memory = torch.rand(BATCH_SIZE, MAX_SEQ_LENGTH, MODEL_DIM)  # (seq_len * batch_size * embed_dim)
#     inp_memory_key_padding_mask = torch.randint(size=(BATCH_SIZE, MAX_SEQ_LENGTH), low=0,
#                                                 high=1).bool()  # Mask cho embed từ encoder
#
#     inp_tgt = torch.rand(BATCH_SIZE, MAX_SEQ_LENGTH, MODEL_DIM)  # (seq_len * batch_size * embed_dim)
#     inp_tgt_key_padding_mask = torch.randint(size=(BATCH_SIZE, MAX_SEQ_LENGTH), low=0,
#                                              high=1).bool()  # Mask cho embed từ decoder
#     inp_tgt_mask = torch.randint(size=(MAX_SEQ_LENGTH, MAX_SEQ_LENGTH), low=0, high=1).bool()
#
#     print(abs_enc(inp_memory, inp_tgt,
#                   tgt_key_padding_mask=inp_memory_key_padding_mask,
#                   memory_key_padding_mask=inp_tgt_key_padding_mask).shape)
#
# if __name__ == '__main__':
#     pho_bert = PhoBert(large=False, temp_dir='../model_cache')
#     seq_len = MAX_SEQ_LENGTH
#     BATCH_SIZE = 1
#     inputs = torch.randint(low=0, high=200, size=(BATCH_SIZE, seq_len))
#     masks = torch.randint(low=0, high=2, size=(BATCH_SIZE, seq_len))
#     segments = torch.randint(low=0, high=2, size=(BATCH_SIZE, seq_len))
#     # inp_cls_pos = torch.randint(low=0, high=255, size=(BATCH_SIZE, 3))
#
#     # print(inp_cls_pos.shape)
#     with torch.no_grad():
#         features = pho_bert(input_ids=inputs, token_type_ids=segments, attention_mask=masks, is_freeze=True)
#         # print(get_cls_embed(tok_embed=features, cls_pos=inp_cls_pos).shape)
#         print(features.shape)
