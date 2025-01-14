import torch
from torch import nn
from torch.nn import Sequential

from config_const import CACHED_MODEL_PATH, MODEL_DIM, MAX_SEQ_LENGTH
from model_builder.abs_decoder import AbsDecoder
from model_builder.phobert_model import PhoBert
from model_builder.utils import get_cls_embed, padding_and_stack_cls


class AbsBertSumm(nn.Module):
    def __init__(self, vocab_size, tokenizer):
        super(AbsBertSumm, self).__init__()
        self.phase1_bert = PhoBert(large=False, temp_dir=CACHED_MODEL_PATH, is_freeze=False)
        self.tokenizer = tokenizer
        self.phase2_trans_decoder = AbsDecoder()
        self.embedding_inp = self.phase1_bert.model.embeddings
        self.token_decode = Sequential(
            nn.Linear(MODEL_DIM, vocab_size)
        )

    def greedy_inference(self, src_ids, src_pad_mask, src_token_type, src_cls_pos, max_len=50):
        embed_phase1 = self.phase1_bert(input_ids=src_ids,
                                        token_type_ids=src_token_type,
                                        attention_mask=src_pad_mask)

        cls_embed = get_cls_embed(tok_embed=embed_phase1, cls_pos=src_cls_pos)  # n_batch * n_cls * n_embed
        padded_cls_embed, pad_mask = padding_and_stack_cls(cls_embed)  # n_batch * MAX_SEQ_LENGTH * n_embed

        init_tgt_seq = torch.tensor(
            [self.tokenizer.phobert_tokenizer.cls_token] + [0 for i in range(MAX_SEQ_LENGTH - 1)]).unsqi
        init_tgt_mask = torch.tensor([1] + [0 for i in range(MAX_SEQ_LENGTH - 1)])
        init_tgt_token_type_ids = torch.tensor([1] + [0 for i in range(MAX_SEQ_LENGTH - 1)])

        for i in range(max_len - 1):
            recent_ids = i + 1
            embed_tgt = self.embedding_inp(input_ids=init_tgt_seq,
                                           token_type_ids=init_tgt_token_type_ids)  # n_batch * MAX_SEQ_LENGTH * 768
            out = self.phase2_trans_decoder(tgt=embed_tgt, tgt_key_padding_mask=init_tgt_mask,
                                            memory=padded_cls_embed, memory_key_padding_mask=pad_mask)
            logits = self.token_decode(out)

            logits

    def forward(self, src_ids, src_pad_mask, src_token_type, src_cls_pos
                , tgt_ids, tgt_pad_mask, tgt_token_type):
        """
        :param src_ids: embeding token của src  (batch_size * seq_len)
        :param src_pad_mask: đánh dấu padding của src (để attention không tính đến nó nữa) (batch_size * seq_len)
        :param src_token_type: đánh dấu đoạn A và B của câu src (batch_size * seq_len)
        :param is_freeze_phase1: có đóng băng phase 1 (pho BERT) hay không (boolean)
        :param src_cls_pos: vị trí của các token cls trong src seq (batch_size * num_cls)
        :param tgt_ids: embeding của target (n_batch, n_seq)
        :param tgt_pad_mask: padding mask của target (n_batch, n_seq)
        :param tgt_token_type: đánh dấu đoạn A và B (batch_size * seq_len)
        :return:
        """
        # n_batch * n_tokens * n_embed_dim
        embed_phase1 = self.phase1_bert(input_ids=src_ids,
                                        token_type_ids=src_token_type,
                                        attention_mask=src_pad_mask)

        cls_embed = get_cls_embed(tok_embed=embed_phase1, cls_pos=src_cls_pos)  # n_batch * n_cls * n_embed
        padded_cls_embed, pad_mask = padding_and_stack_cls(cls_embed)  # n_batch * MAX_SEQ_LENGTH * n_embed

        embed_tgt = self.embedding_inp(input_ids=tgt_ids,
                                       token_type_ids=tgt_token_type)  # n_batch * MAX_SEQ_LENGTH * 768

        out = self.phase2_trans_decoder(tgt=embed_tgt, tgt_key_padding_mask=tgt_pad_mask,
                                        memory=padded_cls_embed, memory_key_padding_mask=pad_mask)

        logits = self.token_decode(out)
        return logits
