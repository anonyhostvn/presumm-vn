import json
import torch

from model_builder.abs_bert_summ import AbsBertSumm
from model_builder.ext_bert_summ_pylight import ExtBertSummPylight
from tokenize_input.summ_tokenize import SummTokenize

THRESHOLD = 0.3


def inference_json(json_path, model):
    tokenizer = SummTokenize()

    with open(json_path, 'r') as f:
        data = json.load(f)

    (src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask) \
        , tgt = \
        tokenizer.tokenizing_formatted_input(**data, is_pad=True)

    src_inp_ids = torch.unsqueeze(src_inp_ids, 0)
    src_tok_type_ids = torch.unsqueeze(src_tok_type_ids, 0)
    src_lis_cls_pos = torch.unsqueeze(src_lis_cls_pos, 0)
    src_mask = torch.unsqueeze(src_mask, 0)

    lis_res = model.predict_step(batch=[
        src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask,
        tgt_inp_ids, tgt_tok_type_ids, tgt_lis_cls_pos, tgt_mask],
        batch_idx=0)
    print(lis_res[0])


if __name__ == '__main__':
    tokenizer = SummTokenize()
    vocab_size = tokenizer.phobert_tokenizer.vocab_size
    abs_bert_summ_pylight = AbsBertSumm(vocab_size=vocab_size, )
    inference_json(json_path='/Users/LongNH/Workspace/presumm-vn/inferences/test_sample.json',
                   model=abs_bert_summ_pylight)
