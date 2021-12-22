from tokenize_input.summ_tokenize import SummTokenize
import json
import torch
from tqdm import tqdm

if __name__ == '__main__':
    tokenizer = SummTokenize()

    phase = 'test'
    with open(f'/Users/LongNH/Workspace/presumm-vn/ext_bert_data/json_data.{phase}.json', 'r') as f:
        lis_data = json.load(f)
    for i, data in enumerate(tqdm(lis_data)):
        (src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask), lis_tgt = \
            tokenizer.tokenizing_ext_input(**data, is_pad=True)
        ext_stacked_inp = torch.stack([
            src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask,
            lis_tgt
        ], dim=0)
        torch.save(ext_stacked_inp, f'/Users/LongNH/Workspace/data/ext_bert_data/{phase}/ext_bert_data{i}.pt')
