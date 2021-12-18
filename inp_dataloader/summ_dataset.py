from torch.utils.data import Dataset
import json
from tokenize_input.summ_tokenize import SummTokenize


class SummDataset(Dataset):
    def __init__(self, json_filepath):
        with open(json_filepath, 'r') as f:
            list_src_tgt = json.load(f)
            self.list_src_tgt = list_src_tgt
        self.tokenizer = SummTokenize()

    def __len__(self):
        return len(self.list_src_tgt)

    def __getitem__(self, idx):
        src_tokenized, tgt_tokenized = self.tokenizer.tokenizing_formatted_input(**self.list_src_tgt[idx], is_pad=True)
        src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask = src_tokenized
        tgt_inp_ids, tgt_tok_type_ids, tgt_lis_cls_pos, tgt_mask = tgt_tokenized
        return (src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask
                , tgt_inp_ids, tgt_tok_type_ids, tgt_lis_cls_pos, tgt_mask)
