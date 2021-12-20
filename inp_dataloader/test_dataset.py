from inp_dataloader.summ_dataset import SummDataset
from torch.utils.data import DataLoader
import torch

from model_builder.abs_bert_summ import AbsBertSumm
from torch.nn.functional import one_hot
from torch.nn import CrossEntropyLoss

if __name__ == '__main__':
    ds = SummDataset('/Users/LongNH/Workspace/presumm-vn/json_data/json_data.train.json')
    dl = DataLoader(dataset=ds, batch_size=8, shuffle=True)
    vocab_size = ds.tokenizer.phobert_tokenizer.vocab_size
    abs_bert_summ = AbsBertSumm(vocab_size=vocab_size)
    loss_fn = CrossEntropyLoss()

    for (src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask
         , tgt_inp_ids, tgt_tok_type_ids, tgt_lis_cls_pos, tgt_mask) in dl:
        logits = abs_bert_summ(src_ids=src_inp_ids,
                               src_pad_mask=src_mask,
                               src_token_type=src_tok_type_ids,
                               # is_freeze_phase1=True,
                               src_cls_pos=src_lis_cls_pos,
                               tgt_ids=tgt_inp_ids,
                               tgt_pad_mask=tgt_mask,
                               tgt_token_type=tgt_tok_type_ids)

        out_prob = torch.softmax(logits, dim=2)
        tgt_one_hot = one_hot(tgt_inp_ids, vocab_size).float()

        loss = None
        for item_id in range(len(out_prob)):
            num_used_token = tgt_mask[item_id].sum()
            used_prob = out_prob[item_id][:num_used_token]
            used_tgt_one_hot = tgt_one_hot[item_id][:num_used_token]
            single_loss = loss_fn(used_prob, used_tgt_one_hot)
            if loss is None:
                loss = single_loss
            else:
                loss += single_loss
        # tensor(0.0267, grad_fn= < DivBackward1 >)
        print(loss)
        print(loss.backward())

        break
