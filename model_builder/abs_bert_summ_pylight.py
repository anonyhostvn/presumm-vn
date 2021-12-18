import torch
from torch.nn.functional import one_hot
from pytorch_lightning import LightningModule
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import CrossEntropyLoss


class AbsBertSummPylight(LightningModule):

    def __init__(self, model, vocab_size
                 , learning_rate: float = 2e-5
                 , adam_epsilon: float = 1e-8
                 , warmup_steps: int = 0
                 , weight_decay: float = 0.0
                 , **kwargs):
        super().__init__()
        self.save_hyperparameters()
        self.total_steps = 10
        self.model = model
        self.vocab_size = vocab_size
        self.loss_fn = CrossEntropyLoss()

    def forward(self, src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask
                , tgt_inp_ids, tgt_tok_type_ids, tgt_lis_cls_pos, tgt_mask):
        return self.model(src_ids=src_inp_ids, src_pad_mask=src_mask, src_token_type=src_tok_type_ids
                          , is_freeze_phase1=True, src_cls_pos=src_lis_cls_pos
                          , tgt_ids=tgt_inp_ids, tgt_pad_mask=tgt_mask, tgt_token_type=tgt_tok_type_ids)

    def training_step(self, batch, batch_idx):
        """
        :param batch: input dataloader (src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask
                        , tgt_inp_ids, tgt_tok_type_ids, tgt_lis_cls_pos, tgt_mask)
        :param batch_idx: Số thứ tự của batch
        :return:
        """
        src_inp_ids, src_tok_type_ids, src_lis_cls_pos, src_mask \
            , tgt_inp_ids, tgt_tok_type_ids, tgt_lis_cls_pos, tgt_mask = batch
        logits = self.model(src_ids=src_inp_ids, src_pad_mask=src_mask, src_token_type=src_tok_type_ids
                            , is_freeze_phase1=True, src_cls_pos=src_lis_cls_pos
                            , tgt_ids=tgt_inp_ids, tgt_pad_mask=tgt_mask, tgt_token_type=tgt_tok_type_ids)
        out_prob = torch.softmax(logits, dim=2)
        tgt_one_hot = one_hot(tgt_inp_ids, num_classes=self.vocab_size)

        loss = None
        for item_id in range(len(out_prob)):
            num_used_token = tgt_mask[item_id].sum()
            used_prob = out_prob[item_id][:num_used_token]
            used_tgt_one_hot = tgt_one_hot[item_id][:num_used_token]
            single_loss = self.loss_fn(used_prob, used_tgt_one_hot)
            if loss is None:
                loss = single_loss
            else:
                loss += single_loss
        return loss

    def configure_optimizers(self):
        """
        Chuẩn bị Optimizer với chiến lược warm-up và weight-decay
        """
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
