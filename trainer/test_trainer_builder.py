from torch.utils.data import DataLoader

from inp_dataloader.summ_dataset import SummDataset
from model_builder.abs_bert_summ import AbsBertSumm
from model_builder.abs_bert_summ_pylight import AbsBertSummPylight
from trainer.trainer_builder import start_training

if __name__ == '__main__':
    train_dataset = SummDataset('/Users/LongNH/Workspace/presumm-vn/json_data/json_data.train.json')
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

    vocab_size = train_dataset.tokenizer.phobert_tokenizer.vocab_size
    abs_bert_summ = AbsBertSumm(vocab_size=vocab_size)
    abs_bert_summ_pylight = AbsBertSummPylight(model=abs_bert_summ, vocab_size=vocab_size)

    val_dataset = SummDataset('/Users/LongNH/Workspace/presumm-vn/json_data/json_data.train.json')
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=True)

    start_training(abs_bert_summ_model=abs_bert_summ_pylight, train_dataloader=train_dataloader,
                   val_dataloader=val_dataloader)

