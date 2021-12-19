from torch.utils.data import DataLoader

from inp_dataloader.summ_dataset import SummDataset
from model_builder.abs_bert_summ import AbsBertSumm
from model_builder.abs_bert_summ_pylight import AbsBertSummPylight
from trainer.trainer_builder import start_training
import argparse


def args_parser():
    # Construct the argument parser
    ap = argparse.ArgumentParser()

    # Add the arguments to the parser
    ap.add_argument('-jsondat', '--json_data', required=True, help='Input in json format')
    ap.add_argument('-gpus', '--gpus', required=False, help='Specify gpus device')
    args = vars(ap.parse_args())

    return args


if __name__ == '__main__':
    cmd_args = args_parser()

    train_dataset = SummDataset(cmd_args.get('json_data'))
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True)

    vocab_size = train_dataset.tokenizer.phobert_tokenizer.vocab_size
    abs_bert_summ = AbsBertSumm(vocab_size=vocab_size)
    abs_bert_summ_pylight = AbsBertSummPylight(model=abs_bert_summ, vocab_size=vocab_size)

    val_dataset = SummDataset(cmd_args.get('json_data'))
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=8, shuffle=True)

    start_training(abs_bert_summ_model=abs_bert_summ_pylight, train_dataloader=train_dataloader,
                   val_dataloader=val_dataloader, gpus=cmd_args.get('gpus'))
