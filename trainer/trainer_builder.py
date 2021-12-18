from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import Callback


class PrintCallback(Callback):
    def on_train_start(self, trainer, pl_module):
        print("Training is started!")

    def on_train_end(self, trainer, pl_module):
        print("Training is done.")


def start_training(abs_bert_summ_model, train_dataloader, val_dataloader):
    trainer = Trainer(accelerator='auto')
    trainer.fit(model=abs_bert_summ_model, train_dataloaders=train_dataloader)
