import os
os.environ["TOKENIZERS_PARALLELISM"] = "True"
from torch import optim, nn, utils, Tensor
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks import ModelCheckpoint
import transformers
from transformers import AutoTokenizer,AutoConfig, AutoModel
from torch.utils.data import DataLoader
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
from Model import Tier
from Dataset import TACREDDataModule
def main():

    parser = ArgumentParser()
    parser = Tier.add_train_args(parser)
    parser = Tier.add_model_structure_args(parser)

    args = parser.parse_args()
    seed_everything(args.seed)
    dataset = TACREDDataModule(model_name_or_path=args.basemodel_name,
                               max_seq_length=args.max_seq_length,
                               train_batch_size= args.batch_size,
                               eval_batch_size=args.batch_size,
                               num_classes=args.num_classes,
                               noisy_rate = args.noisy_rate,
                               data_dir=args.data_dir)
    dataset.setup()
    train_dataset = dataset.train_dataloader()
    dev_dataset = dataset.val_dataloader()
    args.total_steps = int(len(train_dataset) * args.num_train_epochs)
    args.warmup_steps = int(args.total_steps * args.warmup_ratio)

    TierModel = Tier(args)


    # train with both splits


    wandb_logger = WandbLogger(project=args.projetc_name)
    checkpoint_callback=ModelCheckpoint(filepath='./save/checkpoint.ckpt', save_top_k=1, monitor="dev_rev_f1",mode='max',save_last=True)
    trainer = Trainer(gpus=args.device,max_epochs = args.num_train_epochs,logger=wandb_logger,default_root_dir="./save/",checkpoint_callback =checkpoint_callback,early_stop_callback=EarlyStopping(monitor="dev_rev_f1", mode="max"))#enabling checkpoint
    wandb_logger.watch(TierModel)
    print(dataset.length_tokenizer())
    TierModel.resize_token_embeddings(dataset.length_tokenizer())
    print(TierModel.bert.embeddings.word_embeddings.weight.shape)
    trainer.fit(TierModel, train_dataset, dev_dataset)
    TierModel.eval()
    trainer.test(TierModel, test_dataloaders=dataset.test_dataloader(),ckpt_path='best')

if __name__ == '__main__':
    main()