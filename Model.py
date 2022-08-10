import os
import torch
from torch import optim, nn, utils, Tensor
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
import transformers
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import (
    AutoModel,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
from torch.optim import AdamW
import wandb
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader
from argparse import ArgumentParser
import torchmetrics
import random

class Tier(LightningModule):
    def __init__(self,args):
        super().__init__()
        self.save_hyperparameters()
        self.args=args

        self.temperature = args.temperature
        self.alpha = args.alpha
        self.beta = args.beta

        self.num_classes = args.num_classes
        self.n_group = args.n_group
        self.group_size = args.group_size
        self.n_model = args.n_group * args.group_size
        self.div = [args.batch_size//args.n_group* i for i in range(0,args.n_group)]
        self.div.append(-1)

        self.bert = AutoModel.from_pretrained("bert-base-cased")
        self.classifiers = nn.ModuleList([
            nn.Linear(self.bert.config.hidden_size, self.num_classes) for i in range(self.n_model)])

        self.loss_fnt = nn.CrossEntropyLoss(reduction='mean')
        self.loss_kl = nn.KLDivLoss(reduction='batchmean')

        self.F1Score = torchmetrics.F1Score(num_classes=args.num_classes,average='micro')

    @staticmethod
    def add_model_structure_args(parent_parser):
        parser = parent_parser.add_argument_group("TierModel")
        parser.add_argument("--group_size", type=int, default=4)
        parser.add_argument("--n_group", type=int, default=4)
        parser.add_argument("--warmup_ratio", default=0.06, type=float)
        parser.add_argument("--seed", type=int, default=22)
        parser.add_argument("--num_class", type=int, default=42)
        parser.add_argument("--dropout_prob", type=float, default=0.1)
        parser.add_argument("--project_name", type=str, default="pyltest")
        parser.add_argument("--evaluation_method", default="min_ie", type=str)
        parser.add_argument("--alpha", type=float, default=5.0)
        parser.add_argument("--beta", type=float, default=-0.001)
        parser.add_argument("--alpha_warmup_ratio", default=0.1, type=float)
        parser.add_argument("--temperature", type=float, default=0.3)

        return parent_parser

    @staticmethod
    def add_train_args(parent_parser):
        parser = parent_parser.add_argument_group("TrainArgs")
        parser.add_argument("--basemodel_name", type=str, default="bert-base-cased")
        parser.add_argument("--projetc_name", type=str, default="pyltest")
        parser.add_argument("--data_dir", type=str, default="./data/")
        parser.add_argument("--num_train_epochs", default=5, type=int)
        parser.add_argument("--device", default='3', type=str)
        parser.add_argument("--num_classes", default=42, type=int)
        parser.add_argument("--batch_size", default=64, type=int)
        parser.add_argument("--max_seq_length", default=512, type=int)
        parser.add_argument("--learning_rate", default=6e-5, type=float)
        parser.add_argument("--beta1", type=float, default=0.8)
        parser.add_argument("--beta2", type=float, default=0.98)
        parser.add_argument("--weight_decay", type=float, default=0.8)
        parser.add_argument("--decay_step", type=float, default=1)
        parser.add_argument("--eps", type=float, default=1e-8)
        parser.add_argument("--noisy_rate", type=int, default=0)
        return parent_parser

    @staticmethod
    def info_entropy(p):
        return torch.mul(p, -p.log()).sum(-1)

    def forward(self, input_ids, attention_mask,entity_ids=None,entity_attention_mask=None,entity_position_ids=None, labels=None):
        h = self.bert(input_ids=input_ids,
                    attention_mask=attention_mask,
                    return_dict=False)[0]
        h_cls = h[:, 0]#get cls, batch_size * hidden_size
        #
        logits = []
        for i in range(self.n_model):
            logit= self.classifiers[i](h_cls)
            logits.append(logit)
        logit = sum(logits)

        return logit

    def training_step(self, batch, batch_idx):
        label = batch['labels']
        batch.pop('labels')
        h = self.bert(**batch,return_dict=False)[0]
        h_cls = h[:, 0]  # get cls, batch_size * hidden_size
        #
        shift = random.randint(0,self.n_model)
        index = -shift
        loss_list = []
        for i in range(self.n_group):
            if self.div[i + 1] != -1:
                h_c = h_cls[self.div[i]:self.div[i + 1]]
                group_label = label[self.div[i]:self.div[i + 1]]
            else:
                h_c = h_cls[self.div[i]:]
                group_label = label[self.div[i]:]
            grouplogits = []
            for j in range(self.group_size):
                logit = self.classifiers[index](h_c)
                logit = logit / self.temperature
                index += 1
                grouplogits.append(logit)
            groupprobs = [F.softmax(logit, dim=-1) for logit in grouplogits]
            softtarget = torch.stack(groupprobs, dim=0).mean(0)
            task_loss = sum([self.loss_fnt(logit, group_label) for logit in grouplogits]) / self.group_size
            cr_loss = sum([self.loss_kl(prob, softtarget) for prob in groupprobs]) / self.group_size
            ier_loss = sum([self.info_entropy(prob) for prob in groupprobs]) / self.group_size
            ier_loss = ier_loss.mean()
            loss_list.append(task_loss + self.alpha * cr_loss + self.beta * ier_loss)
        loss = sum([l for l in loss_list]) / self.n_group
        return {'loss':loss,'log':{'loss':loss.item()}}


    def test_step(self, batch, batch_idx):
        labels = batch['labels']
        batch.pop('labels')
        logits = self.forward(**batch)
        preds = torch.argmax(logits, -1)
        return {"preds": preds, "labels": labels}


    def validation_step(self, batch, batch_idx):
        labels = batch['labels']
        batch.pop('labels')
        logits = self.forward(**batch)
        preds = torch.argmax(logits, -1)
        return {"preds": preds, "labels": labels}

    def validation_epoch_end(self, outputs):

        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])

        f1 = self.F1Score(target=labels,preds= preds)
        return {'log':{"dev_rev_f1":f1}}

    def test_epoch_end(self, outputs):

        preds = torch.cat([x["preds"] for x in outputs])
        labels = torch.cat([x["labels"] for x in outputs])

        f1 = self.F1Score(target=labels,preds= preds)
        return {'log':{"test_rev_f1": f1}}
    def configure_optimizers(self):
        """Prepare optimizer and schedule (linear warmup and decay)"""

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [p for n, p in self.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.eps)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.args.warmup_steps,
            num_training_steps=self.args.total_steps,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]

    def resize_token_embeddings(self, n):
        self.bert.resize_token_embeddings(n)
