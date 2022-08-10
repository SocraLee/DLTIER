import pytorch_lightning as pl
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)
import os
import json
from tqdm import tqdm
from torch.utils.data import DataLoader
import torch
class TACREDDataModule():

    loader_columns = [
        "datasets_idx",
        "input_ids",
        "token_type_ids",
        "attention_mask",
        "start_positions",
        "end_positions",
        "labels",
    ]

    def __init__(
        self,
        model_name_or_path: str,
        max_seq_length: int = 512,
        train_batch_size: int = 64,
        eval_batch_size: int = 64,
        num_classes:int = 42,
        noisy_rate = 0,
        data_dir = './data/'
    ):
        super().__init__()
        self.LABEL_TO_ID = {'no_relation': 0, 'per:title': 1, 'org:top_members/employees': 2, 'per:employee_of': 3,
                            'org:alternate_names': 4, 'org:country_of_headquarters': 5, 'per:countries_of_residence': 6,
                            'org:city_of_headquarters': 7, 'per:cities_of_residence': 8, 'per:age': 9,
                            'per:stateorprovinces_of_residence': 10, 'per:origin': 11, 'org:subsidiaries': 12,
                            'org:parents': 13, 'per:spouse': 14, 'org:stateorprovince_of_headquarters': 15,
                            'per:children': 16, 'per:other_family': 17, 'per:alternate_names': 18, 'org:members': 19,
                            'per:siblings': 20, 'per:schools_attended': 21, 'per:parents': 22, 'per:date_of_death': 23,
                            'org:member_of': 24, 'org:founded_by': 25, 'org:website': 26, 'per:cause_of_death': 27,
                            'org:political/religious_affiliation': 28, 'org:founded': 29, 'per:city_of_death': 30,
                            'org:shareholders': 31, 'org:number_of_employees/members': 32, 'per:date_of_birth': 33,
                            'per:city_of_birth': 34, 'per:charges': 35, 'per:stateorprovince_of_death': 36,
                            'per:religion': 37, 'per:stateorprovince_of_birth': 38, 'per:country_of_birth': 39,
                            'org:dissolved': 40, 'per:country_of_death': 41}
        self.model_name_or_path = model_name_or_path
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_classes = num_classes
        self.noisy_rate = noisy_rate
        self.data_dir=data_dir
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        self.new_tokens = []
    # def prepare_data(self):
    #     AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self):
        if self.noisy_rate == 0:
            train_file = os.path.join(self.data_dir, "train.json")
        else:
            train_file = os.path.join(self.data_dir, "noisytrain" + str(self.noisy_rate) + ".json")
        dev_file = os.path.join(self.data_dir, "dev.json")
        test_file = os.path.join(self.data_dir, "test.json")
        dev_rev_file = os.path.join(self.data_dir, "dev_rev.json")
        test_rev_file = os.path.join(self.data_dir, "test_rev.json")
        self.train_features = self.read(train_file)
        self.dev_features = self.read(dev_file)
        self.test_features = self.read(test_file)
        self.dev_rev_features = self.read(dev_rev_file)
        self.test_rev_features = self.read(test_rev_file)



    def tokenize(self, tokens, ss, se, os, oe, subj_type, obj_type):
        sents = []
        subj_type = '[SUBJ-{}]'.format(subj_type)
        obj_type = '[OBJ-{}]'.format(obj_type)
        for token in (subj_type, obj_type):
            if token not in self.new_tokens:
                self.new_tokens.append(token)
                self.tokenizer.add_tokens([token])

        for i_t, token in enumerate(tokens):
            tokens_wordpiece = self.tokenizer.tokenize(token)
            if ss <= i_t <= se or os <= i_t <= oe:
                tokens_wordpiece = []
                if i_t == ss:
                    tokens_wordpiece = [subj_type]
                if i_t == os:
                    tokens_wordpiece = [obj_type]
            sents.extend(tokens_wordpiece)
        sents = sents[:self.max_seq_length - 2]
        input_ids = self.tokenizer.convert_tokens_to_ids(sents)
        input_ids = self.tokenizer.build_inputs_with_special_tokens(input_ids)
        return input_ids
    @staticmethod
    def collate_fn(batch):
        max_len = max([len(f["input_ids"]) for f in batch])
        input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
        attention_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
        labels = [f["labels"] for f in batch]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        attention_mask = torch.tensor(attention_mask, dtype=torch.float)
        labels = torch.tensor(labels, dtype=torch.long)
        output = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }
        return output

    @staticmethod
    def convert_token(token):
        if (token.lower() == '-lrb-'):
            return '('
        elif (token.lower() == '-rrb-'):
            return ')'
        elif (token.lower() == '-lsb-'):
            return '['
        elif (token.lower() == '-rsb-'):
            return ']'
        elif (token.lower() == '-lcb-'):
            return '{'
        elif (token.lower() == '-rcb-'):
            return '}'
        return token

    def read(self, file_in):
        features = []
        with open(file_in, "r") as fh:
            data = json.load(fh)
        print('data processing')
        for d in tqdm(data):
            ss, se = d['subj_start'], d['subj_end']
            os, oe = d['obj_start'], d['obj_end']

            tokens = d['token']
            tokens = [self.convert_token(token) for token in tokens]
            input_ids = self.tokenize(tokens, ss, se, os, oe, d['subj_type'], d['obj_type'])
            rel = self.LABEL_TO_ID[d['relation']]
            feature = {
                'input_ids': input_ids,
                'labels': rel,
            }
            features.append(feature)
        return features

    def train_dataloader(self):
        return DataLoader(self.train_features, num_workers=64,batch_size=self.train_batch_size, collate_fn = self.collate_fn,shuffle=True)

    def val_dataloader(self):
        return  DataLoader(self.dev_rev_features, num_workers=64,pin_memory=True,batch_size=self.eval_batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return  DataLoader(self.test_rev_features, num_workers=64,pin_memory=True,batch_size=self.eval_batch_size, collate_fn=self.collate_fn)


    def length_tokenizer(self):
        return len(self.tokenizer)
