import random
from dataclasses import dataclass
from typing import List, Tuple

import torch
import datasets
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer, BatchEncoding, DataCollatorWithPadding


from .arguments import DataArguments
from .trainer import TevatronTrainer

import logging
logger = logging.getLogger(__name__)


class TrainDataset(Dataset):
    def __init__(
            self,
            data_args: DataArguments,
            dataset: datasets.Dataset,
            tokenizer: PreTrainedTokenizer,
            trainer: TevatronTrainer = None,
    ):
        self.train_data = dataset
        self.tok = tokenizer
        self.trainer = trainer

        self.data_args = data_args
        self.total_len = len(self.train_data)

    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __len__(self):
        return self.total_len

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        group = self.train_data[item]
        epoch = int(self.trainer.state.epoch)

        _hashed_seed = hash(item + self.trainer.args.seed)

        qry_id = group['query_id']
        qry = group['query']
        encoded_query = self.create_one_example(qry, is_query=True)

        psg_ids = []
        encoded_passages = []
        group_positives = [(docid, pos) for docid, pos in zip(group['pos_docids'], group['positives'])]
        group_negatives = [(docid, neg) for docid, neg in zip(group['neg_docids'], group['negatives'])]

        if self.data_args.positive_passage_no_shuffle:
            pos_psg = group_positives[0]
        else:
            pos_psg = group_positives[(_hashed_seed + epoch) % len(group_positives)]
        psg_ids.append(pos_psg[0])
        encoded_passages.append(self.create_one_example(pos_psg[1]))

        negative_size = self.data_args.train_n_passages - 1
        if len(group_negatives) < negative_size:
            negs = random.choices(group_negatives, k=negative_size)
        elif self.data_args.train_n_passages == 1:
            negs = []
        elif self.data_args.negative_passage_no_shuffle:
            negs = group_negatives[:negative_size]
        else:
            _offset = epoch * negative_size % len(group_negatives)
            negs = [x for x in group_negatives]
            random.Random(_hashed_seed).shuffle(negs)
            negs = negs * 2
            negs = negs[_offset: _offset + negative_size]

        for neg_psg in negs:
            psg_ids.append(neg_psg[0])
            encoded_passages.append(self.create_one_example(neg_psg[1]))

        return qry_id, psg_ids, encoded_query, encoded_passages


class EncodeDataset(Dataset):
    input_keys = ['text_id', 'text']

    def __init__(self, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer, max_len=128):
        self.encode_data = dataset
        self.tok = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> Tuple[str, BatchEncoding]:
        text_id, text = (self.encode_data[item][f] for f in self.input_keys)
        encoded_text = self.tok.encode_plus(
            text,
            max_length=self.max_len,
            truncation='only_first',
            padding=False,
            return_token_type_ids=False,
        )
        return text_id, encoded_text


class RankDataset(Dataset):
    input_keys =  ['query_id', 'query', 'doc_id', 'doc']

    def __init__(self, data_args: DataArguments, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer):
        self.rank_data = dataset
        self.tok = tokenizer
        self.data_args = data_args
        self.total_len = len(self.rank_data)

    def __len__(self):
        return len(self.rank_data)
    
    def create_one_example(self, text_encoding: List[int], is_query=False):
        item = self.tok.encode_plus(
            text_encoding,
            truncation='only_first',
            max_length=self.data_args.q_max_len if is_query else self.data_args.p_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        data = self.rank_data[item]
        qid = data['query_id']
        did = data['doc_id']
        encoded_query = self.create_one_example(data['query'], is_query=True)
        encoded_passage = self.create_one_example(data['doc'], is_query=False)

        return qid, did, encoded_query, encoded_passage


class RerankDataset(Dataset):
    input_keys =  ['query_id', 'query', 'doc_id', 'doc']

    def __init__(self, data_args: DataArguments, dataset: datasets.Dataset, tokenizer: PreTrainedTokenizer):
        self.rank_data = dataset
        self.tok = tokenizer
        self.data_args = data_args
        self.total_len = len(self.rank_data)

    def __len__(self):
        return len(self.rank_data)
    
    def create_one_example(self, query_encoding: List[int], doc_encoding: List[int]):
        item = self.tok.encode_plus(
            text=query_encoding,
            text_pair=doc_encoding,
            truncation='only_first',
            max_length=self.data_args.seq_max_len,
            add_special_tokens=True,
            padding=False,
            return_attention_mask=True,
            return_token_type_ids=True,
        )
        return item

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding]]:
        data = self.rank_data[item]
        qid = data['query_id']
        did = data['doc_id']
        encoded_qp = self.create_one_example(data['query'], data['doc'])

        return qid, did, encoded_qp
