import torch
from torch.utils import data
from datasets.arrow_dataset import Dataset as HFDataset
from datasets import Dataset
from datasets.load import load_dataset, load_metric
import evaluate
from transformers import (
    DataCollatorForLanguageModeling,
    AutoTokenizer, 
    DataCollatorForSeq2Seq,
    PreTrainedTokenizer,
    BartForCausalLM
)
import numpy as np
import logging
from datasets import Dataset

from typing import Any, Callable, List, Optional, Union, Dict, Sequence


from dp_transformers import DataCollatorForPrivateCausalLanguageModeling

from functools import partial, reduce
import statistics

logger = logging.getLogger(__name__)



class SAMSumDataset():
    def __init__(self, tokenizer: AutoTokenizer, max_seq_length, pad_to_max_length, disable_dp, use_bart=False) -> None:
        super().__init__()
        self.raw_datasets = load_dataset("samsum", trust_remote_code=True)
        
        self.tokenizer = tokenizer

        # Padding strategy
        if pad_to_max_length:
            self.padding = "max_length"
            print("Max padding chosen")
        else:
            self.padding = False
            print("No padding chosen")


        if max_seq_length > tokenizer.model_max_length:
            logger.warning(
                f"The max_seq_length passed ({max_seq_length}) is larger than the maximum length for the"
                f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
            )
        self.max_seq_length = min(max_seq_length, tokenizer.model_max_length)
        

        if use_bart:
            self.raw_datasets = self.raw_datasets.map(
                self.preprocess_seq2seq_function,
                batched=True,
                load_from_cache_file= False,
                desc="Running tokenizer on dataset",
            )
        else:
            self.raw_datasets = self.raw_datasets.map(
                self.preprocess_function,
                batched=True,
                load_from_cache_file= True,
                desc="Running tokenizer on dataset",
            )
        
        print(self.raw_datasets)
        self.raw_datasets = self.raw_datasets.remove_columns(['dialogue', 'summary', 'id'])


        if disable_dp:
            if use_bart:
                self.data_collator = DataCollatorForSeq2Seq(tokenizer, model="facebook/bart-base")
                print("Bert data collator loaded")    
            else:
                self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
                print("Normal data collator loaded")
        else:
            if use_bart:
                self.data_collator = DataCollatorForSeq2Seq(tokenizer, model="facebook/bart-base")
                print("Seq2Seq DP data collator loaded")
                pass
            else:
                self.data_collator = DataCollatorForPrivateCausalLanguageModeling(tokenizer)
                print("DP data collator loaded")


    def preprocess_function(self, examples):
        targets = list(self.tokenizer.bos_token + "Summarize the following conversation. \n\n### Input:\n" + np.array(examples["dialogue"], dtype=object) + " \n\nSummary:\n" + np.array(examples["summary"], dtype=object) + self.tokenizer.eos_token)
        tokenized_targets = self.tokenizer(targets, max_length=self.max_seq_length, padding=self.padding, truncation=True, return_tensors='pt')
        return tokenized_targets


    def preprocess_seq2seq_function(self, examples):
        inputs = list("Summarize: " + np.array(examples["dialogue"], dtype=object))
        labels = examples["summary"]

        model_inputs = self.tokenizer(inputs, max_length=self.max_seq_length, padding=True, truncation=True)

        labels = self.tokenizer(labels, max_length=128, padding=True, truncation=True)

        model_inputs["labels"] = labels["input_ids"]

        return model_inputs

    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    
        
    