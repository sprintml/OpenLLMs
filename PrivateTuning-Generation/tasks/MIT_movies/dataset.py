import torch
from torch.utils import data
from datasets.arrow_dataset import Dataset as HFDataset
from datasets import Dataset, DatasetDict
from datasets.load import load_dataset, load_metric
import evaluate
from transformers import (
    DataCollatorForLanguageModeling,
    AutoTokenizer
)
import numpy as np
import logging
from datasets import Dataset


from dp_transformers import DataCollatorForPrivateCausalLanguageModeling

from functools import partial, reduce
import statistics

logger = logging.getLogger(__name__)


class MITMoviesDataset():
    def __init__(self, tokenizer: AutoTokenizer, max_seq_length, pad_to_max_length, disable_dp, mit_task="genre") -> None:
        mit_genre_bool = (mit_task == "genre")
        super().__init__()

        if mit_genre_bool:
            self.raw_datasets = DatasetDict.load_from_disk("./datasets/movie_data/Genre/") 
        else:
            self.raw_datasets = DatasetDict.load_from_disk("./datasets/movie_data/Director/")     

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
        
        self.raw_datasets = self.raw_datasets.map(
            self.preprocess_function_gen if mit_genre_bool else self.preprocess_function_dir,
            batched=True,
            load_from_cache_file= True,
            desc="Running tokenizer on dataset",
        )
        
        print(self.raw_datasets)
        self.raw_datasets = self.raw_datasets.remove_columns(['content', 'label'])


        if disable_dp:
            self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
            print("Normal data collator loaded")
        else:
            self.data_collator = DataCollatorForPrivateCausalLanguageModeling(tokenizer)
            print("DP data collator loaded")

    def preprocess_function_gen(self, examples):
        
        targets = list(self.tokenizer.bos_token + "Give the genre of the movie mentioned in the input. \n\n### Input:\n" + np.array(examples["content"], dtype=object) + " " + self.tokenizer.eos_token + " \n\n### Genre:\n" + np.array(examples["label"], dtype=object) + " " + self.tokenizer.eos_token)
        tokenized_targets = self.tokenizer(targets, max_length=self.max_seq_length, padding=self.padding, truncation=True, return_tensors='pt')
        
        return tokenized_targets
    
    def preprocess_function_dir(self, examples):
        
        targets = list(self.tokenizer.bos_token + "Give the director of the movie mentioned in the input. \n\n### Input:\n" + np.array(examples["content"], dtype=object) + " " + self.tokenizer.eos_token + " \n\n### Director:\n" + np.array(examples["label"], dtype=object) + " " + self.tokenizer.eos_token)
        tokenized_targets = self.tokenizer(targets, max_length=self.max_seq_length, padding=self.padding, truncation=True, return_tensors='pt')
        
        return tokenized_targets


    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels

    
        
    