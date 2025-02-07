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


class DocVQADataset():
    def __init__(self, tokenizer: AutoTokenizer, max_seq_length, pad_to_max_length, disable_dp, seed) -> None:
        super().__init__()

        self.raw_datasets = DatasetDict.load_from_disk("./datasets/DocVQA/")
        self.raw_datasets["validation"] = self.raw_datasets["validation"].shuffle(seed).select(range(2500))
        
        self.tokenizer = tokenizer

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
            self.preprocess_function,
            batched=True,
            load_from_cache_file= True,
            desc="Running tokenizer on dataset",
        )
        
        print(self.raw_datasets)
        self.raw_datasets = self.raw_datasets.remove_columns(['contexts', 'questions', 'answers', 'question_id'])


        if disable_dp:
            self.data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)
            print("Normal data collator loaded")
        else:
            self.data_collator = DataCollatorForPrivateCausalLanguageModeling(tokenizer)
            print("DP data collator loaded")

    def preprocess_function(self, examples):
        targets = list(self.tokenizer.bos_token + " Answer the question given the following context. \n\n### Context:\n" + np.array(examples["contexts"], dtype=object) + " " + self.tokenizer.eos_token + " \n\nQuestion:\n" + np.array(examples["questions"], dtype=object) + " " + self.tokenizer.eos_token + " \n\nAnswer:\n" + np.array(examples["answers"], dtype=object) + " " + self.tokenizer.eos_token)
        tokenized_targets = self.tokenizer(targets, max_length=self.max_seq_length, padding=self.padding, truncation=True, return_tensors='pt')
        
        return tokenized_targets


    def postprocess_text(self, preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [[label.strip()] for label in labels]

        return preds, labels
    

    def preprocess_logits_for_metrics(self, logits, labels):
        # https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/11
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """

        pred_ids = torch.argmax(logits, dim=-1)
        return pred_ids, labels
    
    def private_preprocess_logits_for_metrics(self, logits, labels):
        # https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/11
        """
        Original Trainer may have a memory leak. 
        This is a workaround to avoid storing too many tensors that are not needed.
        """
        pred_ids = torch.argmax(logits[0], dim=-1)
        return pred_ids, labels

    
        
    