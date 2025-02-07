import logging

import math
import torch

from transformers import (
    Trainer,
    TrainerCallback,
    get_constant_schedule,
    AutoConfig, 
    AutoTokenizer, 
    Seq2SeqTrainer
)

from model.utils import get_model
from tasks.SAMSum.dataset import SAMSumDataset
from tasks.utils import FixedOpacusDPSeq2SeqTrainer, FixedOpacusDPTrainer

import dp_transformers
from dp_transformers.grad_sample.transformers import conv_1d


class PPLCallback(TrainerCallback):
    def on_log(self, args, state, control, **kwargs):
        try:
            print(f"ppl: {math.exp(state.log_history[-1]['loss'])}")
        except Exception:
            pass

        try:
            print(f"eval_ppl: {math.exp(state.log_history[-1]['eval_loss'])}")
        except Exception:
            pass


logger = logging.getLogger(__name__)


def get_trainer(args):
    model_args, data_args, training_args, privacy_args = args
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        use_fast=True,
        revision=model_args.model_revision,
    )

    use_bart = ("bart" in model_args.model_name_or_path)

    model = get_model(model_args, use_bart)

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    adam_optim = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    constant_scheduler = get_constant_schedule(adam_optim)



    dataset = SAMSumDataset(tokenizer, data_args.max_seq_length, data_args.pad_to_max_length, privacy_args.disable_dp, use_bart)
    print(dataset.raw_datasets.column_names)
    print(dataset.raw_datasets)
    if use_bart and privacy_args.disable_dp:
        trainer = Seq2SeqTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset.raw_datasets["train"] if training_args.do_train else None,
            eval_dataset= dataset.raw_datasets["validation"] if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            callbacks=[PPLCallback],
            optimizers=(adam_optim, constant_scheduler) if model_args.constant_scheduler else (None, None)
        )
    elif privacy_args.disable_dp:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset.raw_datasets["train"] if training_args.do_train else None,
            eval_dataset= dataset.raw_datasets["validation"] if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            callbacks=[PPLCallback],
            optimizers=(adam_optim, constant_scheduler) if model_args.constant_scheduler else (None, None)
        )
    elif use_bart:
        print("DP trainer chosen")
        print(f"Epsilon: {privacy_args.target_epsilon}")
        privacy_args.target_delta = 1/len(dataset.raw_datasets["train"])
        print(len(dataset.raw_datasets["train"]))
        print(f"Delta: {privacy_args.target_delta}")
        print("Sequence to Sequence Model Found")
        trainer = FixedOpacusDPSeq2SeqTrainer(
            model=model,
            args=training_args, 
            privacy_args=privacy_args,
            train_dataset=dataset.raw_datasets["train"] if training_args.do_train else None,
            eval_dataset= dataset.raw_datasets["validation"] if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            optimizers=(adam_optim, constant_scheduler) if model_args.constant_scheduler else (None, None)
        )
    
    else:
        print("DP trainer chosen")
        print(f"Epsilon: {privacy_args.target_epsilon}")
        privacy_args.target_delta = 1/len(dataset.raw_datasets["train"])
        print(len(dataset.raw_datasets["train"]))
        print(f"Delta: {privacy_args.target_delta}")
        trainer = FixedOpacusDPTrainer(
            model=model,
            args=training_args, 
            privacy_args=privacy_args,
            train_dataset=dataset.raw_datasets["train"] if training_args.do_train else None,
            eval_dataset= dataset.raw_datasets["validation"] if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            optimizers=(adam_optim, constant_scheduler) if model_args.constant_scheduler else (None, None)
    )

    return trainer, model, tokenizer