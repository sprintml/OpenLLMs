import logging

import math
import torch

from transformers import (
    GPT2Config,
    GPT2Tokenizer,
    Trainer,
    GPT2LMHeadModel,
    TrainerCallback,
    get_constant_schedule,
    AutoConfig, 
    AutoTokenizer, 
    GPTNeoXForCausalLM
)

from model.utils import get_model
from tasks.DocVQA.dataset import DocVQADataset

import dp_transformers
from dp_transformers.grad_sample.transformers import conv_1d

import inspect



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
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
    )
    
    model = get_model(model_args)

    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    model.resize_token_embeddings(len(tokenizer))

    adam_optim = torch.optim.AdamW(model.parameters(), lr=training_args.learning_rate)
    constant_scheduler = get_constant_schedule(adam_optim)

    dataset = DocVQADataset(tokenizer, data_args.max_seq_length, data_args.pad_to_max_length, privacy_args.disable_dp, training_args.seed)
    print(dataset.raw_datasets.column_names)
    print(dataset.raw_datasets)

    if privacy_args.disable_dp:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset.raw_datasets["train"] if training_args.do_train else None,
            eval_dataset= dataset.raw_datasets["validation"] if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            callbacks=[PPLCallback],
            optimizers=(adam_optim, constant_scheduler) if model_args.constant_scheduler else (None, None),
            preprocess_logits_for_metrics=dataset.preprocess_logits_for_metrics,
        )
    else:
        print("DP trainer chosen")
        print(f"Epsilon: {privacy_args.target_epsilon}")
        privacy_args.target_delta = 1/len(dataset.raw_datasets["train"])
        print(len(dataset.raw_datasets["train"]))
        print(f"Delta: {privacy_args.target_delta}")
        trainer = dp_transformers.dp_utils.OpacusDPTrainer(
            model=model,
            args=training_args, 
            privacy_args=privacy_args,
            train_dataset=dataset.raw_datasets["train"] if training_args.do_train else None,
            eval_dataset= dataset.raw_datasets["validation"] if training_args.do_eval else None,
            tokenizer=tokenizer,
            data_collator=dataset.data_collator,
            optimizers=(adam_optim, constant_scheduler) if model_args.constant_scheduler else (None, None),
            preprocess_logits_for_metrics=dataset.private_preprocess_logits_for_metrics,
    )

    return trainer, model, tokenizer