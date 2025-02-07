import logging
import os
import sys
import numpy as np
from typing import Dict

import torch

import datasets
import transformers
from transformers import set_seed, Trainer
from transformers.trainer_utils import get_last_checkpoint

from arguments import get_args

from tasks.utils import *
from dp_transformers.grad_sample.transformers import conv_1d


os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

def train(trainer, resume_from_checkpoint=None, last_checkpoint=None):
    checkpoint = None
    if resume_from_checkpoint is not None:
        checkpoint = resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    train_result = trainer.train()

    metrics = train_result.metrics

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    

if __name__ == '__main__':
    args = get_args()
    print(args)
    _, data_args, training_args, privacy_args = args
    print(f"Path to save at: {training_args.output_dir}")
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    

    if not os.path.isdir("checkpoints") or not os.path.exists("checkpoints"):
        os.mkdir("checkpoints")
      

    set_seed(training_args.seed)


    if data_args.task_name.lower() == 'docvqa' :
        from tasks.DocVQA.get_trainer import get_trainer

    elif data_args.task_name.lower() == 'samsum' :
        from tasks.SAMSum.get_trainer import get_trainer

    elif data_args.task_name.lower() == 'mitmovie' :
        from tasks.MIT_movies.get_trainer import get_trainer



    trainer, model, tokenizer = get_trainer(args)

    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )


    if training_args.do_train:
        train(trainer, training_args.resume_from_checkpoint, last_checkpoint)

        model.save_pretrained(training_args.output_dir)
        tokenizer.save_pretrained(training_args.output_dir)
        print(f"Model successfully saved at {training_args.output_dir}")
        if not privacy_args.disable_dp:
            eps_prv = trainer.get_prv_epsilon()
            eps_rdp = trainer.get_rdp_epsilon()
            print(f"final_epsilon_prv: {eps_prv}")
            print(f"final_epsilon_rdp: {eps_rdp}")

