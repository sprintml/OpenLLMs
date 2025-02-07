from enum import Enum
import argparse
import dataclasses
from dataclasses import dataclass, field
from typing import Optional

from transformers import HfArgumentParser, TrainingArguments, TrainingArguments, Seq2SeqTrainingArguments

from tasks.utils import *

import dp_transformers

TASKS = ["docvqa", "samsum", "mitmovie"]

DATASETS = ["docvqa", "samsum", "mitmovie"]

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.

    Using `HfArgumentParser` we can turn this class
    into argparse arguments to be able to specify them on
    the command line.training_args
    """

    task_name: str = field(
        metadata={
            "help": "The name of the task to train on: " + ", ".join(TASKS),
            "choices": TASKS
        },
    )
    dataset_name: str = field(
        default=None,
        metadata={
            "help": "The name of the dataset to use: " + ", ".join(DATASETS),
            "choices": DATASETS
        }
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached preprocessed datasets or not."}
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    prediction_file: Optional[str] = field(
        default="predictions.txt"
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the training data."}
    )
    validation_file: Optional[str] = field(
        default=None, metadata={"help": "A csv or a json file containing the validation data."}
    )
    test_file: Optional[str] = field(
        default=None, 
        metadata={"help": "A csv or a json file containing the test data."}
    )
    template_id: Optional[int] = field(
        default=0,
        metadata={
            "help": "The specific prompt string to use"
        }
    )
    use_split_train: bool = field(
        default=False, metadata={"help": "Use the split training data for half of the training."}
    )
    train_table: bool = field(
        default=False, metadata={"help": "Use only the table as train input"}
    )
    mit_task: Optional[str] = field(
        default="genre", metadata={"help": "which MIT movie task to solve. Either director or genre. Default: genre"}
    )
@dataclass
class ModelArguments:
    """
    Arguments pretaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained Prefix model or model identifier from huggingface.co/models"}
    )
    lm_model_name_or_path: str = field(
        default = "gpt2",
        metadata={"help": "Path to pretrained LMHead model or LMHead model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )
    prefix: bool = field(
        default=True,
        metadata={
            "help": "Will use P-tuning v2 during training"
        }
    )
    prompt: bool = field(
        default=False,
        metadata={
            "help": "Will use prompt tuning during training"
        }
    )
    lora: bool = field(
        default=False,
        metadata={
            "help": "Will use PEFT LoRA during training"
        }
    )
    pre_seq_len: int = field(
        default=4,
        metadata={
            "help": "The length of prompt"
        }
    )
    prefix_projection: bool = field(
        default=True,
        metadata={
            "help": "Apply a two-layer MLP head over the prefix embeddings"
        }
    ) 
    prefix_hidden_size: int = field(
        default=512,
        metadata={
            "help": "The hidden size of the MLP projection head in Prefix Encoder if prefix projection is used"
        }
    )
    hidden_dropout_prob: float = field(
        default=0,
        metadata={
            "help": "The dropout probability used in the models"
        }
    )
    loading_4_bit: bool = field(
        default=False,
        metadata={
            "help": "if the model should be loaded in 4 bit"
        }
    ) 
    constant_scheduler: bool = field(
        default=False,
        metadata={
            "help": "Use the constant scheduler"
        }
    ) 


def get_args():
    """Parse all the args."""
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments, dp_transformers.PrivacyArguments))
    args = parser.parse_args_into_dataclasses()

    return args