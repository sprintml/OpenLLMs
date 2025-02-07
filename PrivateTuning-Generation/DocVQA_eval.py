import argparse
import os

import torch

from datasets.load import load_dataset

import numpy as np
from datasets import Dataset, DatasetDict

import peft

from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

import tqdm

import nltk
import evaluate

from Levenshtein import ratio
import statistics


parser = argparse.ArgumentParser()


parser.add_argument('--eval_len', type=int, default=100,
                    help='amount of tokens that are generated')

parser.add_argument('--amount_test', type=int, default=1000,
                    help='Amount of test data used for evaluation')

parser.add_argument('--init_checkpoint', default=None, type=str, help='initial checkpoint')

parser.add_argument('--verbose', type=bool, default=False, 
                    help='If the predictions should be printed in the command line')

parser.add_argument('--load_4_bit', type=bool, default=True, help='loading the model in 4 bits')

parser.add_argument('--seed' , type=int, default=11, help='hf seed')

parser.add_argument('--ft', type=bool, default=False, help='use full fine tuning')

def print_args(args):
    print('=' * 100)
    for k, v in args.__dict__.items():
        print('        - {} : {}'.format(k, v))
    print('=' * 100)


def func_evaluate(model, tokenized_inputs, tokenizer, args):
    nltk.download('punkt')
    model.eval()
    dataset_len = len(tokenized_inputs)
    all_predictions = []

    with torch.no_grad():
        for data in tqdm.tqdm(tokenized_inputs):
            data = data.to("cuda")
            generated = model.generate(input_ids=data["input_ids"], attention_mask=data["attention_mask"], max_new_tokens=args.eval_len)
            input_len = len(data["input_ids"][0])
            decoded_preds = tokenizer.decode(generated[0][(input_len):], skip_special_tokens=True)
            all_predictions.append("\n".join(nltk.sent_tokenize(decoded_preds.strip())))

    return all_predictions


if __name__ == '__main__':
    args = parser.parse_args()
    print_args(args)
    raw_dataset = DatasetDict.load_from_disk("./datasets/DocVQA/")
    tokenizer = AutoTokenizer.from_pretrained(args.init_checkpoint)
    raw_dataset = raw_dataset["validation"].shuffle(seed=args.seed).select(range(args.amount_test))

    inputs = []
    tokenized_inputs = []
    targets = []
        
    for idx, ex in enumerate(raw_dataset):
        tokenized_inputs.append(tokenizer(tokenizer.bos_token + " Answer the question given the following context. \n\n### Context:\n" + np.array(ex["contexts"], dtype=object) + " " + tokenizer.eos_token + " \n\nQuestion:\n" + np.array(ex["questions"], dtype=object) + " " + tokenizer.eos_token + " \n\nAnswer:\n", return_tensors='pt'))

    bnb_config_4 = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )

    if args.ft:
        model = AutoModelForCausalLM.from_pretrained(args.init_checkpoint)
        lm_net = model.to("cuda")

    else:
        config = peft.PeftConfig.from_pretrained(args.init_checkpoint)
        model = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path, quantization_config=bnb_config_4 if args.load_4_bit else None)
        model.resize_token_embeddings(len(tokenizer))
        model = peft.PeftModel.from_pretrained(model, args.init_checkpoint)
        lm_net = model.to("cuda")
        
        lm_net.print_trainable_parameters()
    


    print('model sampling ...')
    print(len(tokenized_inputs))
    predictions = func_evaluate(lm_net, tokenized_inputs, tokenizer, args)

    
    rouge = evaluate.load('rouge')
    results = rouge.compute(predictions=predictions, references=raw_dataset["answers"])
    print(results)

    bleu = evaluate.load('sacrebleu')
    results = bleu.compute(predictions=predictions, references=raw_dataset["answers"])
    print(results)

    levenshtein_ratio = []
    for (p, t) in zip(predictions, raw_dataset["answers"]):
        levenshtein_ratio.append(ratio(p, t))

    print(f"Levenshtein ratio: {statistics.mean(levenshtein_ratio)}")


    pred_file = os.path.join("evaluation/DocVQA", args.prediction_file) 
    with open(pred_file, 'w', encoding='utf8') as writer:
        for _i in range(len(predictions)):
            writer.write(predictions[_i] + '\n')
    print(f"{len(predictions)} predictions written to {pred_file}")