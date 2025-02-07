import json
import os
from data_utils import load_dataset_hf, limit_tokens
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets, DatasetDict
import argparse
import numpy as np
from utils import random_sampling
from utils import *

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--save_file', type=str)
    parser.add_argument('--model', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    parser.add_argument('--seed', type=int, required=True,
                        help='seeding for the test samples selection. Must match seeding from teacher classification.')
    parser.add_argument('--num_tokens_to_predict', type=int, required=True,
                        help='Number of token the model should predict')
    parser.add_argument('--api_num_log_prob', dest='api_num_log_prob', action='store', required=False, type=int,
                        default=100,
                        help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')
    parser.add_argument('--num_student_query', required=True, type=int, help='Number of unlabeled student queries')

    args = parser.parse_args()
    args = vars(args) #To avoid conflit with chunck size function


    if 'docvqa' in args['dataset']:
        args['prompt_prefix'] = "Answer the question based on the context.\n\n"
        args["q_prefix"] = "Question: "
        args["a_prefix"] = "\nAnswer: "
        dataset = DatasetDict.load_from_disk("./datasets/DocVQA/hf_format_datasets/DocVQA_client0123")
    elif 'mit-d' in args['dataset']:
        args['prompt_prefix'] = "" #"Based on the provided movie description or plot in the 'Content' field, your task is to fill in the 'Label' field with the name of the director of the movie. Your prediction should match with one of the directors represented in the 'content' field.\n\n"
        args["q_prefix"] = "Sentence: "
        args["a_prefix"] = "\nDirector: "
        dataset = DatasetDict.load_from_disk("./datasets/movie_data/Director/hf_dataset")

    elif 'mit-g' in args['dataset']:
        args['prompt_prefix'] = "" #"Given a brief summary or plot of a movie in the 'Content' field, your task is to fill in the 'Label' field with the genre of the movie. Your prediction should align with one of the genres represented in the 'content' field.\n\n"
        args["q_prefix"] = "Sentence: "
        args["a_prefix"] = "\nGenre: "
        dataset = DatasetDict.load_from_disk("./datasets/movie_data/Genre/hf_dataset")

    elif 'samsum' in args['dataset']:
        args['prompt_prefix'] = "Summarize the dialogue.\n\n"
        args["q_prefix"] = "Dialogue: "
        args["a_prefix"] = "\nSummary: "
        dataset = load_dataset(args["dataset"])

    args['task_format'] = "summarization"
    args['prediction_mode'] = "zeroshot" #To show that this is 'zeroshot' with no formatting if 'None', it means it's zeroshot but has its format already


    if args["dataset"] == "samsum":
        np.random.seed(args["seed"])
        test_sentences, test_labels, _ = random_sampling(dataset['validation']['dialogue'],
                                                         dataset['validation']['summary'], args["num_student_query"], [])
        # print("test_sentences", test_sentences)
    elif args["dataset"] == "mit-d" or args["dataset"] == "mit-g":
        np.random.seed(args["seed"])
        test_sentences, test_labels, _ = random_sampling(dataset['validation']['content'], dataset['validation']['label'], args["num_student_query"], [])
        # print("test_sentences", test_sentences)

    elif args["dataset"] == "docvqa":
        np.random.seed(args["seed"])
        # Reduce the example context and query context to 800
        dataset['validation'] = dataset['validation'].map(limit_tokens)
        # Limit the number of tokens in the context
        # dataset['validation']['contexts'] = [' '.join(tokenizer.tokenize(c)[:800]) for c in dataset['validation']['contexts']]
        test_sentences, test_labels, _ = random_sampling([f'{q} \nContext: {c}' for q, c in zip(dataset['validation']['questions'], dataset['validation']['contexts'])],
                                                         dataset['validation']['answers'], args["num_student_query"], [])

        # print("test_sentences", test_sentences)

    else:
        np.random.seed(args["seed"])
        test_sentences, test_labels, _ = random_sampling(dataset['test']['sentence'], dataset['test']['label'], 300, [])


    raw_resp_test = get_model_response(args, None, None, None, override_prompt=test_sentences)  # Get model response

    # print("raw_resp_test", raw_resp_test)

    with open(args['save_file'] +"_"+str(args["seed"]), 'w') as file:
        # print("raw_resp_test", raw_resp_test)  # Note that this will be list of dicts
        for dict_item in raw_resp_test:
            file.write(json.dumps(dict_item) + '\n')

