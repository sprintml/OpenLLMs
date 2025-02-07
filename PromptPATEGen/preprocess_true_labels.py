import json
import os
from data_utils import load_dataset_hf, limit_tokens
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets, DatasetDict
import argparse
import numpy as np 
from utils import random_sampling
from transformers import AutoTokenizer

if __name__=="__main__" :
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--save_file', type=str, help='For saving the public true labels')
    parser.add_argument('--perform_evaluation', action='store_true', help='For saving the evaluation true labels')
    parser.add_argument('--seed', type=int, required=True, help='seeding numpy. also for saving the right file')
    parser.add_argument('--num_student_query', type=int, help='Number of unlabeled student queries')
    parser.add_argument('--num_eval_query', type=int, help='Number of evaluation queries')
    args = parser.parse_args()

    if args.dataset == "disaster" :
        dataset = load_from_disk(f'./data_disaster')
        label2id = {'Relevant' : 0, 'Not Relevant' : 1}
        dataset['train'] = Dataset.from_dict({'sentence' : dataset['train']['sentence1'], 
                                                    'label' : [label2id[label] for label in dataset['train']['label']]})
        dataset['test'] = Dataset.from_dict({'sentence' : dataset['test']['sentence1'], 
                                                    'label' : [label2id[label] for label in dataset['test']['label']]})
        raw_datasets = concatenate_datasets([dataset['train'], dataset['test']])
        dataset = raw_datasets.train_test_split(test_size=len(dataset['test']['label']), shuffle=False)

    elif args.dataset == "docvqa":
        dataset = DatasetDict.load_from_disk("./datasets/DocVQA/hf_format_datasets/DocVQA_client0123")
    elif args.dataset == "mit-d":
        dataset = DatasetDict.load_from_disk("./datasets/movie_data/Director/hf_dataset")
    elif args.dataset == "mit-g":
        dataset = DatasetDict.load_from_disk("./datasets/movie_data/Genre/hf_dataset")
    else:
        dataset = load_dataset(args.dataset)

    
    if args.dataset == 'sst2' :
        np.random.seed(args.seed)
        test_sentences, test_labels, _ = random_sampling(dataset['validation']['sentence'], dataset['validation']['label'], 300, [])
    elif args.dataset == "samsum":
        if args.perform_evaluation:
            # saves the evaluation file. This case, the evaluation / test set!
            np.random.seed(0) #We fix the test set throughout the experiment
            test_sentences, test_labels, _ = random_sampling(dataset['test']['dialogue'],
                                                             dataset['test']['summary'], args.num_eval_query, [])
            print("Evaluation labels", test_labels)

        else:
            np.random.seed(args.seed)
            test_sentences, test_labels, _ = random_sampling(dataset['validation']['dialogue'],
                                                             dataset['validation']['summary'], args.num_student_query, [])

            print("Student labels", test_labels)


    elif args.dataset == "docvqa":
        if args.perform_evaluation:
            # saves the evaluation file. This case, the evaluation / validation set!
            np.random.seed(0) #We fix the validation set throughout the experiment

            # Reduce the example context and query context to 800

            # Apply the function to each example in the dataset
            dataset['validation'] = dataset['validation'].map(limit_tokens)

            # Limit the number of tokens in the context
            # dataset['validation']['contexts'] = [' '.join(tokenizer.tokenize(c)[:800]) for c in dataset['validation']['contexts']]

            test_sentences, test_labels, _ = random_sampling([f'{q} \nContext: {c}' for q, c in zip(dataset['validation']['questions'], dataset['validation']['contexts'])],
                                                             dataset['validation']['answers'], args.num_eval_query, [])
            print("Evaluation labels", test_labels)

        else:
            np.random.seed(args.seed)
            # Apply the function to each example in the dataset
            dataset['test'] = dataset['test'].map(limit_tokens)

            # Limit the number of tokens in the context
            # dataset['test']['contexts'] = [' '.join(tokenizer.tokenize(c)[:800]) for c in dataset['test']['contexts']]

            test_sentences, test_labels, _ = random_sampling([f'{q} \nContext: {c}' for q, c in zip(dataset['test']['questions'], dataset['test']['contexts'])],
                                                             dataset['test']['answers'], args.num_student_query, [])

            print("Student labels", test_labels)


    elif args.dataset == "mit-d" or args.dataset == "mit-g":
        if args.perform_evaluation:
            # saves the evaluation file. This case, the evaluation / test set!
            np.random.seed(0) #We fix the test set throughout the experiment
            test_sentences, test_labels = dataset['test']['content'], dataset['test']['label']
            print("Evaluation labels", test_labels)

        else:
            np.random.seed(args.seed)
            test_sentences, test_labels, _ = random_sampling(dataset['validation']['content'],
                                                             dataset['validation']['label'], args.num_student_query, [])

            print("Student labels", test_labels)


    else:
        np.random.seed(args.seed)
        test_sentences, test_labels, _ = random_sampling(dataset['test']['sentence'], dataset['test']['label'], 300, [])

    if args.dataset != "samsum" and args.dataset != "docvqa" and args.dataset != "mit-d" and args.dataset != "mit-g":
        with open(args.save_file, 'w') as file :
            line = " ".join([str(label) for label in test_labels])
            file.write(line)
    else:
        with open(args.save_file+"_"+str(args.seed), 'w') as file:
            file.write(json.dumps(test_labels)) #To conform with that of DPICL paper format

