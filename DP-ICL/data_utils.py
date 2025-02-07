from datasets import load_dataset, load_from_disk, Dataset

def load_dataset_hf(params) :
    if 'disaster' not in params['dataset'] and 'trec' not in params['dataset']:
        try:
            dataset = load_dataset(params['dataset'])
        except:
            print(f"{params['dataset']} is not a valid dataset path on huggingface")
    elif 'trec' in params['dataset'] :
        dataset = load_dataset(params['dataset'])
        dataset = dataset.rename_column('coarse_label', 'label').remove_columns('fine_label').rename_column('text', 'sentence')
    else :
        # NOTE : suppose having disaster dataset loaded in local folder
        dataset = load_from_disk(f'./data_disaster')
        label2id = {'Relevant' : 0, 'Not Relevant' : 1}
        for split in dataset.keys() :
            dataset[split] = Dataset.from_dict({'sentence' : dataset[split]['sentence1'], 
                                                        'label' : [label2id[label] for label in dataset[split]['label']]})
    
    train_sentences, train_labels = dataset['train']['sentence'], dataset['train']['label']

    if 'sst2' in params['dataset'] :
        test_sentences, test_labels = dataset['validation']['sentence'], dataset['validation']['label']
    else : 
        test_sentences, test_labels = dataset['test']['sentence'], dataset['test']['label']

    if 'sst2' in params['dataset'] or 'mpqa' in params['dataset'] :
        params['prompt_prefix'] = "Classify this sentence as positive or negative."
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
        params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif 'trec' in params['dataset'] :
        params['prompt_prefix'] = "Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\n"
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer Type: "
        params['label_dict'] = {0: ['Number'], 1: ['Location'], 2: ['Person'], 3: ['Description'], 4: ['Entity'], 5: ['Ab']}
        params['inv_label_dict'] = {'Number': 0, 'Location': 1, 'Person': 2, 'Description': 3, 'Entity': 4, 'Ab': 5}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif 'disaster' in params['dataset'] :
        params['prompt_prefix'] = "Classify the sentence whether they are relevant to a disaster.\n\n"
        params["q_prefix"] = "Sentence: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['Relevant'], 1: ['Not']}
        params['inv_label_dict'] = {'Relevant': 0, 'Not': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    return train_sentences, train_labels, test_sentences, test_labels