from torch.utils.data.dataloader import DataLoader
from datasets import load_dataset, Dataset, load_from_disk
from transformers.data.data_collator import DataCollatorWithPadding
from transformers import BertTokenizerFast, BertTokenizer, GPT2Tokenizer, LlamaTokenizer, AutoTokenizer


def get_dataloader(task:str, model_checkpoint:str, split:str, dataloader_drop_last:bool=True, shuffle:bool=False,
                   batch_size:int=16, dataloader_num_workers:int=0, dataloader_pin_memory:bool=True) -> DataLoader:
    """To create encoded dataset dataloader for a given GLUE task.

    Args:
        task (str): GLUE task.
        model_checkpoint (str): tokenizer restoring model_checkpoint.
        split (str): "train", "validation", "test".
        dataloader_drop_last (bool, optional): Whether to drop the last incomplete batch (if the length of the dataset is not divisible by the batch size) or not. Defaults to True.
        batch_size (int): Number of samples in each batch.
        dataloader_num_workers (int, optional): Number of subprocesses to use for data loading (PyTorch only). 0 means that the data will be loaded in the main process. Defaults to 0.
        dataloader_pin_memory (bool, optional): Whether you want to pin memory in data loaders or not. Defaults to True.

    Returns:
        dataloader(DataLoader): A tokenized and encoded dataloader.
    """
    task_to_keys = {
    "cola": ("sentence", None),
    "mnli": ("premise", "hypothesis"),
    "mnli-mm": ("premise", "hypothesis"),
    "mrpc": ("sentence1", "sentence2"),
    "qnli": ("question", "sentence"),
    "qqp": ("question1", "question2"),
    "rte": ("sentence1", "sentence2"),
    "sst2": ("sentence", None),
    "stsb": ("sentence1", "sentence2"),
    "wnli": ("sentence1", "sentence2"),
    "trec": ('text', None),
    "mpqa" :('sentence', None),
    "disaster": ('sentence1', None)
    }

    sentence1_key, sentence2_key = task_to_keys[task]
    
    def preprocess(examples) :
        return tokenizer(examples[sentence1_key], padding=True, truncation=True)

    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint, use_fast=True, revision='main')
    tokenizer.pad_token = tokenizer.eos_token
    data_collator = DataCollatorWithPadding(tokenizer)

    actual_task = "mnli" if task == "mnli-mm" else task
    if task == 'trec' :
        print('loading trec dataset')
        dataset = load_dataset('trec')
    elif task == 'mpqa' :
        print('loading mpqa dataset')
        dataset = load_dataset('jxm/mpqa')
    elif task == 'disaster' : 
        print('loading disaster dataset')
        # Assumes that disaster loaded in local file
        dataset = load_from_disk(f'./data_disaster')
        label2id = {'Relevant' : 0, 'Not Relevant' : 1}
        dataset[split] = Dataset.from_dict({'sentence1' : dataset[split]['sentence1'], 
                                                    'label' : [label2id[label] for label in dataset[split]['label']]})
    else : 
        dataset = load_dataset(actual_task)
    encoded_dataset = dataset.map(preprocess, batched=True)
    
    if task == 'trec' :
        encoded_dataset = encoded_dataset.rename_column('coarse_label', 'label')
    columns_to_return = ['input_ids', 'label', 'attention_mask'] 
    encoded_dataset.set_format(type='torch', columns=columns_to_return)
    encoded_dataset.rename_column('label', 'labels')

    if (split == "validation" or split == "test") and task == "mnli":
        split = "validation_matched"
    if (split == "validation" or split == "test") and task == "mnli-mm":
        split = "validation_mismatched"
    
    dataloader = DataLoader(
                    encoded_dataset[split],
                    shuffle=shuffle,
                    batch_size=batch_size,
                    collate_fn=data_collator,
                    # drop_last=dataloader_drop_last,
                    num_workers=dataloader_num_workers,
                    pin_memory=False,
    )
    
    return dataloader