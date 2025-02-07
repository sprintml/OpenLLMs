import json
import os
from datasets import load_dataset, Dataset

if __name__=='__main__' :
    dataset = load_dataset('./data_orig').remove_columns('processed_sent')['test'].train_test_split(test_size=1000)
    print(dataset)
    dataset.save_to_disk('./data')