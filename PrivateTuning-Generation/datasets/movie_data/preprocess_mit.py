from datasets import Dataset, concatenate_datasets
import pandas as pd
from datasets.load import load_dataset

if __name__=='__main__' :

    data_files = {"train": "./Genre/train.csv", "test": "./Genre/test.csv"}
    genre_dataset = load_dataset("csv", data_files=data_files)
    genre_dataset = genre_dataset.remove_columns(['Unnamed: 0'])

    print(genre_dataset)
    print(genre_dataset["train"][2000])

    genre_dataset.save_to_disk('./Genre/')


    data_files = {"train": "./Director/train.csv", "test": "./Director/test.csv"}
    director_dataset = load_dataset("csv", data_files=data_files)
    director_dataset = director_dataset.remove_columns(['Unnamed: 0'])

    print(director_dataset)
    print(director_dataset["train"][1000])

    director_dataset.save_to_disk('./Director/')
