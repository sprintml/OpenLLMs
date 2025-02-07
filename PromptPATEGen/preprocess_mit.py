from datasets import Dataset, concatenate_datasets
import pandas as pd
from datasets.load import load_dataset

if __name__=='__main__' :

    data_files = {"train": "./datasets/movie_data/Genre/train.csv", "test": "./datasets/movie_data/Genre/test.csv"}
    genre_dataset = load_dataset("csv", data_files=data_files)
    genre_dataset = genre_dataset.remove_columns(['Unnamed: 0'])

    print(genre_dataset)
    # Create the student dataset from the train set

    train_dataset = genre_dataset['train']
    validation_dataset = train_dataset.select(range(2000, len(train_dataset)))  # First 2000 .shard(2, 0)
    train_dataset = train_dataset.select(range(2000))  # Second half .shard(2, 1)

    # Update the 'train' dataset and add the 'validation' dataset to the DatasetDict
    genre_dataset['train'] = train_dataset
    genre_dataset['validation'] = validation_dataset

    print(genre_dataset["train"][200])
    print(genre_dataset)

    genre_dataset.save_to_disk('./datasets/movie_data/Genre/hf_dataset')


    data_files = {"train": "./datasets/movie_data/Director/train.csv", "test": "./datasets/movie_data/Director/test.csv"}
    director_dataset = load_dataset("csv", data_files=data_files)
    director_dataset = director_dataset.remove_columns(['Unnamed: 0'])

    print(director_dataset)


    train_dataset = director_dataset['train']
    validation_dataset = train_dataset.select(range(1000, len(train_dataset)))  # First 2000 .shard(2, 0)
    train_dataset = train_dataset.select(range(1000))  # Second half .shard(2, 1)

    # Update the 'train' dataset and add the 'validation' dataset to the DatasetDict
    director_dataset['train'] = train_dataset
    director_dataset['validation'] = validation_dataset

    print(director_dataset["train"][100])

    print(director_dataset)

    director_dataset.save_to_disk('./datasets/movie_data/Director/hf_dataset')
