import logging
import os
import sys
import numpy as np
from typing import Dict

from datasets import Dataset
from datasets import DatasetDict


os.environ["WANDB_DISABLED"] = "true"

logger = logging.getLogger(__name__)

if __name__ == '__main__':

# Train set
    question_id = []
    questions = []
    contexts = []
    answers_ = []
    print("Processing train dataset")
    for client_id in range(4):
        print(f"Client_id: {client_id}")

        data = np.load(os.path.join("./datasets/DocVQA/raw_files/", f"imdb_train_client_{client_id}.npy"), allow_pickle=True)[1:]
        print(f"amount of data from client {client_id}: {len(data)}")
        for idx, record in enumerate(data):
            question = record["question"]
            answers = record['answers'].lower()
            context = " ".join([word.lower() for word in record['ocr_tokens']])

            question_id.append(record.get('question_id', "{:s}-{:d}".format(record['set_name'], idx)))
            questions.append(question)
            contexts.append(context)
            answers_.append(answers)

    print(f"len of questions ids: {len(question_id)}")
    print(f"len of set questions ids: {len(set(question_id))}")


    data_dict = {"question_id": question_id, "questions": questions, "contexts": contexts, "answers": answers_}
    train_dataset = Dataset.from_dict({"question_id": question_id, "questions": questions, "contexts": contexts, "answers": answers_})






# test set
    question_id = []
    questions = []
    contexts = []
    answers_ = []
    print("Processing test dataset")
    for client_id in range(4, 8):
        print(f"Client_id: {client_id}")

        data = np.load(os.path.join("./datasets/DocVQA/raw_files/", f"imdb_train_client_{client_id}.npy"), allow_pickle=True)[1:]
        print(f"amount of data from client {client_id}: {len(data)}")
        for idx, record in enumerate(data):
            question = record["question"]
            answers = record['answers'].lower()
            context = " ".join([word.lower() for word in record['ocr_tokens']])

            question_id.append(record.get('question_id', "{:s}-{:d}".format(record['set_name'], idx)))
            questions.append(question)
            contexts.append(context)
            answers_.append(answers)

    print(f"len of questions ids: {len(question_id)}")
    print(f"len of set questions ids: {len(set(question_id))}")


    data_dict = {"question_id": question_id, "questions": questions, "contexts": contexts, "answers": answers_}
    test_dataset = Dataset.from_dict({"question_id": question_id, "questions": questions, "contexts": contexts, "answers": answers_})








# Validation set
    question_id = []
    questions = []
    contexts = []
    answers_ = []

    print(f"Processing validation dataset")

    data = np.load(os.path.join("./datasets/DocVQA/raw_files/", f"imdb_val.npy"), allow_pickle=True)[1:]
    print(f"amount of data from validset: {len(data)}")
    for idx, record in enumerate(data):
        question = record["question"]
        answers = record['answers'].lower()
        context = " ".join([word.lower() for word in record['ocr_tokens']])

        question_id.append(record.get('question_id', "{:s}-{:d}".format(record['set_name'], idx)))
        questions.append(question)
        contexts.append(context)
        answers_.append(answers)

    print(f"len of questions ids: {len(question_id)}")
    print(f"len of set questions ids: {len(set(question_id))}")
    valid_dataset = Dataset.from_dict({"question_id": question_id, "questions": questions, "contexts": contexts, "answers": answers_})



    train_test_data = DatasetDict({"train": train_dataset, "test": test_dataset, "validation": valid_dataset})

    train_test_data.save_to_disk("./datasets/DocVQA/hf_format_datasets/DocVQA_client0123")


    for i in range(5):
        print(train_test_data["train"][i])
        print(train_test_data["test"][i])
        print(train_test_data["validation"][i])
    