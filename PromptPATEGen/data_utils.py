import pandas as pd
import json
import pickle
import numpy as np
from utils import ROOT_DIR
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
import sys
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")


# To limit the number of tokens in the context
def limit_tokens(example):
    tokens = tokenizer.tokenize(example['contexts'])
    limited_tokens = tokens[:800]
    example['contexts'] = tokenizer.convert_tokens_to_string(limited_tokens)
    return example


def load_sst2():
    def process_raw_data_sst(lines):
        """from lines in dataset to two lists of sentences and labels respectively"""
        labels = []
        sentences = []
        for line in lines:
            labels.append(int(line[0]))
            sentences.append(line[2:].strip())
        return sentences, labels

    with open(f"{ROOT_DIR}/data/sst2/stsa.binary.train", "r") as f:
        train_lines = f.readlines()
    with open(f"{ROOT_DIR}/data/sst2/stsa.binary.test", "r") as f:
        test_lines = f.readlines()
    train_sentences, train_labels = process_raw_data_sst(train_lines)
    test_sentences, test_labels = process_raw_data_sst(test_lines)
    return train_sentences, train_labels, test_sentences, test_labels

def load_agnews():
    train_data = pd.read_csv(f'{ROOT_DIR}/data/agnews/train.csv')
    test_data = pd.read_csv(f'{ROOT_DIR}/data/agnews/test.csv')

    train_sentences = train_data['Title'] + ". " + train_data['Description']
    train_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in train_sentences]) # some basic cleaning
    train_labels = list(train_data['Class Index'])
    test_sentences = test_data['Title'] + ". " + test_data['Description']
    test_sentences = list(
        [item.replace(' #39;s', '\'s').replace(' quot;', "\"").replace('\\', " ").replace(' #39;ll', "'ll") for item
         in test_sentences]) # some basic cleaning
    test_labels = list(test_data['Class Index']) 
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4
    test_labels = [l - 1 for l in test_labels]
    return train_sentences, train_labels, test_sentences, test_labels

def load_trec():
    inv_label_dict = {'NUM': 0, 'LOC': 1, 'HUM': 2, 'DESC': 3, 'ENTY': 4, 'ABBR': 5}
    train_sentences = []
    train_labels = []
    with open(f'{ROOT_DIR}/data/trec/train.txt', 'r') as train_data:
        for line in train_data:
            train_label = line.split(' ')[0].split(':')[0]
            train_label = inv_label_dict[train_label]
            train_sentence = ' '.join(line.split(' ')[1:]).strip()
            # basic cleaning
            train_sentence = train_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            train_labels.append(train_label)
            train_sentences.append(train_sentence)

    test_sentences = []
    test_labels = []
    with open(f'{ROOT_DIR}/data/trec/test.txt', 'r') as test_data:
        for line in test_data:
            test_label = line.split(' ')[0].split(':')[0]
            test_label = inv_label_dict[test_label]
            test_sentence = ' '.join(line.split(' ')[1:]).strip()
            test_sentence = test_sentence.replace(" 's", "'s").replace('`` ', '"').replace(" ''",'"').replace(' ?','?').replace(' ,',',')
            test_labels.append(test_label)
            test_sentences.append(test_sentence)
    return train_sentences, train_labels, test_sentences, test_labels

def get_cb():
    train_questions = []
    train_answers = []
    with open(f"{ROOT_DIR}/data/cb/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            curr_label = myjson['label']
            if curr_label == 'contradiction':
                train_answers.append(0)
            elif curr_label == 'neutral':
                train_answers.append(1)
            elif curr_label == 'entailment':
                train_answers.append(2)
            # being a bit lazy here. We put the "question: " into the input and treat it like single sentence classification.
            train_questions.append(p.strip() + '\n' + 'question: ' + q + '. true, false, or neither?')

    test_questions = []
    test_answers = []
    with open(f"{ROOT_DIR}/data/cb/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'contradiction':
                test_answers.append(0)
            elif myjson['label'] == 'neutral':
                test_answers.append(1)
            elif myjson['label'] == 'entailment':
                test_answers.append(2)
            else:
                exit('answer')
            test_questions.append(p.strip() + '\n' + 'question: ' + q + '. true, false, or neither?')

    return train_questions, train_answers, test_questions, test_answers

def load_dbpedia():
    train_data = pd.read_csv(f'{ROOT_DIR}/data/dbpedia/train_subset.csv')
    test_data = pd.read_csv(f'{ROOT_DIR}/data/dbpedia/test.csv')

    train_sentences = train_data['Text']
    train_sentences = list([item.replace('""', '"') for item in train_sentences])
    train_labels = list(train_data['Class'])

    test_sentences = test_data['Text']
    test_sentences = list([item.replace('""', '"') for item in test_sentences])
    test_labels = list(test_data['Class'])
    
    train_labels = [l - 1 for l in train_labels] # make them 0, 1, 2, 3 instead of 1, 2, 3, 4...
    test_labels = [l - 1 for l in test_labels]
    return train_sentences, train_labels, test_sentences, test_labels

def load_slot_movies(field_name):
    all_fields = ["Actor", "Award", "Character_Name", "Director", "Genre", "Opinion", "Origin", "Plot", "Quote", "Relationship", "Soundtrack", "Year"]
    assert field_name in all_fields
    all_fields.remove(field_name)
    filter_tags = [f"B-{field}" for field in all_fields] + [f"I-{field}" for field in all_fields] + ["O"]
    target_tags = [f"B-{field_name}", f"I-{field_name}"]

    with open(f'{ROOT_DIR}/data/slot-movies/train', 'r') as f:
        lines = f.readlines()
        lines = [line.replace(' <=> <NULL>','').strip() for line in lines]
    train_answers = []
    train_sentences = []
    for line in lines:
        answer = ''
        untagged_line = ''
        for word in line.split(' '):
            contains_target = [tag in word for tag in target_tags]
            if np.any(contains_target):
                for tag in target_tags:
                    word = word.replace(':' + tag, '')
                answer += word + ' '
            for tag in filter_tags:
                word = word.replace(':' + tag, '')
            untagged_line += word + ' '

        if answer != '':
            train_answers.append(answer.strip())
            train_sentences.append(untagged_line.strip())

    with open(f'{ROOT_DIR}/data/slot-movies/test', 'r') as f:
        lines = f.readlines()
        lines = [line.replace(' <=> <NULL>','').strip() for line in lines]
    test_answers = []
    test_sentences = []
    for line in lines:
        answer = ''
        untagged_line = ''
        for word in line.split(' '):
            contains_target = [tag in word for tag in target_tags]
            if np.any(contains_target):
                for tag in target_tags:
                    word = word.replace(':' + tag, '')
                answer += word + ' '
            for tag in filter_tags:
                word = word.replace(':' + tag, '')
            untagged_line += word + ' '

        if answer != '':
            test_answers.append(answer.strip())
            test_sentences.append(untagged_line.strip())

    return train_sentences, train_answers, test_sentences, test_answers

def load_atis(tag_name):
    with open(f'{ROOT_DIR}/data/atis/atis.train.pkl', 'rb') as stream:
        ds,dicts = pickle.load(stream)

    t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids','intent_ids'])
    i2t, i2s, i2in = map(lambda d: {d[k]:k for k in d.keys()}, [t2i,s2i,in2i])
    query, slots, intent =  map(ds.get, ['query', 'slot_labels', 'intent_labels'])

    tags_dict = {}
    train_sentences = []
    train_slot_strings = []
    for i in range(len(query)):
        slot_string = ''
        beginning_count = 0 # when there are multiple mentions of the destination city, we want to avoid those
        for j in range(len(query[i])):
            tag = i2s[slots[i][j]][2:]
            if tag in tags_dict.keys():
                tags_dict[tag] += 1
            else:
                tags_dict[tag] = 1

            if f'B-{tag_name}' in i2s[slots[i][j]]:
                beginning_count += 1
            if tag_name in i2s[slots[i][j]]:
                slot_string += i2t[query[i][j]] + ' '
        if slot_string != '' and beginning_count == 1:
            train_sentences.append(' '.join(map(i2t.get, query[i][1:-1]))) # [1:-1] cuts off BOS and EOS
            train_slot_strings.append(slot_string.strip())

    with open(f'{ROOT_DIR}/data/atis/atis.test.pkl', 'rb') as stream:
        ds,dicts = pickle.load(stream)

    t2i, s2i, in2i = map(dicts.get, ['token_ids', 'slot_ids','intent_ids'])
    i2t, i2s, i2in = map(lambda d: {d[k]:k for k in d.keys()}, [t2i,s2i,in2i])
    query, slots, intent =  map(ds.get, ['query', 'slot_labels', 'intent_labels'])

    test_sentences = []
    test_slot_strings = []
    for i in range(len(query)):
        slot_string = ''
        beginning_count = 0 # when there are multiple mentions of the destination city, we want to avoid those
        for j in range(len(query[i])):
            if f'B-{tag_name}' in i2s[slots[i][j]]:
                beginning_count += 1
            if tag_name in i2s[slots[i][j]]:
                slot_string += i2t[query[i][j]] + ' '
        if slot_string != '' and beginning_count == 1:
            test_sentences.append(' '.join(map(i2t.get, query[i][1:-1]))) # [1:-1] cuts off BOS and EOS
            test_slot_strings.append(slot_string.strip())

    return train_sentences, train_slot_strings, test_sentences, test_slot_strings

def load_lama(which_lama):
    ### Load test data
    with open(f'{ROOT_DIR}/data/lama/original_rob/P{which_lama}/test.jsonl', 'r') as json_file:
        json_list = list(json_file)
    all_y_test = []
    all_x_test = []
    for json_str in json_list:
        result = json.loads(json_str)
        all_y_test.append(result['obj_label'])
        all_x_test.append(result['sub_label'])

    ### Load train data
    with open(f'{ROOT_DIR}/data/lama/original_rob/P{which_lama}/train.jsonl', 'r') as json_file:
        json_list = list(json_file)
    all_y_train = []
    all_x_train = []
    for json_str in json_list[:1000]:
        result = json.loads(json_str)
        all_y_train.append(result['obj_label'])
        all_x_train.append(result['sub_label'])

    with open(f'{ROOT_DIR}/data/lama/relations.jsonl', 'r') as json_file:
        json_list = list(json_file)
    template = None
    for json_str in json_list:
        result = json.loads(json_str)
        idx = int(result['relation'][1:])
        if idx == which_lama:
            template = result['template']
            x_pos = template.find('[X]')
            y_pos = template.find('[Y]')
            assert (x_pos >= 0) and (y_pos >= 0), "placeholder not found"
            if x_pos > y_pos:
                print("Not auto-regressive, skip")
                template = "INVALID"
            break

    return all_x_train, all_y_train, all_x_test, all_y_test, template

def load_rte():
    train_questions = []
    train_answers = []
    with open("data/rte/train.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                train_answers.append(0)
            elif myjson['label'] == 'entailment':
                train_answers.append(1)
            else:
                exit('answer')
            train_questions.append(p + '\n' + 'question: ' + q + ' True or False?')

    test_questions = []
    test_answers = []
    with open("data/rte/val.jsonl", "r") as f:
        for line in f:
            myjson = json.loads(line)
            q = myjson['hypothesis']
            p = myjson['premise']
            if myjson['label'] == 'not_entailment':
                test_answers.append(0)
            elif myjson['label'] == 'entailment':
                test_answers.append(1)
            else:
                exit('answer')
            test_questions.append(p + '\n' + 'question: ' + q + ' True or False?')

    return train_questions, train_answers, test_questions, test_answers


def load_samsum():
    samsum_dataset = load_dataset("samsum")
    # print("samsum dataset", samsum_dataset)
    samsum_train_dialogue = samsum_dataset["train"][0]["dialogue"]
    samsum_train_summary = samsum_dataset["train"][0]["summary"]
    # print(samsum_train_dialogue)
    # print(samsum_train_summary)


    # Extract the dialogues and summaries from each split
    train_dialogues = [example['dialogue'] for example in samsum_dataset['train']]
    train_summarizations = [example['summary'] for example in samsum_dataset['train']]

    test_dialogues = [example['dialogue'] for example in samsum_dataset['test']]
    test_summarizations = [example['summary'] for example in samsum_dataset['test']]

    validation_dialogues = [example['dialogue'] for example in samsum_dataset['validation']]
    validation_summarizations = [example['summary'] for example in samsum_dataset['validation']]

    return train_dialogues, train_summarizations, test_dialogues, test_summarizations #, validation_dialogues, validation_summarizations


def load_train_dataset(params):
    """
    Load train and test data
    :param params: experiment parameter, which contains dataset spec
    :return: train_x, train_y, test_x, test_y
    """
    print(params['train_dataset'])
    if params['train_dataset'] == 'sst2':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst2()
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
        params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['train_dataset'] == 'agnews':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_agnews()
        params['prompt_prefix'] = "Classify the news articles into the categories of World, Sports, Business, and Technology.\n\n"
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['World'], 1: ['Sports'], 2: ['Business'], 3: ['Technology', 'Science']}
        params['inv_label_dict'] = {'World': 0, 'Sports': 1, 'Business': 2, 'Technology': 3, 'Science': 3} # notice index start from 1 here
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1


    elif params['train_dataset'] == 'news':
        #orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_agnews()
        with open("data/news.txt", "r") as f:
    # Read the contents of the file and store each line as a string in a list
            orig_train_sentences = [line.strip() for line in f.readlines()]
        #orig_test_sentences = orig_test_sentences
        orig_train_labels = [0] * len(orig_train_sentences)
        params['prompt_prefix'] = "Classify the news articles into the categories of World, Sports, Business, and Technology.\n\n"
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['World'], 1: ['Sports'], 2: ['Business'], 3: ['Technology', 'Science']}
        params['inv_label_dict'] = {'World': 0, 'Sports': 1, 'Business': 2, 'Technology': 3, 'Science': 3} # notice index start from 1 here
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1


    elif params['train_dataset'] == 'imdb':
        orig_train_sentences = None
        with open("data/imdb.txt", "r") as f:
    # Read the contents of the file and store each line as a string in a list
            orig_train_sentences = [line.strip() for line in f.readlines()]
        orig_train_sentences = orig_train_sentences
        orig_train_labels = [0] * len(orig_train_sentences)
        params['prompt_prefix'] = ""
        params["q_prefix"] = "Review: "
        params["a_prefix"] = "Sentiment: "
        params['label_dict'] = {0: ['Negative'], 1: ['Positive']}
        params['inv_label_dict'] = {'Negative': 0, 'Positive': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1


    elif params['train_dataset'] == 'trec':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_trec()
        params['prompt_prefix'] = "Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\n"
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer Type: "
        params['label_dict'] = {0: ['Number'], 1: ['Location'], 2: ['Person'], 3: ['Description'], 4: ['Entity'], 5: ['Ab']}
        params['inv_label_dict'] = {'Number': 0, 'Location': 1, 'Person': 2, 'Description': 3, 'Entity': 4, 'Ab': 5}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['train_dataset'] == 'qqp':
        orig_train_sentences = None
        with open("data/qqp.txt", "r") as f:
    # Read the contents of the file and store each line as a string in a list
            orig_test_sentences = [line.strip() for line in f.readlines()]
        orig_train_sentences = orig_test_sentences
        orig_train_labels = [0] * len(orig_test_sentences)
        params['prompt_prefix'] = "Classify the questions based on whether their answer type is a Number, Location, Person, Description, Entity, or Abbreviation.\n\n"
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer Type: "
        params['label_dict'] = {0: ['Number'], 1: ['Location'], 2: ['Person'], 3: ['Description'], 4: ['Entity'], 5: ['Ab']}
        params['inv_label_dict'] = {'Number': 0, 'Location': 1, 'Person': 2, 'Description': 3, 'Entity': 4, 'Ab': 5}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['train_dataset'] == 'rte':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_rte()
        params['prompt_prefix'] = ""
        params["q_prefix"] = " "
        params["a_prefix"] = "answer: "
        params['label_dict'] = {0: ['False'], 1: ['True']}
        params['inv_label_dict'] = {'False': 0, 'True': 1}
        params['num_user_input'] = 2
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['train_dataset'] == 'cb':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = get_cb()
        params['prompt_prefix'] = ""
        params["q_prefix"] = ""
        params["a_prefix"] = "answer: "
        params['label_dict'] = {0: ['false'], 1: ['neither'], 2: ['true']}
        params['inv_label_dict'] = {'false': 0, 'neither': 1, 'true': 2}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['train_dataset'] == 'dbpedia':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_dbpedia()
        params['prompt_prefix'] = "Classify the documents based on whether they are about a Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, or Book.\n\n"
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['Company'], 1: ['School'], 2: ['Artist'], 3: ['Ath'], 4: ['Polit'], 5: ['Transportation'], 6: ['Building'], 7: ['Nature'], 8: ['Village'], 9: ['Animal'], 10: ['Plant'], 11: ['Album'], 12: ['Film'], 13: ['Book']}
        params['inv_label_dict'] = {'Company': 0, 'School': 1, 'Artist': 2, 'Ath': 3, 'Polit': 4, 'Transportation': 5, 'Building': 6, 'Nature': 7, 'Village': 8, 'Animal': 9, 'Plant': 10, 'Album': 11, 'Film': 12, 'Book': 13}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['train_dataset'] == 'agnews_public':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_agnews()
        params['prompt_prefix'] = "Classify the documents based on whether they are about a Company, School, Artist, Athlete, Politician, Transportation, Building, Nature, Village, Animal, Plant, Album, Film, or Book.\n\n"
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['Company'], 1: ['School'], 2: ['Artist'], 3: ['Ath'], 4: ['Polit'], 5: ['Transportation'], 6: ['Building'], 7: ['Nature'], 8: ['Village'], 9: ['Animal'], 10: ['Plant'], 11: ['Album'], 12: ['Film'], 13: ['Book']}
        params['inv_label_dict'] = {'Company': 0, 'School': 1, 'Artist': 2, 'Ath': 3, 'Polit': 4, 'Transportation': 5, 'Building': 6, 'Nature': 7, 'Village': 8, 'Animal': 9, 'Plant': 10, 'Album': 11, 'Film': 12, 'Book': 13}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1


    elif params['train_dataset'] == 'dbpedia_public':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_dbpedia()
        params['prompt_prefix'] = "Classify the news articles into the categories of World, Sports, Business, and Technology.\n\n"
        params["q_prefix"] = "Article: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = {0: ['World'], 1: ['Sports'], 2: ['Business'], 3: ['Technology', 'Science']}
        params['inv_label_dict'] = {'World': 0, 'Sports': 1, 'Business': 2, 'Technology': 3, 'Science': 3} # notice index start from 1 here
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif params['train_dataset'] == 'samsum':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_samsum()
        params['prompt_prefix'] = "Summarize the dialogue.\n\n"
        params["q_prefix"] = "Dialogue: "
        params["a_prefix"] = "Summarization: "
        params['label_dict'] = " "
        params['inv_label_dict'] = " "
        params['task_format'] = 'summarization'
        params['num_tokens_to_predict'] = 100 #This shoould be variable rather than fixed from3 to 300!

    elif params['train_dataset'][:4] == 'lama':
        which_lama = int(params['dataset'].split('_')[-1])
        all_x_train, all_y_train, all_x_test, all_y_test, template = load_lama(which_lama)

        # reject if template is not valid
        if template == "INVALID":
            params['template'] = template
            return None, None, None, None

        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = all_x_train, all_y_train, all_x_test, all_y_test
        params['prompt_prefix'] = ""
        params['task_format'] = 'qa'
        params['num_tokens_to_predict'] = 1
        params['template'] = template

        x_pos = template.find('[X]')
        y_pos = template.find('[Y]')
        seg1 = template[0:x_pos]
        seg2 = template[x_pos+3:y_pos]

        def single_prompt_func(entity, target):
            return f"{seg1}{entity}{seg2}{target}"

        def prompt_func(params, train_sentences, train_labels, test_sentence, test_label_option=None):
            assert seg2[-1] == " "
            prompt = ""
            for x, y in zip(train_sentences, train_labels):
                prompt += single_prompt_func(x, y)
                prompt += "\n\n"

            if test_label_option is None:
                prompt += f"{seg1}{test_sentence}{seg2}"[:-1]
            else:
                prompt += f"{seg1}{test_sentence}{seg2}"[:-1] + test_label_option
            return prompt

        example = single_prompt_func(orig_train_sentences[0], orig_train_labels[0])
        print(f"Sentence example: ||{example}||")

        params['prompt_func'] = prompt_func
        params['single_prompt_func'] = single_prompt_func

    elif params['train_dataset'][:9] == 'mit_movie':
        field_name = params['dataset'][10:]
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_slot_movies(field_name)
        """
        Actor 944
        Award 54
        Character_Name 225
        Director 415
        Genre 780
        Opinion 190
        Origin 178
        Plot 1459
        Quote 43
        Relationship 147
        Soundtrack 7
        Year 655
        """

        params['prompt_prefix'] = ""
        params["q_prefix"] = "Sentence: "
        pararun_classification_index.pyms["a_prefix"] = f"{field_name}: "
        params['task_format'] = 'qa'
        params['num_tokens_to_predict'] = 1


        def prompt_func(params, train_sentences, train_labels, test_sentence, test_label_option=None):
            q_prefix = params["q_prefix"]
            a_prefix = params["a_prefix"]

            prompt = params['prompt_prefix']
            for x, y in zip(train_sentences, train_labels):
                prompt += f"{q_prefix}{x}\n{a_prefix}{y}"
                prompt += "\n\n"

            if test_label_option is None:
                prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"[:-1]
            else:
                prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"[:-1] + test_label_option
            return prompt

        params['prompt_func'] = prompt_func

    elif params['train_dataset'][:4] == 'atis':
        tag_name = params['dataset'][5:]
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_atis(tag_name)

        name2prefix = {
            "airline_name": "Airline name",
            "depart_time.period_of_day": "Depart time - Period of day",
            "depart_date.day_name": "Depart date - Day name"
        }

        params['prompt_prefix'] = ""
        params["q_prefix"] = "Sentence: "
        params["a_prefix"] = f"{name2prefix[tag_name]}: "
        params['task_format'] = 'qa'
        params['num_tokens_to_predict'] = 1

        def prompt_func(params, train_sentences, train_labels, test_sentence, test_label_option=None):
            q_prefix = params["q_prefix"]
            a_prefix = params["a_prefix"]

            prompt = params['prompt_prefix']
            for x, y in zip(train_sentences, train_labels):
                prompt += f"{q_prefix}{x}\n{a_prefix}{y}"
                prompt += "\n\n"

            if test_label_option is None:
                prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"[:-1]
            else:
                prompt += f"{q_prefix}{test_sentence}\n{a_prefix}"[:-1] + test_label_option
            return prompt

        params['prompt_func'] = prompt_func

    else:
        raise NotImplementedError
    return orig_train_sentences, orig_train_labels


def load_test_dataset(params):
    """
    Load train and test data
    :param params: experiment parameter, which contains dataset spec
    :return: train_x, train_y, test_x, test_y
    """

    if params['test_dataset'] == 'sst2':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_sst2()
        #print(orig_test_labels)

    elif params['test_dataset'] == 'agnews':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_agnews()

    elif params['test_dataset'] == 'agnews_public':
        orig_test_sentences, orig_test_labels, _, _ = load_agnews()

    elif params['test_dataset'] == 'trec':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_trec()
        #print(orig_test_labels)
    elif params['test_dataset'] == 'rte':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_rte()


    elif params['test_dataset'] == 'cb':
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = get_cb()


    elif params['test_dataset'] == 'dbpedia':
        #print(orig_test_labels)
        orig_train_sentences, orig_train_labels, orig_test_sentences, orig_test_labels = load_dbpedia()
    #print(orig_test_labels)

    elif params['test_dataset'] == 'dbpedia_public':
        orig_test_sentences, orig_test_labels, _, _ = load_dbpedia()
        

    elif params['test_dataset'] == 'imdb':
        orig_train_sentences = None
        with open("data/imdb.txt", "r") as f:
    # Read the contents of the file and store each line as a string in a list
            orig_test_sentences = [line.strip() for line in f.readlines()]
        orig_test_sentences = orig_test_sentences
        orig_test_labels = [0] * len(orig_test_sentences)
        #dataset = load_dataset('imdb', split='unsupervised')
        #orig_train_sentences = dataset["text"][:2000]
        #orig_train_labels = dataset["label"][:2000]

    elif params['test_dataset'] == 'news':
        orig_train_sentences = None
        with open("data/news.txt", "r") as f:
    # Read the contents of the file and store each line as a string in a list
            orig_test_sentences = [line.strip() for line in f.readlines()]
        orig_test_sentences = orig_test_sentences
        orig_test_labels = [0] * len(orig_test_sentences)

    elif params['test_dataset'] == 'qqp':
        orig_train_sentences = None
        with open("data/qqp.txt", "r") as f:
    # Read the contents of the file and store each line as a string in a list
            orig_test_sentences = [line.strip() for line in f.readlines()]
        orig_test_sentences = orig_test_sentences
        orig_test_labels = [0] * len(orig_test_sentences)


    return orig_test_sentences, orig_test_labels

def load_dataset_local(params):
    train_sentences, train_labels =  load_train_dataset(params) 
    test_sentences, test_labels = load_test_dataset(params)
    return train_sentences, train_labels, test_sentences, test_labels

# NOTE : for the moment ONLY for task in which input key is 'sentence' and just sst2 in fact because of the 'validation' key
def load_dataset_hf(params) :
    if 'disaster' not in params['dataset'] and 'docvqa' not in params['dataset'] and 'mit-d' not in params['dataset'] and 'mit-g' not in params['dataset']:
        try:
            dataset = load_dataset(params['dataset'])
        except:
            print(f"{params['dataset']} is not a valid dataset path on huggingface")
    else:
        if 'docvqa' in params['dataset']:
            dataset = DatasetDict.load_from_disk("./datasets/DocVQA/hf_format_datasets/DocVQA_client0123")
        elif 'mit-d' in params['dataset']:
            dataset = DatasetDict.load_from_disk("./datasets/movie_data/Director/hf_dataset")
        elif 'mit-g' in params['dataset']:
            dataset = DatasetDict.load_from_disk("./datasets/movie_data/Genre/hf_dataset")
        else:
            dataset = load_from_disk(f'./data_disaster')
            label2id = {'Relevant' : 0, 'Not Relevant' : 1}
            for split in dataset.keys() :
                dataset[split] = Dataset.from_dict({'sentence' : dataset[split]['sentence1'],
                                                            'label' : [label2id[label] for label in dataset[split]['label']]})

    if 'samsum' in params['dataset']:
        train_sentences, train_labels = dataset['train']['dialogue'], dataset['train']['summary']
    elif 'docvqa' in params['dataset']:
        # Reduce the example context and query context to 800

        # Apply the function to each example in the dataset
        dataset['train'] = dataset['train'].map(limit_tokens)

        # dataset['train']['contexts'] = [' '.join(tokenizer.tokenize(c)[:800]) for c in dataset['train']['contexts']]
        train_sentences, train_labels =  [f'{q} \nContext: {c}' for q, c in zip(dataset['train']['questions'], dataset['train']['contexts'])], dataset["train"]["answers"]

    elif 'mit-d' in params['dataset'] or 'mit-g' in params['dataset']:
        train_sentences, train_labels = dataset['train']['content'], dataset['train']['label']

    else:
        train_sentences, train_labels = dataset['train']['sentence'], dataset['train']['label']

    if 'sst2' in params['dataset']:
        test_sentences, test_labels = dataset['validation']['sentence'], dataset['validation']['label']
    elif 'samsum' in params['dataset']:
        test_sentences, test_labels = dataset['validation']['dialogue'], dataset['validation']['summary'] #Public data
        eval_sentences, eval_labels = dataset['test']['dialogue'], dataset['test']['summary'] #For evaluation
    elif 'docvqa' in params["dataset"]:
        # Reduce the example context and query context to 800

        # Apply the function to each example in the dataset
        dataset['test'] = dataset['test'].map(limit_tokens)
        dataset['validation'] = dataset['validation'].map(limit_tokens)
        # Limit the number of tokens in the context
        # dataset['test']['contexts'] = [' '.join(tokenizer.tokenize(c)[:800]) for c in dataset['test']['contexts']]
        # dataset['validation']['contexts'] = [' '.join(tokenizer.tokenize(c)[:800]) for c in dataset['validation']['contexts']]

        test_sentences, test_labels = [f'{q} \nContext: {c}' for q, c in zip(dataset['test']['questions'], dataset['test']['contexts'])], dataset["test"]["answers"] # Public data
        eval_sentences, eval_labels = [f'{q} \nContext: {c}' for q, c in zip(dataset['validation']['questions'], dataset['validation']['contexts'])], dataset["validation"]["answers"]  # For evaluation
    elif 'mit-d' in params['dataset'] or 'mit-g' in params['dataset']:
        test_sentences, test_labels = dataset['validation']['content'], dataset['validation']['label'] #Public data
        eval_sentences, eval_labels = dataset['test']['content'], dataset['test']['label'] #For evaluation

    else:
        test_sentences, test_labels = dataset['test']['sentence'], dataset['test']['label']

    if 'sst2' in params['dataset'] or 'mpqa' in params['dataset'] :
        params['prompt_prefix'] = ""
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
        # NOTE : Choose 'Yes' and 'No' format to have only 1 generated token for the class
        params['prompt_prefix'] = "Classify the sentence whether they are relevant to a disaster.\n\n"
        params["q_prefix"] = "Sentence: "
        params["a_prefix"] = "Label: "
        params['label_dict'] = {0: ['Yes'], 1: ['No']}
        params['inv_label_dict'] = {'Yes': 0, 'No': 1}
        params['task_format'] = 'classification'
        params['num_tokens_to_predict'] = 1

    elif 'samsum' in params['dataset']:
        params['prompt_prefix'] = "Summarize the dialogue.\n\n"
        params["q_prefix"] = "Dialogue: "
        params["a_prefix"] = "Summary: "
        params['label_dict'] = " "
        params['inv_label_dict'] = " "
        params['task_format'] = 'summarization'
        params['num_tokens_to_predict'] = 100 #This should be variable rather than fixed from3 to 300!

    elif 'docvqa' in params['dataset']:
        params['prompt_prefix'] = "Answer the question based on the context.\n\n"
        params["q_prefix"] = "Question: "
        params["a_prefix"] = "Answer: "
        params['label_dict'] = " "
        params['inv_label_dict'] = " "
        params['task_format'] = 'summarization'
        params['num_tokens_to_predict'] = 100

    elif 'mit-d' in params['dataset']:
        params['prompt_prefix'] = "" #"Based on the provided movie description or plot in the ‘content’ field, your task is to fill in the ‘label’ field with the name of the director of the movie. Your prediction should match with one of the directors represented in the 'content' field.\n\n"
        params["q_prefix"] = "Sentence: "
        params["a_prefix"] = "Director: "
        params['label_dict'] = " "
        params['inv_label_dict'] = " "
        params['task_format'] = 'summarization'
        params['num_tokens_to_predict'] = 5

    elif 'mit-g' in params['dataset']:
        params['prompt_prefix'] = "" #"Given a brief summary or plot of a movie in the 'content' field, your task is to fill in the 'label' field with the genre of the movie. Your prediction should align with one of the genres represented in the 'content' field.\n\n"
        params["q_prefix"] = "Sentence: "
        params["a_prefix"] = "Genre: "
        params['label_dict'] = " "
        params['inv_label_dict'] = " "
        params['task_format'] = 'summarization'
        params['num_tokens_to_predict'] = 5

    if 'samsum' in params['dataset'] or 'docvqa' in params['dataset'] or 'mit-d' in params['dataset'] or 'mit-g' in params['dataset']:
        return train_sentences, train_labels, test_sentences, test_labels, eval_sentences, eval_labels
    else:
        return train_sentences, train_labels, test_sentences, test_labels