import argparse
import sys

from data_utils import load_dataset_hf, limit_tokens
from utils import *
import os
import pandas as pd
import tqdm
from datasets import load_dataset, load_from_disk, Dataset, concatenate_datasets, DatasetDict
import evaluate

metric = evaluate.load("rouge")

# ROOT_DIRECTORY = os.getcwd()

def main(args):
    """
    Run experiment or load past results, print accuracy
    """
    save_results(args)





def em_accuracy_helper(prediction, label):
    correctness_list = []
    # print("prediction prediction", prediction)
    # print("label", label)
    # Preprocess the predictions and labels
    pred_processed = [p.rstrip().rstrip('.').lower() for p in prediction]
    label_processed = [l.rstrip().rstrip('.').lower() for l in label]

    # print("pred_processed", pred_processed)
    # print("label_processed", label_processed)

    # Compute accuracy
    correct_predictions = sum(p == l for p, l in zip(pred_processed, label_processed))
    accuracy = correct_predictions / len(prediction)
    accuracy = accuracy * 100
    print(f"Accuracy: {accuracy}%")
    return accuracy

    # for pred, l in zip(prediction, label):
    #     pred = pred.split("\n")[0]
    #     # Remove trailing full stop
    #     pred = pred.rstrip().rstrip('.').lower()
    #     print("pred", pred)
    #     if pred == l:
    #         correctness_list.append(1)
    #     else:
    #         correctness_list.append(0)
    # return np.mean(correctness_list)

def create_query(file_name, consider_pass, num_items, data_name):
    # Load the data from the file
    with open(file_name, 'r') as f:
        data = json.load(f)

    # Convert the dictionary items to a list
    data_items = list(data.items())

    print("Publicly labelled data", data_items)

    # If we should consider the "pass" value, filter the items where "pass" equals 1
    if consider_pass:
        print("Only consider examples that passed PTR")
        data_items = [(key, value) for key, value in data_items if value.get('pass') == 1]

    # Slice the list to only include the first num_items items
    data_items = data_items[:num_items]

    # Initialize the final query with "Summarize the dialogue."
    if data_name == "samsum":
        final_query = "Summarize the dialogue."
    elif data_name == "docvqa":
        final_query = "Answer the question based on the context."
    elif data_name == "mit-d":
        final_query = "" #"Based on the provided movie description or plot in the ‘content’ field, your task is to fill in the ‘label’ field with the name of the director of the movie. Your prediction should match with one of the directors represented in the 'content' field."
    elif data_name == "mit-g":
        final_query = "" #"Given a brief summary or plot of a movie in the 'content' field, your task is to fill in the 'label' field with the genre of the movie. Your prediction should align with one of the genres represented in the 'content' field."


    # Iterate over the sliced list of items
    for key, value in data_items:
        if data_name == "samsum":
            # Remove "Summarize the dialogue." or in the case of accepted PTR "Summarize the dialogue with the word suggestions in the "Summary"." and everything after "\nSummary:"
            dialogue = value['origin_prompt'].split('\nSummary:')[0].replace('Summarize the dialogue.', '').replace('Summarize the dialogue with the word suggestions in the "Summary".\n\n', '').replace('', '')
            # Add the dialogue and the prediction to the final query
            final_query += f"\n\n{dialogue} \nSummary: {value['prediction']}"
        elif data_name == "mit-d":
            content = value['origin_prompt'].split('\nDirector:')[0].replace("Derive the full name of the movie director from the 'Sentence' using the word suggestions in the 'Director' field. The name of the director must be from the sentence field.", "")
            #.replace( "Based on the provided movie description or plot in the 'Content' field, your task is to fill in the 'Label' field with the name of the director of the movie using the word suggestions in the 'Label' field. Your prediction should match with one of the directors represented in the 'Content' field.\n\n", '')
            # final_query += f"\n\n{content} \nLabel: {value['prediction']}"
            final_query += f"{content} \nDirector: {value['prediction']}" # \nThe name of the movie director is:
        elif data_name == "mit-g":
            content = value['origin_prompt'].split('\nGenre:')[0].replace("Derive the genre of the movie from the 'Sentence' using the word suggestions in the 'Genre'. The name of the genre must be from the sentence field.", "")
            #.replace("Given a brief summary or plot of a movie in the 'Content' field, your task is to fill in the 'label' field with the genre of the movie using the word suggestions in the 'Label' field. Your prediction should align with one of the genres represented in the 'content' field.\n\n", '')

            # final_query += f"\n\n{content} \nLabel: {value['prediction']}"
            final_query += f"{content} \nGenre: {value['prediction']}" # \nThe movie genre is:
        elif data_name == "docvqa":
            question = value['origin_prompt'].split('\nAnswers:')[0].replace('Answer the question based on the context.', '').replace('Answer the question based on the context using the word suggestion in the "Answers" field.\n\n', '').replace('', '')
            # Add the dialogue and the prediction to the final query
            final_query += f"\n\n{question} \nAnswers: {value['prediction']}"


    return final_query

# Usage:
# print(create_query('your_file.json', consider_pass=True))  # Only consider items where "pass" is 1
# print(create_query('your_file.json', consider_pass=False))  # Consider all items


def save_results(args):
    """
    Run the model and save its responses and the rest of configs into a pickle file
    """
    print(args)

    ### load data

    # load test labels from teachers votes
    if args['dataset'] == "samsum" or args['dataset'] == "docvqa" or args['dataset'] == "mit-d" or args['dataset'] == "mit-g":
        _, _, _, _, all_test_sentences, all_test_labels = load_dataset_hf(args)
    else:
        _, _, all_test_sentences, all_test_labels = load_dataset_hf(args)

    if args['dataset'] == "samsum" or args['dataset'] == "docvqa" or args['dataset'] == "mit-d" or args['dataset'] == "mit-g":
        np.random.seed(0)  #We fix the test set throughout the experiment

        if args['dataset'] == "samsum":
            dataset = load_dataset(args["dataset"])
            test_sentences, test_labels, _ = random_sampling(dataset['test']['dialogue'],
                                                             dataset['test']['summary'], args["subsample_test_set"], []) #100
        elif args['dataset'] == "mit-d":
            dataset = DatasetDict.load_from_disk("./datasets/movie_data/Director/hf_dataset")
            test_sentences, test_labels = dataset['test']['content'], dataset['test']['label']

        elif args['dataset'] == "mit-g":
            dataset = DatasetDict.load_from_disk("./datasets/movie_data/Genre/hf_dataset")
            test_sentences, test_labels = dataset['test']['content'], dataset['test']['label']

        elif args['dataset'] == "docvqa":
            dataset = DatasetDict.load_from_disk("./datasets/DocVQA/hf_format_datasets/DocVQA_client0123")
            # Apply the function to each example in the dataset
            dataset['test'] = dataset['test'].map(limit_tokens)

            # Limit the number of tokens in the context
            # dataset['test']['contexts'] = [' '.join(tokenizer.tokenize(c)[:800]) for c in dataset['test']['contexts']]

            test_sentences, test_labels, _ = random_sampling([f'{q} \nContext: {c}' for q, c in zip(dataset['test']['questions'], dataset['test']['contexts'])],
                                                             dataset['test']['answers'], args["subsample_test_set"], []) #10000


        # import private_labels.txt
        # Take the output of each

        # Select the good ones i.e the ones that passed PTR if use_pass_only_ptr_examples is true and use all 100 public examples if False

        final_student_prompt = create_query(args["public_labels_from_teachers_filename"]+"_"+str(args["seed"])+ "_eps" + str(args["epsilon"]),
                                            args["use_pass_only_ptr_examples"], args["num_student_prompts"], args["dataset"])

        rouge1, rouge2, rougeL, rougeLsum = [], [], [], []

        final_pred = []
        final_raw_resp_test = []

        for query in test_sentences:
            if args['dataset'] == "samsum":
                current_prompt = final_student_prompt + "\n\nDialogue: " + query + "\nSummary: "
            elif args['dataset'] == "docvqa":
                current_prompt = final_student_prompt + "\n\nQuestions: " + query + "\nAnswers: "
            elif args['dataset'] == "mit-d":
                current_prompt = final_student_prompt + "\n\nSentence: " + query + "\nDirector: \nThe name of the movie director is:"
            elif args['dataset'] == "mit-g":
                current_prompt = final_student_prompt + "\n\nSentence: " + query + "\nGenre:  \nThe movie genre is:"

            print("final_student_prompt", current_prompt)

            raw_resp_test = get_model_response(args, None, None, None,
                                           override_prompt=current_prompt)

            pred = raw_resp_test["prediction"]
            # print("pred", pred)

            final_pred.append(pred)
            final_raw_resp_test.append(raw_resp_test)


        # Open the file in read mode
        with open(args["eval_labels_filename"]+"_"+str(args["seed"]), 'r') as file:
            # Read the contents of the file
            true_eval_labels = file.read()
            # Use json.loads to convert the contents to a list
            true_eval_labels = json.loads(true_eval_labels)

        print("true_eval_labels", len(true_eval_labels))


        if args["dataset"] == "mit-d" or args["dataset"] == "mit-g":
            # Compute Accuracy
            N = len(final_pred)
            # correct = 0
            # for (p, t) in zip(final_pred, true_eval_labels):
            #     if p.lower() == t.lower():
            #         correct += 1
            #     # else:
            #     #     print(f"{p} = {t}")

            acc = em_accuracy_helper(final_pred, true_eval_labels)
            print(f"Accuracy: {acc}% of {N} samples")

        else:
            # Compute Rouge
            score = metric.compute(predictions=final_pred, references=true_eval_labels)
            rouge1.append(score['rouge1'])
            rouge2.append(score['rouge2'])
            rougeL.append(score['rougeL'])
            rougeLsum.append(score['rougeLsum'])

            print("rouge1's mean: ", np.mean(np.array(rouge1)), "rouge1's std: ", np.std(np.array(rouge1)))
            print("rouge2's mean: ", np.mean(np.array(rouge2)), "rouge2's std: ", np.std(np.array(rouge2)))
            print("rougeL's mean: ", np.mean(np.array(rougeL)), "rougeL's std: ", np.std(np.array(rougeL)))
            print("rougeLsum's mean: ", np.mean(np.array(rougeLsum)), "rougeLsum's std: ", np.std(np.array(rougeLsum)))


    else:

        public_labels = pd.read_csv(args['public_labels_teachers'], header=None, names=['label'])['label'].tolist()
        public_indexes = pd.read_csv(args['public_indexes'], header=None, names=['idx'])['idx'].tolist()

        ### sample test set
        np.random.seed(0)  #We fix the test set throughout the experiment
        test_sentences, test_labels, _ = random_sampling(all_test_sentences, all_test_labels,
                                                         args['subsample_test_set'], indices=[])
        print(f"selecting {len(test_labels)} subsample of test set")
        print(test_labels)

        ### Choosing only the correct indexes
        student_test_sentences = [test_sentences[idx] for idx in public_indexes]

        ### sample few-shot training examples
        np.random.seed(args['seed'])
        content_free_inputs = ["N/A", "", "[MASK]"]
        total_accuracies = []
        for i in tqdm.tqdm(range(len(student_test_sentences))):

            # taking student prompt as shots for evaluation
            train_sentences = [student_test_sentences[i]]
            train_labels = [public_labels[i]]

            # test response for each n-shots teacher
            raw_resp_test = get_model_response(args, train_sentences, train_labels, test_sentences)

            # get prob for each label
            all_label_probs = get_label_probs(args, raw_resp_test, train_sentences, train_labels, test_sentences)

            p_cf = get_p_content_free(args, train_sentences, train_labels, content_free_inputs=content_free_inputs)
            list_labels1, acc_original = eval_accuracy(all_label_probs, test_labels)
            list_labels2, acc_calibrated = eval_accuracy(all_label_probs, test_labels, mode="diagonal_W", p_cf=p_cf)
            accuracies = [acc_original, acc_calibrated]
            total_accuracies.append(acc_original)
            total_accuracies.append(acc_calibrated)

            if args['enable_evaluation']:
                print(f"Accuracies: {accuracies}")
                print(f"p_cf      : {p_cf}")

        # NOTE : the max might be on calibrated and thus false regarding standard querying of the closed llm
        print('Accuracy of best student prompt (max also of calibrated and original accuracies)', max(total_accuracies))

        # print_results(result_tree)


def eval_accuracy(all_label_probs, test_labels, mode=None, p_cf=None):
    # print('all_label_probs', all_label_probs)
    # print('test_labels', test_labels)
    # evaluate the accuracy with and without contextual calibration
    num_classes = all_label_probs.shape[1]
    if p_cf is None:
        # do not calibrate
        W = np.identity(num_classes)
        b = np.zeros([num_classes, 1])
    else:
        # calibrate
        if mode == "diagonal_W":
            W = np.linalg.inv(np.identity(num_classes) * p_cf)
            b = np.zeros([num_classes, 1])
        elif mode == "identity_W":
            W = np.identity(num_classes)
            b = -1 * np.expand_dims(p_cf, axis=-1)
        else:
            assert False

    correctness_list = []
    assert len(all_label_probs) == len(test_labels)
    labels = []
    for label_probs, true_label in zip(all_label_probs, test_labels):
        label_probs = label_probs / np.sum(label_probs)  # normalize to 1

        calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b
        # print(calibrate_label_probs)
        ans_label = np.argmax(calibrate_label_probs)
        labels.append(ans_label)
        if ans_label == true_label:
            correctness_list.append(1)
        else:
            correctness_list.append(0)

    return labels, np.mean(correctness_list)


def get_label_probs(args, raw_resp, train_sentences, train_labels, test_sentences):
    """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
    num_classes = len(args['label_dict'])
    approx = args['approx']
    assert len(raw_resp) == len(test_sentences)

    # Fill in the labels that is in the top k prob
    all_label_probs = []
    all_missing_positions = []
    for i, ans in enumerate(raw_resp):
        top_logprobs = ans['logprobs']['top_logprobs'][0]  # [0] since we only ask for complete one more token
        label_probs = [0] * len(args['label_dict'].keys())
        for j, label_list in args['label_dict'].items():
            all_found = True
            for label in label_list:  # each possible label correspond to the same class
                label = " " + label  # notice prompt does not have space after 'A:'
                if label in top_logprobs:
                    label_probs[j] += np.exp(top_logprobs[label])
                else:
                    all_found = False
            if not all_found:
                position = (i, j)  # (which test example, which label)
                all_missing_positions.append(position)
        all_label_probs.append(label_probs)
    all_label_probs = np.array(all_label_probs)  # prob not normalized

    # Fill in the label probs that are NOT in top k probs, by asking the model to rate perplexity
    # This helps a lot in zero shot as most labels wil not be in Top 100 tokens returned by LM
    if (not approx) and (len(all_missing_positions) > 0):
        print(f"Missing probs: {len(all_missing_positions)}/{len(raw_resp) * num_classes}")
        all_additional_prompts = []
        num_prompts_each = []
        for position in all_missing_positions:
            which_sentence, which_label = position
            test_sentence = test_sentences[which_sentence]
            label_list = args['label_dict'][which_label]
            for label in label_list:
                prompt = construct_prompt(args, train_sentences, train_labels, test_sentence)
                prompt += " " + label
                all_additional_prompts.append(prompt)
            num_prompts_each.append(len(label_list))

        # chunk the prompts and feed into model
        chunked_prompts = list(chunks(all_additional_prompts, chunk_size_helper(args)))
        all_probs = []
        for chunk_id, chunk in enumerate(chunked_prompts):
            resp = complete(chunk, 0, args['model'], echo=True, num_log_probs=1)
            for ans in resp['choices']:
                prob = np.exp(ans['logprobs']['token_logprobs'][-1])
                all_probs.append(prob)

        assert sum(num_prompts_each) == len(all_probs)
        assert len(num_prompts_each) == len(all_missing_positions)

        # fill in corresponding entries in all_label_probs
        for index, num in enumerate(num_prompts_each):
            probs = []
            while num > 0:
                probs.append(all_probs.pop(0))
                num -= 1
            prob = np.sum(probs)
            i, j = all_missing_positions[index]
            all_label_probs[i][j] = prob

        assert len(all_probs) == 0, "all should be popped"
        assert (all_label_probs > 0).all(), "all should be populated with non-zero value"
    # print('shape teacher label probs', all_label_probs.shape)
    return all_label_probs  # NOT NORMALIZED


def get_p_content_free(args, train_sentences, train_labels, content_free_inputs=('N/A',)):
    """Query model with content free input, return its prediction probability for each label"""
    label_dict = args['label_dict']

    all_p_y = []
    for content_free_input in content_free_inputs:
        prompt = construct_prompt(args, train_sentences, train_labels, content_free_input)

        p_y = [0] * len(label_dict)
        for i, answers in label_dict.items():
            prob = 0
            for a in answers:
                prob += np.exp(
                    complete(prompt + " " + a, 0, args['model'], echo=True, num_log_probs=1)['choices'][0]['logprobs'][
                        'token_logprobs'][-1])
            p_y[i] = prob
        all_p_y.append(p_y)

    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y)  # normalize
    return p_y


def args_check(args):
    """sanity check the experiment args"""
    assert args['num_tokens_to_predict'] == 1
    # for classification, make sure that all of the class names are one word.
    for key, label_names in args['label_dict'].items():
        for label_id, label_name in enumerate(label_names):
            first_token_of_label_name = \
            complete(' ' + label_name, 1, args['model'], echo=True, num_log_probs=2)['choices'][0]['logprobs'][
                'tokens'][0]
            if first_token_of_label_name[1:] != label_name:
                print('label name is more than 1 token', label_name)
                assert False

    if not (args['dataset'] in ['cb', 'rte']):
        # formatting: there should be a space after question/answer prefix
        assert args["q_prefix"][-1] == " "
        assert args["a_prefix"][-1] == " "
        assert len(args["prompt_prefix"]) == 0 or args["prompt_prefix"][-2:] == '\n\n'


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--model', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--dataset', required=True, help='name of dataset(s), e.g., agnews')
    parser.add_argument('--seed', required=True, help='num seeds for the training set', type=int)
    # parser.add_argument('--num_shots', type=int, required=True, help='num training examples to use')
    # other arguments
    parser.add_argument('--subsample_test_set', dest='subsample_test_set', action='store', required=False, type=int,
                        default=None, help='size of test set to use to speed up eval. None means using all test set')
    parser.add_argument('--api_num_log_prob', dest='api_num_log_prob', action='store', required=False, type=int,
                        default=100,
                        help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    parser.add_argument('--public_labels_from_teachers_filename', type=str, help='Where the query with their private labels are saved')
    parser.add_argument('--eval_labels_filename', type=str, help='Groundtruth Test labels')
    parser.add_argument('--use_pass_only_ptr_examples', action='store_true')
    parser.add_argument('--num_tokens_to_predict', type=int, required=True,
                        help='Number of token the model should predict')
    parser.add_argument('--num_student_prompts', type=int, required=True,
                        help='Number of examples to use as context for generating the summary')
    parser.add_argument('--epsilon', type=int, help='Final Epsilon. Currently supports 1, 3, or 8')

    args = parser.parse_args()
    args = vars(args)


    if args['epsilon'] not in [1, 3, 8]:
        sys.exit(f"Supported epsilon 1, 3 or 8. But you input {args['epsilon']}, which is not supported")


    # Because the prompt is already self-contained!
    args['prompt_prefix'] = ""
    args["q_prefix"] = ""
    args["a_prefix"] = ""
    args['task_format'] = 'summarization'
    args[
        'prediction_mode'] = 'None'  # To show that this is 'zeroshot' with no formatting if 'None', it means it's zeroshot but has its format already


    # simple processing
    def convert_to_list(items, is_int=False):
        if is_int:
            return [int(s.strip()) for s in items.split(",")]
        else:
            return [s.strip() for s in items.split(",")]


    # args['models'] = convert_to_list(args['models'])
    # args['datasets'] = convert_to_list(args['datasets'])
    # args['all_shots'] = convert_to_list(args['all_shots'], is_int=True)

    main(args)
