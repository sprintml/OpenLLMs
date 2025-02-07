import argparse
from data_utils import load_dataset_hf, load_dataset_local
from utils import *
import os
import tqdm
# ROOT_DIRECTORY = os.getcwd()
import evaluate

def main(args):
    """
    Run experiment or load past results, print accuracy
    """
    save_results(args)

def save_results(args):
    """
    Run the model and save its responses and the rest of configs into a pickle file
    """
    np.random.seed(args['seed'])

    # print("\nExperiment name:", args['expr_name'])
    print(args)

    ### load data

    if args['dataset'] == 'samsum' or args['dataset'] == 'docvqa' or args['dataset'] == 'mit-d' or args['dataset'] == 'mit-g':
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels, all_eval_sentences, all_eval_labels = load_dataset_hf(args)
    else:
        all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = load_dataset_hf(
            args)

    # all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = load_dataset_local(args)
    if args["task_format"] == "classification":
        args_check(args)

    ### sample test set
    if args['subsample_test_set'] is None:
        test_sentences, test_labels = all_test_sentences, all_test_labels
        print(f"selecting full test set ({len(all_test_labels)} examples)")
    else:
        test_sentences, test_labels, _ = random_sampling(all_test_sentences, all_test_labels, args['subsample_test_set'], indices=[])
        print(f"selecting {len(test_labels)} subsample of test set")
        print("test_labels", test_labels)

    ### sample few-shot training examples
    teacher_sentences = []
    teacher_labels = []
    indices = []
    content_free_inputs = ["N/A", "", "[MASK]"]

    rouge1, rouge2, rougeL, rougeLsum = [], [], [], []

    with open(args['save_path'] + "_"+str(args["seed"]), 'w') as file:
        for _ in tqdm.tqdm(range(args['num_teachers'])):
            train_sentences, train_labels, idxs = random_sampling(all_train_sentences, all_train_labels, args['num_shots'], indices)
            teacher_sentences.append(train_sentences)
            teacher_labels.append(train_labels)
            for idx in idxs :
                indices.append(idx)

            # test response for each n-shots teacher
            raw_resp_test = get_model_response(args, train_sentences, train_labels, test_sentences) # Get model response

            if args["task_format"] == "summarization":
                # print("raw_resp_test", raw_resp_test)  # Note that this will be list of dicts
                for dict_item in raw_resp_test:
                    file.write(json.dumps(dict_item) + '\n')

            else:
                # get prob for each label
                all_label_probs = get_label_probs(args, raw_resp_test, train_sentences, train_labels, test_sentences)

                # NOTE : this step can be ignored in the pate process

                p_cf = get_p_content_free(args, train_sentences, train_labels, content_free_inputs=content_free_inputs)
                list_labels1, acc_original = eval_accuracy(all_label_probs, test_labels)
                list_labels2, acc_calibrated = eval_accuracy(all_label_probs, test_labels, mode="diagonal_W", p_cf=p_cf)
                accuracies = [acc_original, acc_calibrated]

                if args['enable_evaluation'] :
                    print(f"Accuracies: {accuracies}")
                    print(f"p_cf      : {p_cf}")

                # write results into .txt file
                # line = " ".join([str(label) for label in list_labels1])
                # file.write(line + "\n")
                if acc_original > acc_calibrated :
                    line = " ".join([str(label) for label in list_labels1])
                    file.write(line + "\n")
                else :
                    line = " ".join([str(label) for label in list_labels2])
                    file.write(line + "\n")


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
        label_probs = label_probs / np.sum(label_probs) # normalize to 1

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
                position = (i, j) # (which test example, which label)
                all_missing_positions.append(position)
        all_label_probs.append(label_probs)
    all_label_probs = np.array(all_label_probs) # prob not normalized

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
            resp = complete(chunk, 0, args['model'], echo=True, num_log_probs=1, data_name=args["dataset"])
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
    return all_label_probs # NOT NORMALIZED

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
                prob += np.exp(complete(prompt + " " + a, 0, args['model'], echo=True, num_log_probs=1)['choices'][0]['logprobs']['token_logprobs'][-1])
            p_y[i] = prob
        all_p_y.append(p_y)

    p_y = np.mean(np.array(all_p_y), axis=0)
    p_y = p_y / np.sum(p_y) # normalize
    return p_y


def args_check(args):
    """sanity check the experiment args"""
    assert args['num_tokens_to_predict'] == 1
    # for classification, make sure that all of the class names are one word.
    for key, label_names in args['label_dict'].items():
        for label_id, label_name in enumerate(label_names):
            first_token_of_label_name = complete(' ' + label_name, 1, args['model'], echo=True, num_log_probs=2)['choices'][0]['logprobs']['tokens'][0]
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
    parser.add_argument('--task_format', required=False, help='name of task either classification or summarization')
    parser.add_argument('--num_shots', type=int, required=True, help='num training examples to use')
    # other arguments
    parser.add_argument('--subsample_test_set', dest='subsample_test_set', action='store', required=False, type=int,
                        default=None, help='size of test set to use to speed up eval. None means using all test set')
    parser.add_argument('--api_num_log_prob', dest='api_num_log_prob', action='store', required=False, type=int,
                        default=100, help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    # flags
    parser.add_argument('--use_saved_results', dest='use_saved_results', action='store_const', const=True, default=False,
                        help='whether to load the results from pickle files and not run the model')
    parser.add_argument('--approx', dest='approx', action='store_const', const=True, default=False,
                        help='whether to set token prob to zero if not in top 100')
    parser.add_argument('--num_teachers', type=int, help='number of teacher prompts to select in the training set')
    parser.add_argument('--enable_evaluation', type=bool, default=None)
    parser.add_argument('--num_token_to_predict', type=int, default=1)
    parser.add_argument('--conditioned_on_correct_classes', type=bool, default=True)
    parser.add_argument('--save_path', type=str, default='teacher_votes/sst2')

    args = parser.parse_args()
    args = vars(args)


    main(args)
