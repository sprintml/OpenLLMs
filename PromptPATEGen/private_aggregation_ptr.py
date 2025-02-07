import json
import pprint
import sys

import evaluate
import nltk
import pandas as pd
from tqdm import tqdm

nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.util import ngrams
import string
import random
import openai
import time
import argparse
import pprint
from utils import *
from add_index import *

from prv_accountant.dpsgd import find_noise_multiplier
from scipy import stats
from priv_aggregation_utils import *

metric = evaluate.load("rouge")


# =================================================== private KSA with PTR ===========================================

def rearrange_json(original_json, num_teachers, num_queries):
    # This is to conform with DPICL paper i.e  the query response from each teacher are interleaved
    # Initialize the result dictionary
    result = {}

    # Iterate over the teachers and queries
    for i in range(num_teachers * num_queries):
        # Calculate the original index
        original_index = (i % num_teachers) * num_queries + i // num_teachers

        # Assign the query to the new index
        result[str(i)] = original_json[str(original_index)]

    return result


def change_refer_pred(dr, en):
    data_references = []
    for data_refer in dr:
        for i in range(en):
            data_references.append(data_refer)
    return data_references


def evaluate_performance_zero_teacher_predictions(student_label_path, four_shot_path, zero_shot_path, num_teachers,
                                                  num_queries, is_interleaved_teachers=False):
    with open(student_label_path, 'r') as f:
        student_label = json.load(f)
    with open(zero_shot_path, 'r') as f:
        zero_pred_f = json.load(f)
    zero_pred = []
    for i in range(len(zero_pred_f)):
        zero_pred.append(zero_pred_f[str(i)]['prediction'])
    with open(four_shot_path, 'r') as f:
        four_pred_f = json.load(f)

    if not is_interleaved_teachers:
        print("Interleaving teacher's predictions")
        four_pred_f = rearrange_json(four_pred_f, num_teachers, num_queries)
    # pprint.pprint(four_pred_f)

    four_pred = []
    for i in range(len(four_pred_f)):
        four_pred.append(four_pred_f[str(i)]['prediction'])

    fourshot_pred = four_pred

    zero_ref_label = change_refer_pred(student_label, 1)
    # four_ref_label = student_label * 100 #100 is the number of times to repeat the list. That is the number of teachers.
    four_ref_label = change_refer_pred(student_label, 100)
    # print("student_label", student_label)
    # print("zero_ref_label", zero_ref_label)
    # print("four_ref_label", four_ref_label)
    # print("four_pred", four_pred)

    print(
        "================ Non-private evaluation of Zero-shot prediction vs Four-shot prediction on student data=================")
    scores = metric.compute(predictions=zero_pred, references=zero_ref_label)
    print("zero-shot score:", scores)
    scores = metric.compute(predictions=four_pred, references=four_ref_label)
    print("four-shot score:", scores)


def private_aggregation_with_KSA_PTR(label_path_f, four_shot_path_f, zero_shot_path_f, num_teachers, num_queries,
                                     is_interleaved_teachers=False, args=None, privacy_params_f=None):

    with open(zero_shot_path_f, 'r', encoding='utf-8') as f:
        zero_pred_f = json.load(f)
    zero_pred = []
    for i in range(len(zero_pred_f)):
        zero_pred.append(zero_pred_f[str(i)]['prediction'])

    with open(four_shot_path_f, 'r') as f:
        four_pred_f = json.load(f)

    if not is_interleaved_teachers:
        four_pred_f = rearrange_json(four_pred_f, num_teachers, num_queries)

    four_pred = []
    for i in range(len(four_pred_f)):
        four_pred.append(four_pred_f[str(i)]['prediction'])

    with open(label_path_f, 'r') as f:
        label = json.load(f)

    fourshot_pred = four_pred

    # ds_size = 100  # test data size
    repeat = 1
    # ensemble = 100

    np.random.seed(args["seed"])
    random.seed(args["seed"])

    stopword_set = set(stopwords.words('english'))

    count_pass = 0
    gap_change = []
    print("len(fourshot_pred)", len(fourshot_pred))
    rouge1, rouge2, rougeL, rougeLsum = [], [], [], []
    for _ in range(repeat):

        # get predictions from json file
        final_pred = []
        final_raw_resp_test = []

        count_threshold_list = []
        for i in range(num_queries): #ds_size
            all_tokens = {}  # key: token, value: count
            for j in range(num_teachers): #ensemble
                sentence = fourshot_pred[i * num_teachers + j]
                tokens = nltk.word_tokenize(sentence)
                onegrams = set(ngrams(tokens, 1))
                # onegrams = set(onegrams)
                # making onegrams a set to avoid duplicate tokens
                for token in onegrams:
                    # only add one gram for one sentence
                    if token in all_tokens:
                        all_tokens[token] += 1
                    else:
                        all_tokens[token] = 1
            # print(all_tokens)
            all_tokens_sorted = sorted(all_tokens.items(), key=lambda x: x[1], reverse=True)
            # print(all_tokens_sorted)
            # ignore those non-words tokens
            filtered_tokens = {}
            for token, count in all_tokens_sorted:
                if not all(word in string.punctuation for word in token) and token[0] not in stopword_set:
                    filtered_tokens[token] = count
            filtered_tokens_sorted = sorted(filtered_tokens.items(), key=lambda x: x[1], reverse=True)
            # print("filtered_tokens_sorted", filtered_tokens_sorted)
            # print(filtered_tokens)

            ### DP version (RNM)
            gap_lst = []
            for k in range(min(len(filtered_tokens_sorted) - 1, 30)):  # I assume we are only interested in k <= 30
                gap = filtered_tokens_sorted[k][1] - filtered_tokens_sorted[k + 1][1]
                # print("Gap", gap)
                gap_lst.append(gap)
                if k == len(filtered_tokens_sorted) - 2:
                    gap_lst.append(filtered_tokens_sorted[k + 1][1])

            gap_lst = np.array(gap_lst)
            # print("gap_list", gap_lst)
            if len(gap_lst) < 2:
                # Default to zeroshot if all the teacher predict the same thing and no noise can be added since no gap_list
                print('***FAIL TEST! DO Zero-shot Learning***')
                filtered_tokens = []
                print("label", len(label[i]), "zero", len(zero_pred_f[str(i)]['prediction']), "four",
                      len(fourshot_pred[i * num_teachers + 0]))

                pass_val = 0
            else:

                gap_max_prev = np.max(gap_lst)
                noisy_gap_lst = gap_lst + np.random.gumbel(0, 2 / privacy_params_f['eps'], len(gap_lst))
                # print("noisy_gap_lst", noisy_gap_lst)
                kstar = np.argmax(noisy_gap_lst)
                # print("kstar", kstar)
                gap_max = np.max(noisy_gap_lst)
                # print("gap_max", gap_max)
                gap_change.append(gap_max_prev - gap_max)

                # print(count_threshold)
                filtered_tokens = dict(filtered_tokens_sorted)

                ### Non-private
                # filtered_tokens = [k[0] for k, v in filtered_tokens.items() if v >= actually_upper_bound]

                ### DP version (PTR)

                dk = gap_lst[kstar]  # get the value based on the added noise!
                print("dk = gap_lst[kstar]", dk)
                noise1 = np.random.normal(0, 2 * privacy_params_f['sigma'])
                print("noise1", noise1)
                noise2 = stats.norm.isf(privacy_params_f['delta0'], loc=0, scale=2 * privacy_params_f['sigma'])
                print("noise2", noise2)
                dkhat = max(2, dk) + noise1 - noise2
                print('dk={}, noise1={}, noise2={}, dkhat={}'.format(dk, noise1, noise2, dkhat))
                if dkhat > 2:
                    print("label", len(label[i]), "zero", len(zero_pred_f[str(i)]['prediction']), "four",
                          len(fourshot_pred[i * num_teachers + 0]))
                    print('***PASS TEST! RELEASE EXACT TOP-{} TOKENS***'.format(kstar))
                    filtered_tokens = [k[0] for k, v in filtered_tokens.items()]
                    filtered_tokens = filtered_tokens[:kstar]
                    count_pass += 1
                    pass_val = 1
                else:
                    print('***FAIL TEST! DO Zero-shot Learning***'.format(kstar))
                    filtered_tokens = []
                    print("label", len(label[i]), "zero", len(zero_pred_f[str(i)]['prediction']), "four",
                          len(fourshot_pred[i * num_teachers + 0]))

                    pass_val = 0

            # print(filtered_tokens)
            if filtered_tokens == []:
                final_prompt = zero_pred_f[str(i)]['origin_prompt']
            else:
                random.shuffle(filtered_tokens)  # shuffle the list of tokens
                prompt = '['
                for token in filtered_tokens:
                    prompt += token + ', '
                prompt = prompt[:-2] + ']'
                # prompt = prompt + '\nThe summary is:'

                if args["dataset"] == "mit-d":
                    prompt = prompt + '\nThe name of the movie director is:'
                    zero_shot_sentence = "Derive the full name of the movie director from the 'Sentence' using the word suggestions in the 'Director' field. The name of the director must be from the sentence field.\n\n" + zero_pred_f[str(i)]['origin_prompt']
                elif args["dataset"] == "mit-g":
                    prompt = prompt + '\nThe movie genre is:'
                    zero_shot_sentence = "Derive the genre of the movie from the 'Sentence' using the word suggestions in the 'Genre'. The name of the genre must be from the sentence field.\n\n" + zero_pred_f[str(i)]['origin_prompt']
                else:
                    zero_shot_sentence = zero_pred_f[str(i)]['origin_prompt'].replace('Summarize the dialogue.',
                                                                                      'Summarize the dialogue with the word suggestions in the "Summary".')
                final_prompt = zero_shot_sentence + prompt
            # if i < 5:
            #     print(final_prompt)

            print("final_prompt", final_prompt)
            ################################################################################
            ##### Please check your code and run this api to complete the experiment'#######
            ################################################################################

            raw_resp_test = get_model_response(args, None, None, None,
                                               override_prompt=final_prompt)

            # add pass or fail to the raw predictions
            # Pass it to final_raw_resp_test

            raw_resp_test["pass"] = pass_val
            # print("raw_resp_test", raw_resp_test)
            pred = raw_resp_test["prediction"]
            # print("pred", pred)

            # pred = openai.Completion.create(
            #     engine="text-davinci-003",
            #     prompt=final_prompt,
            #     temperature=0,
            #     max_tokens=256,
            #     top_p=1.0,
            #     frequency_penalty=0.0,
            #     presence_penalty=0.0
            # )

            final_pred.append(pred)
            final_raw_resp_test.append(raw_resp_test)

        score = metric.compute(predictions=final_pred, references=label)
        rouge1.append(score['rouge1'])
        rouge2.append(score['rouge2'])
        rougeL.append(score['rougeL'])
        rougeLsum.append(score['rougeLsum'])

    print("================ Private evaluation of Four-shot prediction on student data=================")
    print("rouge1's mean: ", np.mean(np.array(rouge1)), "rouge1's std: ", np.std(np.array(rouge1)))
    print("rouge2's mean: ", np.mean(np.array(rouge2)), "rouge2's std: ", np.std(np.array(rouge2)))
    print("rougeL's mean: ", np.mean(np.array(rougeL)), "rougeL's std: ", np.std(np.array(rougeL)))
    print("rougeLsum's mean: ", np.mean(np.array(rougeLsum)), "rougeLsum's std: ", np.std(np.array(rougeLsum)))

    print("Amount that pass PTR:", count_pass)
    print("Mean Gap change", np.mean(np.array(gap_change)))

    # ***PASS TEST! RELEASE EXACT TOP-3 TOKENS***
    # rouge1's mean:  0.38549113959158926 rouge1's std:  0.004715646681797527
    # rouge2's mean:  0.1442157014406559 rouge2's std:  0.005443081525002633
    # rougeL's mean:  0.2958618078347388 rougeL's std:  0.0045467990552312655
    # rougeLsum's mean:  0.29607762247992897 rougeLsum's std:  0.0043788640926342645
    return final_raw_resp_test, final_pred


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument('--teacher_predictions', type=str, required=True,
                        help='txt file containing teacher_predictions on the test set')
    parser.add_argument('--zero_shot_predictions', type=str, required=True,
                        help='txt file containing zero_shot predicitons on the test set')
    parser.add_argument('--seed', required=True, help='num seeds for the training set', type=int)
    parser.add_argument('--true_labels', type=str, required=True,
                        help='txt file containing the true labels of test set.')
    parser.add_argument('--public_labels_from_teachers_filename', type=str, required=True,
                        help='filename for saving the private labels of test set.')
    parser.add_argument('--num_teachers', type=int, help='number of teachers')
    parser.add_argument('--num_student_query', type=int, help='number of query prompts')
    parser.add_argument('--is_teacher_predictions_interleaved', action='store_true')
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    parser.add_argument('--model', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--num_tokens_to_predict', type=int, required=True,
                        help='Number of token the model should predict')
    parser.add_argument('--api_num_log_prob', dest='api_num_log_prob', action='store', required=False, type=int,
                        default=100,
                        help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--epsilon', type=int, help='Final Epsilon. Currently supports 1, 3, or 8')

    args = parser.parse_args()
    args = vars(args)  # To avoid conflit with chunck size function

    # Because the prompt is already self-contained!
    args['prompt_prefix'] = ""
    args["q_prefix"] = ""
    args["a_prefix"] = ""
    args['task_format'] = 'summarization'
    args[
        'prediction_mode'] = 'None'  # To show that this is 'zeroshot' with no formatting if 'None', it means it's zeroshot but has its format already

    delta = 5e-5
    prob = 400 / 14732  # 400 = 4 private examples x 100 ensemble of teachers
    
    # here you need to check and let the output of compose_subsampled_EMandPTR_to_approxDP
    # to be less than your target epsilon

    print("print(delta / prob)", delta / prob)

    privacy_params = None

    if args["epsilon"] == 8:
        privacy_params = {
            'eps': 1,
            'delta0': 0.99 * delta / prob,
            'sigma': 1.15046,
            'prob': prob,
            'niter': 100
        }  # 91
        final_epsilon = compose_subsampled_EMandPTR_to_approxDP(privacy_params, delta=delta)
        print("final_epsilon", final_epsilon)  # Decomposes to epsilon of 8

    elif args["epsilon"] == 3:
        privacy_params = {
            'eps': 0.9,
            'delta0': 0.99 * delta / prob,
            'sigma': 1.869,
            'prob': prob,
            'niter': 100
        }  # 79

        final_epsilon = compose_subsampled_EMandPTR_to_approxDP(privacy_params, delta=delta)
        print("final_epsilon", final_epsilon)  # Decomposes to epsilon of 3

    elif args["epsilon"] == 1:
        privacy_params = {
            'eps': 0.45,
            'delta0': 0.8 * delta / prob,
            'sigma': 2.62,
            'prob': prob,
            'niter': 100
        }  # 57

        print(privacy_params['delta0'], privacy_params['eps'])
        final_epsilon = compose_subsampled_EMandPTR_to_approxDP(privacy_params, delta=delta)
        print("final_epsilon", final_epsilon)  # Decomposes to epsilon of 1
    else:
        sys.exit(f"Supported epsilon 1, 3 or 8. But you input {args['epsilon']}, which is not supported")

    evaluate_performance_zero_teacher_predictions(args["true_labels"] + "_" + str(args["seed"]),
                                                  args["teacher_predictions"] + "_" + str(args["seed"]),
                                                  args["zero_shot_predictions"] + "_" + str(args["seed"]),
                                                  args["num_teachers"], args["num_student_query"],
                                                  args["is_teacher_predictions_interleaved"])
    responses, predictions = private_aggregation_with_KSA_PTR(args["true_labels"] + "_" + str(args["seed"]),
                                                              args["teacher_predictions"] + "_" + str(args["seed"]),
                                                              args["zero_shot_predictions"] + "_" + str(args["seed"]),
                                                              args["num_teachers"], args["num_student_query"],
                                                              args["is_teacher_predictions_interleaved"], args, privacy_params)
    # Save to file
    write_data_from_list_dict_to_file(responses, args["public_labels_from_teachers_filename"] + "_" + str(args["seed"])+ "_eps" + str(args["epsilon"]))

    # Here, we can just write them to a file already?
    # Then we select the best ones?
