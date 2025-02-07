import argparse
from data_utils import load_dataset_hf
from utils import *
from noisymax import *
import os
import tqdm
from collections import Counter

# Privacy accountants
from prv_accountant.dpsgd import find_noise_multiplier, DPSGDAccountant
from prv_accountant.privacy_random_variables import PoissonSubsampledGaussianMechanism
from prv_accountant import PRVAccountant
from autodp import rdp_acct, rdp_bank


def main(args):
    """
    Run the model and save its responses and the rest of configs into a pickle file
    """

    print(args)

    ### load data
    all_train_sentences, all_train_labels, all_test_sentences, all_test_labels = load_dataset_hf(args)

    ### sample test set
    np.random.seed(0) # constant test set for different hyperparameters sets
    test_sentences, test_labels, _ = random_sampling(all_test_sentences, all_test_labels, args['subsample_test_set'], indices=[])
    print(f"selecting test size : {args['subsample_test_set']} examples")

    ### sample few-shot training examples
    np.random.seed(args['seed'])
    
    ### computing of the dp parameters
    sampling_probability = 10*args['num_shots']/len(all_train_sentences)
    print(sampling_probability)
    num_steps = args['subsample_test_set']

    # NOTE : with autodp library (so rdp)
    acct = rdp_acct.anaRDPacct()
    gaussian = lambda x: rdp_bank.RDP_gaussian({'sigma': args['noise_multiplier']}, x)
    acct.compose_poisson_subsampled_mechanisms(gaussian, sampling_probability, coeff = len(test_sentences))
    print('dp privacy', acct.get_eps(args['target_delta']))

    print('noise multiplier', args['noise_multiplier'])
    
    content_free_inputs = ["N/A", "", "[MASK]"]
    max_accuracy = 0
    running_eps = 0
    labels = []
    for i in tqdm.tqdm(range(len(test_sentences))) :
        test_sentence = test_sentences[i]
        teacher_sentences = []
        teacher_labels = []
        indices = []
        # subsampling of 10 examples per query -> for a dataset of 5k samples does not change anything
        for _ in range(10) :
            train_sentences, train_labels, idxs = random_sampling(all_train_sentences, all_train_labels, args['num_shots'], indices)
            teacher_sentences.append(train_sentences)
            teacher_labels.append(train_labels)
            for idx in idxs :
                indices.append(idx)
        
        if 'claude' in args['model'] :
            ValueError('Anthropic models currently not supported')
        
        else :
            # test response for each n-shots teacher
            raw_resp_test = get_model_response(args, teacher_sentences, teacher_labels, test_sentence)

            # get prob for each label
            all_label_probs = get_label_probs(args, raw_resp_test, teacher_sentences, teacher_labels, test_sentence)

            # should just be compared to the particular test sentence
            voted_label = private_aggregation(all_label_probs, teacher_sentences, teacher_labels, args['noise_multiplier'],
                                              mode="diagonal_W", content_free_inputs=content_free_inputs)
            labels.append(voted_label)


    accuracy = eval_accuracy(labels, test_labels)
    print('final accuracy', accuracy)


def eval_accuracy(predictions, test_labels) :
    predictions = [int(pred) for pred in predictions]
    test_labels = [int(label) for label in test_labels]
    c = 0
    for i in range(len(predictions)) :
        if predictions[i] == test_labels[i] :
            c += 1
    return c/len(predictions)

def private_aggregation(all_label_probs, train_sentences, train_labels, noise_multiplier, mode=None, content_free_inputs=None):
    num_classes = all_label_probs.shape[1]
    labels_per_test_sample = []
    for label_probs, train_sentence, train_label in zip(all_label_probs, train_sentences, train_labels):
        label_probs = label_probs / np.sum(label_probs) # normalize to 1

        if 'gpt-4' in args['model'] :
            labels_per_test_sample.append(np.argmax(label_probs))
        else :
            # model calibration 
            p_cf = get_p_content_free(args, train_sentence, train_label, content_free_inputs=content_free_inputs)
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

            calibrate_label_probs = np.matmul(W, np.expand_dims(label_probs, axis=-1)) + b
            labels_per_test_sample.append(np.argmax(calibrate_label_probs))

    # histogram of the counts 
    counter = Counter(labels_per_test_sample)
    return noisy_max(counter, noise_multiplier, num_classes)

def get_label_probs(args, raw_resp, train_sentences, train_labels, test_sentence):
    """Obtain model's label probability for each of the test examples. The returned prob is NOT normalized"""
    num_classes = len(args['label_dict'])
    approx = args['approx']
    assert len(raw_resp) == len(train_sentences)

    # Fill in the labels that is in the top k prob
    all_label_probs = []
    all_missing_positions = []

    if 'gpt-4' in args['model'] :
        for i, ans in enumerate(raw_resp):
            top_logprobs = [elem.token.strip() for elem in ans.logprobs.content[0].top_logprobs]
            top_values = [elem.logprob for elem in ans.logprobs.content[0].top_logprobs]
            label_probs = [0] * len(args['label_dict'].keys())
            for j, label_list in args['label_dict'].items():
                all_found = True
                for label in label_list:  # each possible label correspond to the same class
                    if label in top_logprobs:
                        index = top_logprobs.index(label)
                        label_probs[j] += np.exp(top_values[index])
                    else:
                        all_found = False
                if not all_found:
                    position = (i, j) # (which test example, which label)
                    all_missing_positions.append(position)
            all_label_probs.append(label_probs)
        all_label_probs = np.array(all_label_probs)

    else :
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

    return all_label_probs # NOT NORMALIZED


# NOTE : calibration not possible for gpt4 API endpoint
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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # required arguments
    parser.add_argument('--model', required=True, help='name of model(s), e.g., GPT2-XL')
    parser.add_argument('--dataset', required=True, help='name of dataset(s), e.g., agnews')
    parser.add_argument('--seed', required=True, help='num seeds for the training set', type=int)
    parser.add_argument('--num_shots', type=int, required=True, help='num training examples to use')
    # other arguments
    parser.add_argument('--subsample_test_set', dest='subsample_test_set', action='store', required=False, type=int,
                        default=None, help='size of test set to use to speed up eval. None means using all test set')
    parser.add_argument('--api_num_log_prob', dest='api_num_log_prob', action='store', required=False, type=int,
                        default=20, help='number of top tokens to ask for when querying the model. Capped at 100 for OpenAI GPT-3 API')
    parser.add_argument('--bs', dest='bs', action='store', required=False, type=int, default=None,
                        help='batch size for model queries. For OpenAI API, capped at 20. For local running, set this to max out your GPU memory.')
    parser.add_argument('--approx', dest='approx', action='store_const', const=True, default=False,
                        help='whether to set token prob to zero if not in top 100')
    parser.add_argument('--num_teachers', type=int, help='number of teacher prompts to select in the training set')
    parser.add_argument('--num_token_to_predict', type=int, default=1)
    parser.add_argument('--conditioned_on_correct_classes', type=bool, default=True)
    parser.add_argument('--eps_inf', type=bool, default=None)
    #parser.add_argument('--target_eps', type=float)
    parser.add_argument('--target_delta', type=float)
    parser.add_argument('--noise_multiplier', type=float)
    args = parser.parse_args()
    args = vars(args)

    main(args)
