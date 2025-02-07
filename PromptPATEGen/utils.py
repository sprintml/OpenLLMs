import numpy as np
import time
from copy import deepcopy
import os
import sys
import torch
import pickle
import openai
import anthropic
import random
import json
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AutoModel, AutoTokenizer, AutoModelForCausalLM
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

ROOT_DIR = os.path.dirname(os.path.realpath(__file__)) 
SAVE_DIR = os.path.join(ROOT_DIR, 'saved_results')
if not os.path.isdir(SAVE_DIR):
    os.mkdir(SAVE_DIR)
    print(f"mkdir at {SAVE_DIR} for saving results")

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def chunk_size_helper(params):
    # Set the batch size (the size of the chunks determines the batch size). Default to 4 for GPT-2 and 20 for OpenAI if
    # no batch size is specified.
    print("params", params)
    bs = params['bs']
    if bs is None:
        if 'gpt2' in params['model']:
            return 1
        if 'vicuna7b' in params['model']:
            return 1
        if 'vicuna7b' in params['model']:
            return 1
        else:
            assert params['model'] in ['ada', "babbage-002", 'curie', 'davinci', 'ada-beta', 'babbage-beta', 'curie-beta', 'davinci-beta']
            return 20
    else:
        return bs

def random_sampling(sentences, labels, num, indices, return_index=True, must_include=-1):
    """randomly sample subset of the training pairs"""
    assert len(sentences) == len(labels)
    if num > len(labels):
        assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"
    boolean = False

    # NOTE : For non overlapping teachers shots 
    while boolean == False :
        idxs = np.random.choice(len(labels), size=num, replace=False)
        set_idxs = set(idxs)
        if not set_idxs.intersection(indices) :
            boolean = True

    selected_sentences = [sentences[i] for i in idxs]
    selected_labels = [labels[i] for i in idxs]
    if must_include > 0 and must_include not in idxs:
        n = random.randrange(num)
        #print("n is " + str(n))
        idxs[n] = must_include
        selected_sentences[n] = sentences[must_include]
        selected_labels[n] = labels[must_include]
    #print(selected_sentences)
    #print(selected_labels)
    #print("*******")

    # delete the selected training samples 
    
    if return_index:
        # print('selected sentence shape', selected_sentences.shape)
        return deepcopy(selected_sentences), deepcopy(selected_labels), idxs        
    else:
        return deepcopy(selected_sentences), deepcopy(selected_labels)

def random_sampling_mi(sentences, labels, num, include, must_idx, must_pos=None):
    """randomly sample subset of the training pairs"""
    assert len(sentences) == len(labels)
    if num > len(labels):
        assert False, f"you tried to randomly sample {num}, which is more than the total size of the pool {len(labels)}"
    idxs = np.random.choice(len(labels), size=num, replace=False)
    selected_sentences = [sentences[i] for i in idxs]
    selected_labels = [labels[i] for i in idxs]
    if include:
        if sentences[must_idx] not in selected_sentences:
            selected_sentences[must_pos] = sentences[must_idx]
            selected_labels[must_pos] = labels[must_idx]
    else:
        while sentences[must_idx] in selected_sentences:
            idxs = np.random.choice(len(labels), size=num, replace=False)
            selected_sentences = [sentences[i] for i in idxs]
            selected_labels = [labels[i] for i in idxs]

        
    return deepcopy(selected_sentences), deepcopy(selected_labels)


gpt2_model = None
gpt2_tokenizer = None
def setup_gpt2(model_name): #This can also take in other models not just gpt2
    # load the GPT-2 model
    global gpt2_model
    global gpt2_tokenizer
    if gpt2_model is None:
        if "vicuna7b" in model_name:
            print("Setting up vicuna model")
            gpt2_model = AutoModelForCausalLM.from_pretrained("lmsys/vicuna-7b-v1.5", torch_dtype=torch.bfloat16)
            gpt2_tokenizer = AutoTokenizer.from_pretrained("lmsys/vicuna-7b-v1.5")

        else:
            print("Setting up GPT-2 model")
            gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
            gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        gpt2_model.eval().cuda()

        # to batch generation, we pad on the left and mask those positions out.
        gpt2_tokenizer.padding_side = "left"
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_tokenizer.pad_token_id = gpt2_tokenizer.eos_token_id
        #gpt2_model.config.max_length = 2048
        gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id

        # Get the maximum input size
        max_input_size = gpt2_tokenizer.model_max_length
        print("max_input_size of the model", max_input_size)
        print("Finished")


# For vicuna model
model = None
tokenizer = None

def setup_model(model_name):
    # load the model
    global model
    global tokenizer
    if model is None:
        print("Setting up model")
        model = AutoModel.from_pretrained(model_name)
        model.eval().cuda()

        tokenizer = AutoTokenizer.from_pretrained(model_name)
        # to batch generation, we pad on the left and mask those positions out.
        tokenizer.padding_side = "left"
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = model.config.eos_token_id
        print("Finished")


# # Call the function with the Vicuna-7b model
# setup_model('vicuna-7b')



def setup_gpt3():
    # get OpenAI access key
    with open(os.path.join(ROOT_DIR, 'openai_key.txt'), 'r') as f:
        key = f.readline().strip()
        openai.api_key = key


def complete_gpt2(prompt, l=10, model_name='gpt2-xl', num_log_probs=None, echo=False, data_name=None):
    ''' This function runs GPT-2 locally but places the outputs into an json that looks just like the one
     provided by the OpenAI API. '''
    if isinstance(prompt, str):
        prompt = [prompt] # the code below assumes a list
        # print("Prompt here")

    # Add to prompt '\nThe summary is: ' to prompt. This is done in for the query. Sometimes the LLM does not understand!
    if data_name == "samsum":
        prompt[-1] = prompt[-1] +'\nThe summary is: '
        max_new_token = 150
    elif data_name == "docvqa":
        prompt[-1] = prompt[-1] +'\nThe answer is: '
        max_new_token = 5
    elif data_name == "mit-d" or data_name == "mit-g":
        prompt[-1] = prompt[-1] #+'\nThe label is: '
        max_new_token = 5
    # print("Our prompt", prompt) # all 4 examples + query
    input_ids = gpt2_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)

    # print(input_ids)

    # greedily generate l tokens
    if l > 0:
        # the generate function can handle left padded inputs automatically in HF
        # total_sequences is now the input + possible generated output
        # print(l + len(input_ids['input_ids'][0]))
        total_sequences = gpt2_model.generate(input_ids=input_ids['input_ids'].cuda(), attention_mask=input_ids['attention_mask'].cuda(), max_new_tokens=max_new_token, do_sample=True) #max_length=l + len(input_ids['input_ids'][0]),
        # print("total_sequences", total_sequences) #tokenIDs
        if 'vicuna7b' in model_name:
            decoded_response = gpt2_tokenizer.decode(total_sequences[0], skip_special_tokens=True)
            # get the response only
            prediction_ids = total_sequences[:, len(input_ids['input_ids'][0]):]
            predictions_only = gpt2_tokenizer.decode(prediction_ids[0], skip_special_tokens=True)

            print("total_sequences", decoded_response) #Text
            print("predictions_only", predictions_only)

            # Convert into DP-ICL format and write it to file!
            data = {
                'origin_prompt': prompt[0],
                'output': decoded_response,
                'prediction': predictions_only
            }
            # write_data_to_file('teacher_predictions/samsum/teacher_predictions.txt', data)
            print("\n\n\n\n")
    else:
        assert echo == True and l == 0
        total_sequences = input_ids['input_ids'].cuda()


    if 'vicuna7b' in model_name:
        return data
    else:


        # they want the probs of the top tokens
        if num_log_probs is not None:
            # we are left padding, so we need to adjust the position IDs
            attention_mask = (total_sequences != 50256).float()
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            # get the logits for the context and the next l tokens
            logits = gpt2_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids, return_dict=True).logits.detach().cpu()
            #print(logits.shape)
            #print(logits)
            if not echo:
                # get the top tokens and probs for the generated l tokens
                #probs = logits[:,-l-1:].cpu()
                # print(len(probs))
                probs = torch.softmax(logits[:,-l-1:], dim=2).cpu()
            else:
                # get the top tokens and probs for the context and the generated l tokens
                probs = torch.softmax(logits, dim=2).cpu()
            #print(logits[:,-l-1:].shape)
            top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
            logprobs = probs
            top_log_probs = top_probs
            #logprobs = torch.log(probs)
            #top_log_probs = torch.log(top_probs)

        # create the return value to resemble OpenAI
        return_json = {}
        choices = []
        for batch_id in range(len(prompt)):
            curr_json = {}
            # text is just the optional context and next l tokens
            if not echo:
                curr_json['text'] = gpt2_tokenizer.decode(total_sequences[batch_id][-l:], skip_special_tokens=True)
            else:
                curr_json['text'] = gpt2_tokenizer.decode(total_sequences[batch_id], skip_special_tokens=True)

            # fill the return json with the top tokens and probs to match the OpenAI return value.
            if num_log_probs is not None:
                curr_json['logprobs'] = {}
                curr_json['logprobs']['top_logprobs'] = []
                curr_json['logprobs']['token_logprobs'] = []
                curr_json['logprobs']['tokens'] = []
                if not echo:
                    # cutoff the -1 here because the probs are shifted one over for LMs
                    for current_element_top_log_probs, current_element_top_tokens in zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1]):
                        # tokens is a list of the top token at each position
                        curr_json['logprobs']['tokens'].append(gpt2_tokenizer.decode([current_element_top_tokens[0]]))
                        # token_logprobs is a list of the logprob of the top token at each position
                        curr_json['logprobs']['token_logprobs'].append(current_element_top_log_probs[0].item())
                        # top_logprobs is a list of dicts for the top K tokens. with each entry being {'token_name': log_prob}
                        temp = {}
                        for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                            temp[gpt2_tokenizer.decode(token.item())] = log_prob.item()
                        curr_json['logprobs']['top_logprobs'].append(temp)
                else:
                    # same as not above but small tweaks
                    # we add null to the front because for the GPT models, they have null probability for the first token
                    # (for some reason they don't have an beginning of sentence token)
                    curr_json['logprobs']['top_logprobs'].append('null')
                    # cutoff the -1 here because the probs are shifted one over for LMs
                    for index, (current_element_top_log_probs, current_element_top_tokens) in enumerate(zip(top_log_probs[batch_id][:-1], top_tokens[batch_id][:-1])):
                        # skip padding tokens
                        if total_sequences[batch_id][index].item() == 50256:
                            continue
                        temp = {}
                        for log_prob, token in zip(current_element_top_log_probs, current_element_top_tokens):
                            temp[gpt2_tokenizer.decode(token.item())] = log_prob.item()
                        curr_json['logprobs']['top_logprobs'].append(temp)
                    for index in range(len(probs[batch_id])):
                        curr_json['logprobs']['tokens'].append(gpt2_tokenizer.decode([total_sequences[batch_id][index]]))
                    curr_json['logprobs']['token_logprobs'].append('null')
                    for index, log_probs_token_position_j in enumerate(logprobs[batch_id][:-1]):
                        # probs are left shifted for LMs
                        curr_json['logprobs']['token_logprobs'].append(log_probs_token_position_j[total_sequences[batch_id][index+1]])

            choices.append(curr_json)
        return_json['choices'] = choices
        return return_json

def complete_gpt3(prompt, l, model_name, temp=0, num_log_probs=None, echo=False, n=None):
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.Completion.create(engine=model_name, prompt=prompt, max_tokens=l, temperature=temp,
                                                logprobs=num_log_probs, echo=echo, stop='\n', n=n)
            
            #print(response)
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False

            print("API error:", error)
            time.sleep(3)
    return response

def complete(prompt, l, model, temp=0, num_log_probs=None, echo=False, n=None, data_name=None):
    """complete the prompt using a language model"""
    assert l >= 0
    assert temp >= 0
    if 'gpt2' in model:
        assert n == None # unsupported at the moment
        assert temp == 0 # unsupported at the moment
        setup_gpt2(model)
        return complete_gpt2(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo)
    elif 'vicuna7b' in model:
        setup_gpt2(model) #Could use setup_model Merged with setup_gpt2
        return complete_gpt2(prompt, l=l, model_name=model, num_log_probs=None, echo=echo, data_name=data_name)

    else:
        setup_gpt3()
        return complete_gpt3(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo, n=n)

def construct_prompt(params, train_sentences, train_labels, test_sentence):
    """construct a single prompt to be fed into the model"""
    # special case when the user defines a custom prompt function. 
    if ('prompt_func' in params.keys()) and (params['prompt_func'] is not None):
        return params['prompt_func'](params, train_sentences, train_labels, test_sentence)

    # take the prompt template and fill in the training and test example
    prompt = params["prompt_prefix"]
    q_prefix = params["q_prefix"]
    a_prefix = params["a_prefix"]
    for s, l in zip(train_sentences, train_labels):
        prompt += q_prefix
        prompt += s + "\n"
        if isinstance(l, int) or isinstance(l, np.int32) or isinstance(l, np.int64): # integer labels for classification
            assert params['task_format'] == 'classification'
            l_str = params["label_dict"][l][0] if isinstance(params["label_dict"][l], list) else params["label_dict"][l]
        else:
            if params['task_format'] == 'summarization':
                l_str = l
            else:
                assert isinstance(l, str) # string labels
                assert params['task_format'] == 'qa'
                l_str = l

        prompt += a_prefix
        prompt += l_str + "\n\n"

    prompt += q_prefix
    prompt += test_sentence + "\n"
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
    # print("This is the prompt", prompt)
    # print("---------------")
    return prompt

def get_model_response(params, train_sentences, train_labels, test_sentences, return_all_prompts=False,
                       num_tokens_to_predict_override=None, override_prompt=None):
    """
    Obtain model's responses on test sentences, given the training examples
    :param params: parameters for the experiment
    :param train_sentences: few-shot training sentences
    :param train_labels: few-shot training labels
    :param test_sentences: few-shot test sentences
    :param return_all_prompts: whether to return all the prompts
    :param num_tokens_to_predict_override: whether to override num token to predict
    :param override_prompt: whether to override prompt
    :return: a list of dictionaries
    """
    all_raw_answers = []

    # can optionally ignore the normal prompt and feed in a custom prompt (used for contextual calibration)
    if override_prompt is None:
        prompts = []
        for test_sentence in test_sentences:
            prompts.append(construct_prompt(params, train_sentences, train_labels, test_sentence))
        # print('used prompts', prompts)
    else:
        prompts = override_prompt


    if 'prediction_mode' in params and params['prediction_mode'] == 'None':
        # This is just for a single query that has all the format already.
        num_tokens_to_predict = params['num_tokens_to_predict']
        resp = complete(prompts, num_tokens_to_predict, params['model'], num_log_probs=params['api_num_log_prob'], data_name=params["dataset"])
        return resp
    else:
        chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
    # print("chunked_prompts before", chunked_prompts)

    # Modifies this to make the zeroshot have the necessary format for query! Only works for zero shot prediction
    if 'prediction_mode' in params and params['prediction_mode'] == 'zeroshot':
        modified_chunked_prompts = []
        # Compose the single prompt
        for resp in chunked_prompts:
            # Format each prompt
            modified_chunked_prompts.append([f"{params['prompt_prefix']}{params['q_prefix']}{resp[0]}{params['a_prefix']}"])
        chunked_prompts = modified_chunked_prompts

    for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
        if num_tokens_to_predict_override is not None:
            num_tokens_to_predict = num_tokens_to_predict_override
        else:
            num_tokens_to_predict = params['num_tokens_to_predict']
        resp = complete(test_chunk_prompts, num_tokens_to_predict, params['model'], num_log_probs=params['api_num_log_prob'], data_name=params['dataset'])

        if 'vicuna7b' in params['model']:
            all_raw_answers.append(resp) #note that resp will be a list of is a dictionary
            # print("resp", resp)
        else:
            for answer_id, answer in enumerate(resp['choices']):
                all_raw_answers.append(answer)
    if return_all_prompts:
        return all_raw_answers, prompts
    else:
        return all_raw_answers





def write_data_to_file(file_name, data):
    with open(file_name, 'a') as f:
        f.write(json.dumps(data) + '\n')


def read_data_from_file(file_name):
    with open(file_name, 'r') as f:
        lines = f.readlines()
    return [json.loads(line) for line in lines]


def add_index(data):
    return {str(i): item for i, item in enumerate(data)}

# # Read data from file
# data = read_data_from_file('data.txt')
#
# # Add index
# indexed_data = add_index(data)
#
# print(indexed_data)


def get_model_response_anthropic(params, train_sentences, train_labels, test_sentences, client, return_all_prompts=False,
                       num_tokens_to_predict_override=None, override_prompt=None):
    
    # Actual response model with prompt
    responsed = False
    while not responsed:
        try:
            response = client.completion(prompt=f"{anthropic.HUMAN_PROMPT} Please output 0 (negative) or 1 (positive) in the <answer> tag. <sentence>{example_sentence[index[0]]}</sentence>\n<answer>{example_label[index[0]]}</answer>\n\n<sentence>{example_sentence[index[1]]}</sentence>\n<answer>{example_label[index[1]]}</answer>\n\n<sentence>{public_sentence[i]}</sentence>{anthropic.AI_PROMPT}\n<answer>", model="claude-v1", max_tokens_to_sample=l, temperature = temp)
            responsed = True
        except:
            print("claude does not respond")
            time.sleep(5)
    print(response["completion"])
    try:
        predictions[i] = int(response["completion"])
    except:
        #print(i, response["completion"])
        predictions[i] = -2


def load_pickle(params):
    # load saved results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    assert os.path.isfile(file_name), f"file does not exist: {file_name}"
    with open(file_name, 'rb') as file:
        data = pickle.load(file)
    print(f"Loaded data from {file_name}")
    return data

def save_pickle(params, data):
    # save results from model
    file_name = os.path.join(SAVE_DIR, f"{params['expr_name']}.pkl")
    if os.path.isfile(file_name):
        print("WARNING! overwriting existing saved files")
    with open(file_name, 'wb') as file:
        pickle.dump(data, file)
    print(f"Saved to {file_name}")
    return data

def print_results(tree, names=('Original Accuracy  ','Calibrated Accuracy')):
    # print out all results
    root = deepcopy(tree)
    for dataset in root.keys():
        print(f"\n\nDataset: {dataset}")
        models_node = root[dataset]
        for model in models_node.keys():
            print(f"\nModel: {model}")
            num_shots_node = models_node[model]
            for num_shots in num_shots_node.keys():
                accuracies = np.array(list(num_shots_node[num_shots].values()))
                accuracies_mean = np.mean(accuracies, axis=0)
                accuracies_low = np.min(accuracies, axis=0)
                accuracies_high = np.max(accuracies, axis=0)
                accuracies_std = np.std(accuracies, axis=0)

                print(f"\n{num_shots}-shot, {len(accuracies)} seeds")
                for i, (m, l, h, s) in enumerate(zip(accuracies_mean, accuracies_low, accuracies_high, accuracies_std)):
                    print(f"{names[i]} | Mean: {m:.4f}, Low: {l:.4f}, High: {h:.4f}, Std: {s:.4f}")
                print()

def load_results(params_list):
    # load saved results from model
    result_tree = dict()
    for params in params_list:
        saved_result = load_pickle(params)
        keys = [params['dataset'], params['model'], params['num_shots']]
        node = result_tree # root
        for k in keys:
            if not (k in node.keys()):
                node[k] = dict()
            node = node[k]
        node[params['seed']] = saved_result['accuracies']
    print_results(result_tree)

