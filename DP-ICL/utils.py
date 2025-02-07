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
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import tiktoken
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

CLAUDE_API_KEY = "" # NOTE : fill up with your own api key
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
    bs = params['bs']
    if bs is None:
        if 'gpt2' in params['model']:
            return 1
        else:
            assert params['model'] in ['ada', 'gpt-4-turbo', "babbage-002", 'curie', 'davinci', 'ada-beta', 'babbage-beta', 'curie-beta', 'davinci-beta']
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
    
    if return_index:
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
def setup_gpt2(model_name):
    # load the GPT-2 model
    global gpt2_model
    global gpt2_tokenizer
    if gpt2_model is None:
        print("Setting up GPT-2 model")
        gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
        gpt2_model.eval().cuda()
        
        gpt2_tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        # to batch generation, we pad on the left and mask those positions out.
        gpt2_tokenizer.padding_side = "left"
        gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
        gpt2_tokenizer.pad_token_id = gpt2_tokenizer.eos_token_id
        #gpt2_model.config.max_length = 2048
        gpt2_model.config.pad_token_id = gpt2_model.config.eos_token_id
        print("Finished")

def setup_gpt3():
    # get OpenAI access key
    with open(os.path.join(ROOT_DIR, 'openai_key.txt'), 'r') as f:
        key = f.readline().strip()
        openai.api_key = key


def complete_gpt2(prompt, l=10, model_name='gpt2-xl', num_log_probs=None, echo=False):
    ''' This function runs GPT-2 locally but places the outputs into an json that looks just like the one
     provided by the OpenAI API. '''
    if isinstance(prompt, str):
        prompt = [prompt] # the code below assumes a list
    input_ids = gpt2_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    
    # greedily generate l tokens
    if l > 0:
        # the generate function can handle left padded inputs automatically in HF
        # total_sequences is now the input + possible generated output
        total_sequences = gpt2_model.generate(input_ids=input_ids['input_ids'].cuda(), attention_mask=input_ids['attention_mask'].cuda(), max_length=l + len(input_ids['input_ids'][0]), do_sample=False)
    else:
        assert echo == True and l == 0
        total_sequences = input_ids['input_ids'].cuda()

    # they want the probs of the top tokens
    if num_log_probs is not None:
        # we are left padding, so we need to adjust the position IDs
        attention_mask = (total_sequences != 50256).float()
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        # get the logits for the context and the next l tokens
        logits = gpt2_model.forward(input_ids=total_sequences, attention_mask=attention_mask, position_ids=position_ids, return_dict=True).logits.detach().cpu()
        if not echo:
            # get the top tokens and probs for the generated l tokens
            probs = torch.softmax(logits[:,-l-1:], dim=2).cpu()
            #print(probs.shape)
        else:
            # get the top tokens and probs for the context and the generated l tokens
            probs = torch.softmax(logits, dim=2).cpu()
        top_probs, top_tokens = torch.topk(probs, k=num_log_probs)
        logprobs = torch.log(probs)
        top_log_probs = torch.log(top_probs)

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


def complete_gpt2(prompt, l=10, model_name='gpt2-xl', num_log_probs=None, echo=False):
    ''' This function runs GPT-2 locally but places the outputs into an json that looks just like the one
     provided by the OpenAI API. '''
    if isinstance(prompt, str):
        prompt = [prompt] # the code below assumes a list
    input_ids = gpt2_tokenizer.batch_encode_plus(prompt, return_tensors="pt", padding=True)
    
    # greedily generate l tokens
    if l > 0:
        # the generate function can handle left padded inputs automatically in HF
        # total_sequences is now the input + possible generated output
        #print(l + len(input_ids['input_ids'][0]))
        total_sequences = gpt2_model.generate(input_ids=input_ids['input_ids'].cuda(), attention_mask=input_ids['attention_mask'].cuda(), max_length=l + len(input_ids['input_ids'][0]), do_sample=False)
    else:
        assert echo == True and l == 0
        total_sequences = input_ids['input_ids'].cuda()

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
            time.sleep(1)
    return response

def complete(prompt, l, model, temp=0, num_log_probs=None, echo=False, n=None):
    """complete the prompt using a language model"""
    assert l >= 0
    assert temp >= 0
    if 'gpt2' in model:
        assert n == None # unsupported at the moment
        assert temp == 0 # unsupported at the moment
        setup_gpt2(model)
        return complete_gpt2(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo)
    elif 'gpt-4' in model :
        setup_gpt3()
        return complete_gpt4(prompt, l, model_name=model, temp=0, num_log_probs=num_log_probs)
    else:
        setup_gpt3()
        return complete_gpt3(prompt, l=l, model_name=model, num_log_probs=num_log_probs, echo=echo, n=n)

def complete_gpt4(prompt, l, model_name, temp=0, num_log_probs=None) :
    # call GPT-3 API until result is provided and then return it
    response = None
    received = False
    while not received:
        try:
            response = openai.ChatCompletion.create(model=model_name, 
                                                    messages=[{"role": "user", 
                                                    "content": prompt}],
                                                    max_tokens=l, 
                                                    temperature=temp, logprobs=True,
                                                    top_logprobs=num_log_probs)
            # print(prompt)
            # print(response)
            received = True
        except:
            error = sys.exc_info()[0]
            if error == openai.error.InvalidRequestError: # something is wrong: e.g. prompt too long
                print(f"InvalidRequestError\nPrompt passed in:\n\n{prompt}\n\n")
                assert False
            print("API error:", error)
            time.sleep(1)
    return response


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
            assert isinstance(l, str) # string labels
            assert params['task_format'] == 'qa'
            l_str = l

        prompt += a_prefix
        prompt += l_str + "\n\n"

    prompt += q_prefix
    prompt += test_sentence + "\n"
    assert a_prefix[-1] == ' '
    prompt += a_prefix[:-1] # GPT models do not want a trailing space, so we cut off -1
    #print(prompt)
    #print("---------------")
    return prompt

def get_model_response(params, train_sentences, train_labels, test_sentence, return_all_prompts=False,
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
        for train_examples, train_label in zip(train_sentences, train_labels):
            prompts.append(construct_prompt(params, train_examples, train_label, test_sentence))
        print(num_tokens_per_prompt(prompts))
    else:
        prompts = override_prompt

    # NOTE : maybe won't work with gpt4
    if 'gpt-4' in params['model'] :
        for prompt in prompts :
            if num_tokens_to_predict_override is not None:
                num_tokens_to_predict = num_tokens_to_predict_override
            else:
                num_tokens_to_predict = params['num_tokens_to_predict']
            resp = complete(prompt, num_tokens_to_predict, params['model'], num_log_probs=params['api_num_log_prob'])
            # print(resp)
            for answer_id, answer in enumerate(resp.choices):
                all_raw_answers.append(answer)
    else :
        chunked_prompts = list(chunks(prompts, chunk_size_helper(params)))
        for chunk_id, test_chunk_prompts in enumerate(chunked_prompts):
            if num_tokens_to_predict_override is not None:
                num_tokens_to_predict = num_tokens_to_predict_override
            else:
                num_tokens_to_predict = params['num_tokens_to_predict']
            resp = complete(test_chunk_prompts, num_tokens_to_predict, params['model'], num_log_probs=params['api_num_log_prob'])
            for answer_id, answer in enumerate(resp['choices']):
                all_raw_answers.append(answer)

    if return_all_prompts:
        return all_raw_answers, prompts
    else:
        return all_raw_answers

def num_tokens_per_prompt(prompts, encoding_name='cl100k_base'):
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = 0
    for prompt in prompts :
        num_tokens += len(encoding.encode(prompt))
    return num_tokens/len(prompts)


# NOTE  : Addition
def complete_anthropic(model, client, prompt, temperature, max_token_to_sample) :
    responsed = False
    while not responsed:
        try:
            response = client.completions.create(model=model, prompt=prompt, max_tokens_to_sample=max_token_to_sample, temperature=temperature)
            responsed = True
        except:
            print("claude does not respond")
            time.sleep(0.01)

    return response.completion


# NOTE  : Addition
def get_model_response_anthropic(params, train_sentences, train_labels, test_sentences, return_all_prompts=False,
                       num_tokens_to_predict_override=None, override_prompt=None) :
    
    client = setup_anthropic()
    # Prompt construction for claude
    predictions = []
    for test_sentence in test_sentences :
        prompt = f"{anthropic.HUMAN_PROMPT} Please output 0 ({params['label_dict'][0][0]}) or 1 ({params['label_dict'][1][0]}) in the <answer> tag."
        for i in range(params['num_shots']) :
            prompt += f"<sentence>{train_sentences[i]}</sentence>\n<answer>{train_labels[i]}</answer>\n\n"
        prompt += f"<sentence>{test_sentence}</sentence>{anthropic.AI_PROMPT}\n<answer>"
        completion = complete_anthropic(model=params['model'], client=client, prompt=prompt, temperature=0, max_token_to_sample=params['num_tokens_to_predict'])
        # print(completion)
        try:
            predictions.append(int(completion))
        except:
            predictions.append(-2)

    return predictions


def setup_anthropic() :
    client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)
    return client

