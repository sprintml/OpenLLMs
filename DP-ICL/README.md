# DP-ICL for text classification
This code is the re-implementation of the text classification part of the paper https://arxiv.org/abs/2305.01639 published at ICLR 2024 for the purpose of our contribution.

### To do before use

To access API OpenAI models, please create an api key and add it into a txt file called ```openai_api.txt``` directly into this repo.

### How to use

Here is an example script executing the 'main.py' file on the SST2 text classification task using GPT3-Babbage :
```python main.py \
    --model 'babbage-002' \
    --dataset 'sst2' \
    --seed 0 \
    --num_shots 4 \
    --subsample_test_set 872 \
    --approx \
    --target_delta 1e-5 \
    --noise_multiplier 3.0
```