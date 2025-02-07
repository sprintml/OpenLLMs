# PromptPATE Gen
The PromptPATEGen extends the promptPATE approach (https://arxiv.org/abs/2305.15594), for text generation tasks. Here, we leverage the privacy analysis of DPICL (https://arxiv.org/abs/2305.01639) approach.
Specifically, we generate privacy preserving student prompts based on keyword space aggregation (KSA) and propose-test-release (PTR) method and append it to the LLM.
By doing this, the LLM can answer unlimited amount of queries without incurring additional privacy costs per-query as opposed to the limited approach of DPICL.

## Install Dependencies
- `pip install -r requirements.txt`
  
[//]: # (# Run Code)

[//]: # (- `bash run_gen_experiment.sh`)

[//]: # (  - **Note:** You will have to name your environment 'llm-env' or remove it completely from the run_samsum.sh)

## Running PromptPATEGen
### Parameters

$model = {vicuna7b}

$dataset = {samsum, docvqa, mit-d, mit-g}

$seed = {Any random seed}


### Step 1: Obtain the prediction on the query from the teacher ensemble. Return the raw output from the teachers
    mkdir -p "./teacher_predictions/${dataset}"
    python3 run_generation.py
      --model $model
      --dataset $dataset 
      --seed $seed 
      --task_format 'summarization' 
      --num_shots 1 
      --subsample_test_set 100 
      --num_teachers 100 
      --enable_evaluation True 
      --save_path "./teacher_predictions/${dataset}/teacher_predictions.txt"


### Step 2:   Save all teacher's prediction
    python3 add_index.py
      --file_name "./teacher_predictions/${dataset}/teacher_predictions.txt"
      --seed $seed


### Step 3: Writing of student labels on txt file'
    python3 preprocess_true_labels.py
      --dataset $dataset
      --save_file "./teacher_predictions/${dataset}/true_student_labels.txt"
      --seed $seed
      --num_student_query 100

### Step 4: Obtain zero shot predictions
    python3 zero_shot_predictions.py
      --model $model
      --dataset $dataset
      --save_file "./teacher_predictions/${dataset}/zero_shot_predictions_student.txt"
      --num_tokens_to_predict 100
      --seed $seed
      --num_student_query 100
  

### Step 5: Save teacher's predictions
    python3 add_index.py
      --file_name "./teacher_predictions/${dataset}/zero_shot_predictions_student.txt"
      --seed $seed
  

### Step 6 : Writing of evaluation labels on txt file
    python3 preprocess_true_labels.py
      --dataset $dataset
      --save_file "./teacher_predictions/${dataset}/true_eval_labels.txt"
      --perform_evaluation
      --seed $seed
      --num_eval_query 10000
  

### Step 7: Obtain private keywords for querying the model
    python3 private_aggregation_ptr.py
        --teacher_predictions "./teacher_predictions/${dataset}/teacher_predictions.txt"
        --zero_shot_predictions "./teacher_predictions/${dataset}/zero_shot_predictions_student.txt"
        --true_labels "./teacher_predictions/${dataset}/true_student_labels.txt"\
        --public_labels_from_teachers_filename "./teacher_predictions/${dataset}/public_labels_from_teachers.txt"\
        --num_teachers 100
        --num_student_query 100
        --model $model
        --dataset $dataset
        --num_tokens_to_predict 100
        --epsilon $epsilon
        --seed $seed


### Step 8: Evaluation of each student prompts on test set and selection of best student prompt
      python3 run_evaluation_students.py
          --model $model
          --dataset $dataset
          --seed $seed
          --subsample_test_set 10000
          --public_labels_from_teachers_filename "./teacher_predictions/${dataset}/public_labels_from_teachers.txt"
          --num_tokens_to_predict 100
          --eval_labels_filename "./teacher_predictions/${dataset}/true_eval_labels.txt"\
          --num_student_prompts $num_student_prompts
          --epsilon $epsilon
          --use_pass_only_ptr_examples > "output_${dataset}_${seed}_eps_${epsilon}_${num_student_prompts}.txt"