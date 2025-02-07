bs=20
lr=5e-5
epoch=10
max_seq_len=128
seed=11

# lmsys/vicuna-7b-v1.5
# openlm-research/open_llama_13b
# EleutherAI/pythia-6.9b
# EleutherAI/pythia-1b
# EleutherAI/pythia-410m
# EleutherAI/pythia-70m
python3 run.py \
  --model_name_or_path "EleutherAI/pythia-1b" \
  --task_name "mitmovie" \
  --dataset_name "mitmovie" \
  --max_seq_length $max_seq_len \
  --do_train \
  --do_eval \
  --evaluation_strategy steps \
  --eval_steps 500 \
  --per_device_train_batch_size $bs \
  --learning_rate $lr \
  --num_train_epochs $epoch \
  --output_dir "./checkpoints/MITMovie/pythia_1b_noDP_LoRA" \
  --overwrite_output_dir \
  --seed $seed \
  --save_strategy no \
  --disable_dp True \
  --prefix False \
  --lora True \
  --logging_step 100 \
  --mit_task "genre" \
  --loading_4_bit True \
  --constant_scheduler False \
  