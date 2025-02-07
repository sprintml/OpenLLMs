import torch
import os
import numpy as np
import argparse
from tqdm import tqdm
from transformers import RobertaForSequenceClassification, RobertaTokenizer, get_linear_schedule_with_warmup
from peft import LoraConfig, PromptTuningConfig, PrefixTuningConfig, TaskType, get_peft_model
import evaluate
from opacus import PrivacyEngine
from opacus.utils.batch_memory_manager import BatchMemoryManager


from util import *
from train import *
from dataloader import get_dataloader


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser(description='Prediction text generation sst2')

parser.add_argument('--model_checkpoint', default='FacebookAI/roberta-large', type=str)

parser.add_argument('--local_model', type=str, default='')

parser.add_argument('--task', type=str, default="sst2")

parser.add_argument('--batch_size', default=8, type=int)

parser.add_argument('--batch_size_eval', default=8, type=int)

parser.add_argument('--lr', default=5e-4, type=float)

parser.add_argument('--epochs', default=5, type=int)

parser.add_argument('--accumulation_steps', default=1, type=int)

parser.add_argument('--privacy', default=None, type=bool)

parser.add_argument('--num_labels', default=2, type=int)

parser.add_argument('--save_dir', type=str)

parser.add_argument('--testing_split', type=str)

parser.add_argument('--target_epsilon', type=float)

parser.add_argument('--target_delta', type=float)

parser.add_argument('--lr_scheduler', type=str)

parser.add_argument('--seed', type=int)

parser.add_argument('--finetuning_type', type=str, default=None, choices=['full', 'lora'])

parser.add_argument('--lora_rank', type=int, default=4)


def print_args(args):
    print('=' * 100)
    for k, v in args.__dict__.items():
        print('        - {} : {}'.format(k, v))
    print('=' * 100)


if __name__=='__main__' :

    args = parser.parse_args()
    print_args(args)

    # Setting training seed
    set_seed(args.seed)

    train_epoch_iterator = get_dataloader(args.task, args.model_checkpoint, "train", batch_size=args.batch_size)
    eval_epoch_iterator = get_dataloader(args.task, args.model_checkpoint, args.testing_split, batch_size=args.batch_size_eval)

    model = RobertaForSequenceClassification.from_pretrained(args.model_checkpoint, num_labels=args.num_labels,
                                                           ).to(device)
    tokenizer = RobertaTokenizer.from_pretrained(args.model_checkpoint, use_fast=True, revision='main')

    if args.finetuning_type == 'lora' :
        peft_config = LoraConfig(task_type=TaskType.SEQ_CLS, r=args.lora_rank, lora_alpha=16, lora_dropout=0.0)
        model = get_peft_model(model, peft_config)
        print('loratuning chosen')
    model.config.pad_token_id = model.config.eos_token_id

    # Load trained weights from possible previous trainings
    if args.local_model != '' and args.privacy :
        state_dict = torch.load(args.local_model)
        model.load_state_dict(state_dict)
    
    # Optimizer list optional here but can be used for lr customization for different models
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.lr_scheduler == 'linear' :
        lr_schedulers = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=args.epochs*len(train_epoch_iterator))
    elif args.lr_scheduler == 'constant' : 
        lr_schedulers = None
    metrics = evaluate.load('accuracy')

    if args.privacy : 
        model.train()    
        privacy_engine = PrivacyEngine()
        model, optimizer, train_epoch_iterator = privacy_engine.make_private_with_epsilon(
            module=model,
            optimizer=optimizer,
            data_loader=train_epoch_iterator,
            max_grad_norm=0.1,
            target_epsilon=args.target_epsilon,
            epochs=args.epochs,
            target_delta=args.target_delta
        )

    model = model.to(device)

    print(f"\nTraining for {args.task} begins in batches of {args.batch_size}.")
    for e in range(args.epochs):
        if args.privacy :
            with BatchMemoryManager(data_loader=train_epoch_iterator, max_physical_batch_size=4, optimizer=optimizer) as new_data_loader:
                print(f"\nEpoch {e+1}/{args.epochs} with BatchMemoryManager")
                tr_loss = 0
                global_steps = 0
                pbar = tqdm(new_data_loader, total=len(new_data_loader))
                # pbar = tqdm(train_epoch_iterator, total=len(train_epoch_iterator))

                accumulated_loss = 0
                accumulated_batch_count = 0

                for batch in pbar :
                    global_steps += 1

                    # slight modification of inputs preparation
                    inputs = prepare_inputs(batch, device)
                    if inputs is not None :
                        if args.accumulation_steps > 1 :
                            accumulated_batch_count, accumulated_loss = training_step(model, inputs, optimizer, lr_schedulers, args, accumulated_batch_count, accumulated_loss)
                            
                            if accumulated_batch_count != 0 :
                                # Update progress bar with the current average batch loss
                                batch_loss = accumulated_loss / (accumulated_batch_count * args.batch_size)
                                pbar.set_description(f"Task : {args.task} " + str(batch_loss), refresh=True)
                                pbar.update()

                        elif args.accumulation_steps == 1 :
                            step_loss = training_step(model, inputs, optimizer, lr_schedulers, args, accumulated_batch_count, accumulated_loss)
                            step_loss = step_loss.item()
                            tr_loss += step_loss

                            batch_loss = (tr_loss/(global_steps*args.batch_size))
                            pbar.set_description(f"Task : {args.task} " + str(batch_loss), refresh=True)
                            pbar.update()

        else :
            print(f"\nEpoch {e+1}/{args.epochs}")
            tr_loss = 0
            global_steps = 0
            pbar = tqdm(train_epoch_iterator, total=len(train_epoch_iterator))

            accumulated_loss = 0
            accumulated_batch_count = 0

            for batch in pbar :
                global_steps += 1

                inputs = prepare_inputs(batch, device)
                if args.accumulation_steps > 1 :
                    accumulated_batch_count, accumulated_loss = training_step(model, inputs, optimizer, lr_schedulers, args.accumulation_steps, accumulated_batch_count, accumulated_loss)
                    
                    if accumulated_batch_count != 0 :
                        # Update progress bar with the current average batch loss
                        batch_loss = accumulated_loss / (accumulated_batch_count * args.batch_size)
                        pbar.set_description(f"Task : {args.task} " + str(batch_loss), refresh=True)
                        pbar.update()

                elif args.accumulation_steps == 1 :
                    step_loss = training_step(model, inputs, optimizer, lr_schedulers, args.accumulation_steps, accumulated_batch_count, accumulated_loss)
                    step_loss = step_loss.item()
                    tr_loss += step_loss

                    batch_loss = (tr_loss/(global_steps*args.batch_size))
                    pbar.set_description(f"Task : {args.task} " + str(batch_loss), refresh=True)
                    pbar.update()

        # Evaluation
        all_predictions = []
        all_labels = []
        eval_loss = 0
        global_steps = 0
        pbar_eval = tqdm(eval_epoch_iterator, total=len(eval_epoch_iterator))
        for batch in pbar_eval:
            global_steps += 1
            inputs = prepare_inputs(batch, device)
            step_loss, predictions, labels = eval_step(model, inputs)
            eval_loss += step_loss.item()

            batch_loss_eval = (eval_loss/(global_steps*args.batch_size))
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        

        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        # metrics.add_batch(predictions=all_predictions, references=all_labels)
        eval_metric = metrics.compute(predictions=all_predictions, references=all_labels)
        print(f"Task : {args.task} " + 'accuracy : ' + str(eval_metric))

        
    if args.privacy :
        os.makedirs(os.path.join(args.save_dir, f'epoch_{e}'), exist_ok=True)  # Create the directory if it doesn't exist
        torch.save(model._module.state_dict(), os.path.join(args.save_dir, f"epoch_{e}/model_epoch_{e}.pth"))

    else : 
        model.save_pretrained(os.path.join(args.save_dir, f'epoch_{e}'))
        
