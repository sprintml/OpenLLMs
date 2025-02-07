import peft


from transformers import (
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
    BitsAndBytesConfig
)

import torch


from dp_transformers.grad_sample.transformers import conv_1d

def get_model(model_args, use_bart=False):
    bnb_config_4 = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
    
    if model_args.prefix:
        print("Prefix model chosen")
        if use_bart:
            print("For Seq2Seq")
            peft_config = peft.PrefixTuningConfig(task_type=peft.TaskType.SEQ_2_SEQ_LM, inference_mode=False, num_virtual_tokens=model_args.pre_seq_len, prefix_projection=model_args.prefix_projection)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
        else:
            print("For Causal LM")
            peft_config = peft.PrefixTuningConfig(task_type=peft.TaskType.CAUSAL_LM, inference_mode=False, num_virtual_tokens=model_args.pre_seq_len, prefix_projection=model_args.prefix_projection)
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, quantization_config=bnb_config_4 if model_args.loading_4_bit else None)
        model = peft.get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    elif model_args.lora:
        print("LORA model chosen")

        if use_bart:
            print("For Seq2Seq")
            peft_config = peft.LoraConfig(task_type=peft.TaskType.SEQ_2_SEQ_LM, inference_mode=False, r=8, lora_alpha=32, lora_dropout=0)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path)
        else:
            print("For Causal LM")
            print(f"Training {'WITH' if model_args.loading_4_bit else 'WITHOUT'} 4 bit loading")
            peft_config = peft.LoraConfig(task_type=peft.TaskType.CAUSAL_LM, inference_mode=False, r=8, lora_alpha=16, lora_dropout=0)
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, quantization_config=bnb_config_4 if model_args.loading_4_bit else None)    
        print(peft_config)
        model = peft.get_peft_model(model, peft_config).to("cuda")
        model.print_trainable_parameters()

    else:
        print("Full Fine Tuning")
        
        if use_bart:
            print("For Seq2Seq")
            model = AutoModelForSeq2SeqLM.from_pretrained(model_args.model_name_or_path).to("cuda")
            
        else:
            print("For Causal LM")
            model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path).to("cuda")

    return model

