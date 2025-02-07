from typing import Any, Dict, Union
import torch
import torch.nn as nn 
import os

def compute_loss(model, inputs):
    """
    Pass the inputs into the model, computes the loss and the predictions if the model is in evaluation mode.

    Args:
        model (:obj:`nn.Module`):
            The model to evaluate.
        inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
    
    Return:
        Union[
            (
                (loss (:obj:`torch.Tensor`), predictions (:obj:`torch.Tensor`), labels (:obj:`torch.Tensor`)),
                (loss (:obj:`torch.Tensor`)
            )       
        ]
    """

    # Passing through the model
    outputs = model(**inputs)

    # maybe outputs.logits
    logits = outputs["logits"]
    loss_fn = nn.CrossEntropyLoss()
    loss = loss_fn(logits, inputs["labels"])
    
    if model.training :
        return loss
    else :
        predictions = outputs.logits.argmax(dim=-1)
        return loss, predictions, inputs["labels"]


def training_step(model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]],
                  optimizer: torch.optim.Optimizer, lr_scheduler: torch.optim.lr_scheduler,
                  args, accumulated_batch_count, accumulated_loss) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (:obj:`nn.Module`):
            The model to train.
        inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
        optimizer (torch.optim.Optimizer): 
            Optimizer instance for the training loop.
        lr_scheduler (torch.optim.lr_scheduler): 
            LR scheduler instance for the training loop.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument :obj:`labels`. Check your model's documentation for all accepted arguments.

    Return:
        :obj:`torch.Tensor`: The tensor with training loss on this batch.
    """
    model.train()
    loss = compute_loss(model, inputs)

    if args.accumulation_steps == 1 :
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if args.lr_scheduler != 'constant' :
            lr_scheduler.step()

        return loss.detach()
    
    elif args.accumulation_steps > 1 :
        accumulated_loss += loss
        loss.backward()

        accumulated_batch_count += 1
        if accumulated_batch_count == args.accumulation_steps :
            optimizer.step()
            if args.lr_scheduler != 'constant' :
                lr_scheduler.step()
            optimizer.zero_grad()
            accumulated_batch_count = 0
            accumulated_loss = 0

        return accumulated_batch_count, accumulated_loss


def eval_step(model: torch.nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
    """
    Perform a training step on a batch of inputs.

    Subclass and override to inject custom behavior.

    Args:
        model (:obj:`nn.Module`):
            The model to train.
        inputs (:obj:`Dict[str, Union[torch.Tensor, Any]]`):
            The inputs and targets of the model.
        optimizer (torch.optim.Optimizer): 
            Optimizer instance for the training loop.
        lr_scheduler (torch.optim.lr_scheduler): 
            LR scheduler instance for the training loop.

            The dictionary will be unpacked before being fed to the model. Most models expect the targets under the
            argument :obj:`labels`. Check your model's documentation for all accepted arguments.

    Return:
        loss (:obj:`torch.Tensor`),
        predictions (:obj:`torch.Tensor`),
        labels (:obj:`torch.Tensor`)
    """
    model.eval()
    model.zero_grad()
    loss, predictions, labels = compute_loss(model, inputs)


    return loss.detach(), predictions, labels


def checkpoint(model, args, e, accuracies) :
    if args.privacy :
        os.makedirs(os.path.join(args.save_dir, f'epoch_{e}'), exist_ok=True)  # Create the directory if it doesn't exist
        torch.save(model._module.state_dict(), os.path.join(args.save_dir, f"epoch_{e}/model_epoch_{e}.pth"))

    else : 
        model.save_pretrained(os.path.join(args.save_dir, f'epoch_{e}'))