import os
import warnings
warnings.simplefilter('ignore')
from transformers import (
    RobertaConfig, 
    RobertaTokenizer, 
    RobertaForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)
import torch
from datasets import load_dataset
import argparse
import pathlib
import wandb
from datetime import date
import yaml

def parser():
    parser = argparse.ArgumentParser()
    
    # train argument parser
    parser.add_argument(
        "--train_config",
        default=None,
        required=True,
        type=pathlib.Path,
        help="yaml file containing training arguments is required!",
    )
    
    args = parser.parse_args()
    return args

# this config setup is specific for the folding model
def define_config(train_config, tokenizer):

    config = RobertaConfig(
        vocab_size=train_config.get("vocab_size", 25),
        num_attention_heads=train_config.get("num_attention_heads", 16),
        num_hidden_layers=train_config.get("num_hidden_layers", 24),
        hidden_size=train_config.get("hidden_size", 1024),
        intermediate_size=train_config.get("intermediate_size", 4096),
        max_len=train_config.get("max_len", 512), 
        max_position_embeddings=train_config.get("max_position_embeddings", 514),
        type_vocab_size=train_config.get("type_vocab_size", 2)
    )
    return config


def define_args(train_config):
    run_name = train_config.get("run_name") + f"_{date.today().isoformat()}"
    
    # setup training arguments
    training_args = TrainingArguments(
        fp16=train_config.get("fp16", True),
        evaluation_strategy=train_config.get("evaluation_strategy", "steps"),
        seed=train_config.get("seed", 42),
        per_device_train_batch_size=train_config.get("batch_size",32),
        per_device_eval_batch_size=train_config.get("batch_size",32),
        max_steps=train_config.get("max_steps", 500000),
        save_steps=train_config.get("save_steps", 50000),
        logging_steps=train_config.get("logging_steps", 100),
        eval_steps=train_config.get("eval_steps", 25000),
        adam_beta1=train_config.get("adam_beta1", 0.9),
        adam_beta2=train_config.get("adam_beta2", 0.98),
        adam_epsilon=train_config.get("adam_epsilon", 1e-6),
        weight_decay=train_config.get("weight_decay", 0.01),
        warmup_steps=train_config.get("warmup_steps", 30000),
        learning_rate=train_config.get("peak_learning_rate", 4e-4),
        gradient_accumulation_steps=train_config.get("gradient_accumulation_steps", 1),

        # output and logging
        output_dir=train_config.get("output_dir", f"../checkpoints/{run_name}"),
        overwrite_output_dir=train_config.get("overwrite_output_dir", True),
        logging_dir=train_config.get("logging_dir", f"../wandb/{run_name}"),
        report_to=train_config.get("report_to", None),
        run_name=run_name,
        logging_first_step=train_config.get("logging_first_step", True),
    )
    
    return training_args

def load_and_tokenize(train_config, tokenizer):
    data_files = {
        "train": [train_config.get("train_file")],
        "eval": [train_config.get("validation_file")],
    }
    dataset = load_dataset(train_config.get("file_type"), data_files=data_files)

    tokenized_dataset = dataset.map(
        lambda x: tokenizer(
            x[train_config.get("sequence_column")],
            padding=train_config.get("padding", "max_length"),
            truncation=train_config.get("truncation", True),
            max_length=train_config.get("max_length", 512),
            add_special_tokens=train_config.get("add_special_tokens", True),
            return_special_tokens_mask=train_config.get("return_special_tokens_mask", True),
        ),
        remove_columns=[train_config.get("sequence_column")],
        num_proc=50
    )
    return tokenized_dataset

def main():
    # parse cl args
    args = parser()
    with open(args.train_config, 'r') as stream:
        train_config = yaml.safe_load(stream)

    # tokenize
    tokenizer = RobertaTokenizer.from_pretrained(train_config.get("tokenizer_path"))
    tokenized_dataset = load_and_tokenize(train_config, tokenizer)

    # define model config
    model_config = define_config(train_config, tokenizer)

    # define training args
    training_args = define_args(train_config)
    
    # collator
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=train_config.get("mlm", True), 
        mlm_probability=train_config.get("mlm_probability", 0.15)
    )
    
    # wandb
    # don't call wandb.init() -> let Trainer call it automatically,
    # otherwise multiple runs will be initilized
    wandb.login()
    os.environ["WANDB_PROJECT"] = train_config.get("wandb_project")
    
    # model
    model = RobertaForMaskedLM(model_config)

    model_size = sum(p.numel() for p in model.parameters())
    print(f"Model size: {model_size/1e6:.2f}M")
    
    # train
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=collator,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["eval"]
    )
    trainer.train()
    
    # save and end
    trainer.save_model(f"./models/{run_name}")
    wandb.finish()
    
if __name__ == "__main__":
    main()