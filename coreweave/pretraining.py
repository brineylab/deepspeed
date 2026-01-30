import os
import warnings
warnings.simplefilter('ignore')
from transformers import (
    EsmConfig,
    EsmTokenizer,
    EsmForMaskedLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
    HfArgumentParser,
)
from dataclasses import dataclass, field
from typing import Optional
from string import Template
from datasets import load_dataset
import wandb
from datetime import date
import yaml
import logging
import sys
import argparse

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    vocab_size: int = field(default=26)
    pad_token_id: int = field(default=21)
    mask_token_id: int = field(default=22)
    num_attention_heads: int = field(default=20)
    num_hidden_layers: int = field(default=32)
    hidden_size: int = field(default=960)
    intermediate_size: int = field(default=3840)
    max_position_embeddings: int = field(default=322)
    position_embedding_type: str = field(default="rotary")


@dataclass
class DataArguments:
    train_file: str = field(default=None)
    validation_file: str = field(default=None)
    file_type: str = field(default="json")
    tokenizer_path: str = field(default="facebook/esm2_t33_650M_UR50D")
    padding: str = field(default="max_length")
    max_length: int = field(default=512)
    truncation: bool = field(default=True)
    separator_token: str = field(default="<sep>")
    mlm: bool = field(default=True)
    mlm_probability: float = field(default=0.15)
    datasets: Optional[dict] = field(default=dict)
    separator: Optional[str] = field(default="<s>")
    num_processes: Optional[int] = field(default=64)
    cache_dir: Optional[str] = field(default="./.cache")


@dataclass
class CustomArguments:
    wandb_group: Optional[str] = field(default="esm-training")
    wandb_project: Optional[str] = field(default="esm-training")


def load_yaml_config(config_path: str) -> dict:
    if not os.path.exists(config_path):
        logger.warning(f"Config file not found: {config_path}")
        return {}

    logger.info(f"Loading config from: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    substitutions = {
        **config,
        "date": f"{date.today().isoformat()}_{os.getpid()}",
    }

    def apply_template_substitution(value, subs):
        if isinstance(value, str):
            prev = None
            while value != prev:
                prev = value
                value = Template(value).safe_substitute(subs)
            return value
        return value

    config = {
        k: apply_template_substitution(v, substitutions) for k, v in config.items()
    }
    return config


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", type=str, required=True, help="Path to the config YAML file.")
    args, overrides = parser.parse_known_args()
    config = load_yaml_config(args.config_file)
    hf_parser = HfArgumentParser((TrainingArguments, ModelArguments, DataArguments, CustomArguments))
    training_args, model_args, data_args, custom_args = hf_parser.parse_dict(
        config,
        allow_extra_keys=False
    )
    if overrides:
        for action in hf_parser._actions:
            action.required = False
            action.default = argparse.SUPPRESS

        namespace, remaining_args = hf_parser.parse_known_args(overrides)
        if remaining_args:
            raise ValueError(f"Some specified arguments are not used by the HfArgumentParser: {remaining_args}")

        for key, value in vars(namespace).items():
            for dataclass_instance in [training_args, model_args, data_args, custom_args]:
                if hasattr(dataclass_instance, key):
                    setattr(dataclass_instance, key, value)
                    break

    return training_args, model_args, data_args, custom_args


def preprocess_dataset(
    sequence,
    tokenizer,
    separator,
    padding = "max_length",
    truncation = True,
    max_len = 320
) -> list:

    # reformat sequences with separator
    paired_sequence = sequence["sequence_aa"]
    # paired_sequence = sequence["sequence_aa_heavy"] + separator + sequence["sequence_aa_light"]

    # tokenize
    tokenized = tokenizer(paired_sequence,
                          padding = padding,
                          max_length = max_len,
                          truncation = truncation)

    # special tokens mask - tokenizer does not account for special tokens already present
    tokenized['special_tokens_mask'] = tokenizer.get_special_tokens_mask(tokenized['input_ids'], already_has_special_tokens=True)

    return tokenized


def load_and_tokenize(data_args, tokenizer):

    # read datasets into huggingface dataset
    dataset = load_dataset(
        data_args.file_type,
        data_files = data_args.datasets
    )

    # preprocess and tokenize
    tokenized_dataset = dataset.map(
        preprocess_dataset,
        fn_kwargs={
            "tokenizer": tokenizer,
            "padding": data_args.padding,
            "max_len": data_args.max_length,
            "truncation": data_args.truncation,
            "separator": data_args.separator_token
        },
        num_proc=data_args.num_processes,
        cache_file_names={k: f"{data_args.cache_dir}/{str(k)}.arrow" for k in dataset},
    )

    return tokenized_dataset


def create_esm_model_config(model_args: ModelArguments) -> EsmConfig:
    config = EsmConfig(
        vocab_size=model_args.vocab_size,
        pad_token_id=model_args.pad_token_id,
        mask_token_id=model_args.mask_token_id,
        num_attention_heads=model_args.num_attention_heads,
        num_hidden_layers=model_args.num_hidden_layers,
        hidden_size=model_args.hidden_size,
        intermediate_size=model_args.intermediate_size,
        max_position_embeddings=model_args.max_position_embeddings,
        position_embedding_type=model_args.position_embedding_type,
    )
    return config


def setup_wandb(custom_args: CustomArguments):
    logger.info("Initializing wandb...")
    wandb.login()
    os.environ["WANDB_PROJECT"] = custom_args.wandb_project


def main():
    # Parse all arguments from config and CLI
    training_args, model_args, data_args, custom_args = parse_arguments()

    # Generate run name
    training_args.run_name = f"{training_args.run_name}_{date.today().isoformat()}" if training_args.run_name else f"esm_train_{date.today().isoformat()}"

    # Log all configuration
    logger.info(f"Run name: {training_args.run_name}")
    logger.info(f"Model arguments: {model_args}")
    logger.info(f"Data arguments: {data_args}")
    logger.info(f"Custom arguments: {custom_args}")
    logger.info(f"Training arguments: {training_args}")

    # Load tokenizer
    logger.info(f"Loading tokenizer from: {data_args.tokenizer_path}")
    tokenizer = EsmTokenizer.from_pretrained(data_args.tokenizer_path)

    # Prepare datasets
    tokenized_datasets = load_and_tokenize(data_args, tokenizer)

    # Create model configuration
    logger.info("Creating model config...")
    model_config = create_esm_model_config(model_args)

    # Create data collator for masked language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=data_args.mlm,
        mlm_probability=data_args.mlm_probability
    )

    # Setup wandb for experiment tracking
    if custom_args.wandb_project:
        setup_wandb(custom_args)

    # Initialize model
    logger.info("Creating model...")
    model = EsmForMaskedLM(model_config)

    model_size = sum(p.numel() for p in model.parameters())
    logger.info(f"Model size: {model_size/1e6:.2f}M parameters")

    # Create trainer and start training
    logger.info("Starting training...")
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets["train"],
        eval_dataset = {
            "real_paired": tokenized_datasets["real_eval"],
            "syn_paired": tokenized_datasets["syn_eval"],
            # "random_paired": tokenized_datasets["random_eval"],
        },
    )
    trainer.train()

    # Save final model
    logger.info(f"Saving model to: ./models/{training_args.run_name}")
    trainer.save_model(f"./models/{training_args.run_name}")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
